import collections
import copy
from logging import getLogger
from itertools import chain
import numpy as np
import torch, time
from torch import nn
from torch.nn import functional as F
from os.path import join, exists
import pfrl
import random
from pfrl.agent import AttributeSavingMixin, BatchAgent
from pfrl.replay_buffer import ReplayUpdater, batch_experiences, batch_recurrent_experiences
from pfrl.utils import clip_l2_grad_norm_
from pfrl.utils.batch_states import batch_states
from pfrl.utils.copy_param import synchronize_parameters
from pfrl.utils.mode_of_distribution import mode_of_distribution
from torch import distributions
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from pfrl.utils.recurrent import (
    get_recurrent_state_at,
    mask_recurrent_state_at,
    one_step_forward,
    pack_and_forward,
    recurrent_state_as_numpy,
    recurrent_state_from_numpy
)
# import torch_xla
# import torch_xla.core.xla_model as xm


def _mean_or_nan(xs):
    """Return its mean a non-empty sequence, numpy.nan for a empty one."""
    return np.mean(xs) if xs else np.nan


class TemperatureHolder(nn.Module):
    """Module that holds a temperature as a learnable value.

    Args:
        initial_log_temperature (float): Initial value of log(temperature).
    """

    def __init__(self, initial_log_temperature=0):
        super().__init__()
        self.log_temperature = nn.Parameter(
            torch.tensor(initial_log_temperature, dtype=torch.float32)
        )

    def forward(self):
        """Return a temperature as a torch.Tensor."""
        return torch.exp(self.log_temperature)


class SoftActorCritic(AttributeSavingMixin, BatchAgent):
    """Soft Actor-Critic (SAC).

    See https://arxiv.org/abs/1812.05905

    Args:
        policy (Policy): Policy.
        q_func1 (Module): First Q-function that takes state-action pairs as input
            and outputs predicted Q-values.
        q_func2 (Module): Second Q-function that takes state-action pairs as
            input and outputs predicted Q-values.
        policy_optimizer (Optimizer): Optimizer setup with the policy
        q_func1_optimizer (Optimizer): Optimizer setup with the first
            Q-function.
        q_func2_optimizer (Optimizer): Optimizer setup with the second
            Q-function.
        replay_buffer (ReplayBuffer): Replay buffer
        gamma (float): Discount factor
        gpu (int): GPU device id if not None nor negative.
        replay_start_size (int): if the replay buffer's size is less than
            replay_start_size, skip update
        minibatch_size (int): Minibatch size
        update_interval (int): Model update interval in step
        phi (callable): Feature extractor applied to observations
        soft_update_tau (float): Tau of soft target update.
        logger (Logger): Logger used
        batch_states (callable): method which makes a batch of observations.
            default is `pfrl.utils.batch_states.batch_states`
        burnin_action_func (callable or None): If not None, this callable
            object is used to select actions before the model is updated
            one or more times during training.
        initial_temperature (float): Initial temperature value. If
            `entropy_target` is set to None, the temperature is fixed to it.
        entropy_target (float or None): If set to a float, the temperature is
            adjusted during training to match the policy's entropy to it.
        temperature_optimizer_lr (float): Learning rate of the temperature
            optimizer. If set to None, Adam with default hyperparameters
            is used.
        act_deterministically (bool): If set to True, choose most probable
            actions in the act method instead of sampling from distributions.
    """

    saved_attributes = (
        "policy",
        "q_func1",
        "q_func2",
        "target_q_func1",
        "target_q_func2",
        "policy_optimizer",
        "q_func1_optimizer",
        "q_func2_optimizer",
        "temperature_holder",
        "temperature_optimizer",
    )

    def __init__(
        self,
        policy,
        q_func1,
        q_func2,
        policy_optimizer,
        q_func1_optimizer,
        q_func2_optimizer,
        replay_buffer,
        gamma,
        gpu=None,
        replay_start_size=10000,
        minibatch_size=100,
        update_interval=1,
        phi=lambda x: x,
        soft_update_tau=5e-3,
        max_grad_norm=None,
        logger=getLogger(__name__),
        batch_states=batch_states,
        burnin_action_func=None,
        initial_temperature=1.0,
        entropy_target=None,
        temperature_optimizer_lr=None,
        act_deterministically=True,
    ):

        self.policy = policy
        self.q_func1 = q_func1
        self.q_func2 = q_func2
        self.countuh = 0

        if gpu is not None and gpu >= 0:
            assert torch.cuda.is_available()
            self.device = torch.device("cuda:{}".format(gpu))
            self.policy.to(self.device)
            self.q_func1.to(self.device)
            self.q_func2.to(self.device)
        else:
            self.device = torch.device("cpu")

        self.replay_buffer = replay_buffer
        self.gamma = gamma
        self.gpu = gpu
        self.phi = phi
        self.soft_update_tau = soft_update_tau
        self.logger = logger
        self.policy_optimizer = policy_optimizer
        self.q_func1_optimizer = q_func1_optimizer
        self.q_func2_optimizer = q_func2_optimizer
        self.replay_updater = ReplayUpdater(
            replay_buffer=replay_buffer,
            update_func=self.update,
            batchsize=minibatch_size,
            n_times_update=1,
            replay_start_size=replay_start_size,
            update_interval=update_interval,
            episodic_update=False,
        )
        self.max_grad_norm = max_grad_norm
        self.batch_states = batch_states
        self.burnin_action_func = burnin_action_func
        self.initial_temperature = initial_temperature
        self.entropy_target = entropy_target
        if self.entropy_target is not None:
            self.temperature_holder = TemperatureHolder(
                initial_log_temperature=np.log(initial_temperature)
            )
            if temperature_optimizer_lr is not None:
                self.temperature_optimizer = torch.optim.Adam(
                    self.temperature_holder.parameters(), lr=temperature_optimizer_lr
                )
            else:
                self.temperature_optimizer = torch.optim.Adam(
                    self.temperature_holder.parameters()
                )
            if gpu is not None and gpu >= 0:
                self.temperature_holder.to(self.device)
        else:
            self.temperature_holder = None
            self.temperature_optimizer = None
        self.act_deterministically = act_deterministically

        self.t = 0

        # Target model
        self.target_q_func1 = copy.deepcopy(self.q_func1).eval().requires_grad_(False)
        self.target_q_func2 = copy.deepcopy(self.q_func2).eval().requires_grad_(False)

        # Statistics
        self.q1_record = collections.deque(maxlen=1000)
        self.q2_record = collections.deque(maxlen=1000)
        self.entropy_record = collections.deque(maxlen=1000)
        self.q_func1_loss_record = collections.deque(maxlen=100)
        self.q_func2_loss_record = collections.deque(maxlen=100)
        self.n_policy_updates = 0

        # tensor = torch.zeros(70)
        # tensor = tensor.to('cuda:0')
        # print(tensor.size())
        # self.weights = self.policy(tensor)

    @property
    def temperature(self):
        if self.entropy_target is None:
            return self.initial_temperature
        else:
            with torch.no_grad():
                return float(self.temperature_holder())

    def sync_target_network(self):
        """Synchronize target network with current network."""
        synchronize_parameters(
            src=self.q_func1,
            dst=self.target_q_func1,
            method="soft",
            tau=self.soft_update_tau,
        )
        synchronize_parameters(
            src=self.q_func2,
            dst=self.target_q_func2,
            method="soft",
            tau=self.soft_update_tau,
        )

    def update_q_func(self, batch):
        """Compute loss for a given Q-function."""

        batch_next_state = batch["next_state"]
        batch_rewards = batch["reward"]
        batch_terminal = batch["is_state_terminal"]
        batch_state = batch["state"]
        batch_actions = batch["action"]
        batch_discount = batch["discount"]

        with torch.no_grad(), pfrl.utils.evaluating(self.policy), pfrl.utils.evaluating(
            self.target_q_func1
        ), pfrl.utils.evaluating(self.target_q_func2):
            next_action_distrib = self.policy(batch_next_state)
            next_actions = next_action_distrib.sample()
            next_log_prob = next_action_distrib.log_prob(next_actions)
            next_q1 = self.target_q_func1((batch_next_state, next_actions))
            next_q2 = self.target_q_func2((batch_next_state, next_actions))
            next_q = torch.min(next_q1, next_q2)
            entropy_term = self.temperature * next_log_prob[..., None]
            assert next_q.shape == entropy_term.shape

            target_q = batch_rewards + batch_discount * (
                1.0 - batch_terminal
            ) * torch.flatten(next_q - entropy_term)
        
        predict_q1 = torch.flatten(self.q_func1((batch_state, batch_actions)))
        predict_q2 = torch.flatten(self.q_func2((batch_state, batch_actions)))

        loss1 = 0.5 * F.mse_loss(target_q, predict_q1)
        loss2 = 0.5 * F.mse_loss(target_q, predict_q2)

        # Update stats
        self.q1_record.extend(predict_q1.detach().cpu().numpy())
        self.q2_record.extend(predict_q2.detach().cpu().numpy())
        self.q_func1_loss_record.append(float(loss1))
        self.q_func2_loss_record.append(float(loss2))

        self.q_func1_optimizer.zero_grad()
        loss1.backward()
        if self.max_grad_norm is not None:
            clip_l2_grad_norm_(self.q_func1.parameters(), self.max_grad_norm)
        self.q_func1_optimizer.step()

        self.q_func2_optimizer.zero_grad()
        loss2.backward()
        if self.max_grad_norm is not None:
            clip_l2_grad_norm_(self.q_func2.parameters(), self.max_grad_norm)
        self.q_func2_optimizer.step()

    def update_temperature(self, log_prob):
        assert not log_prob.requires_grad
        loss = -torch.mean(self.temperature_holder() * (log_prob + self.entropy_target))
        self.temperature_optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            clip_l2_grad_norm_(self.temperature_holder.parameters(), self.max_grad_norm)
        self.temperature_optimizer.step()

    def update_policy_and_temperature(self, batch):
        """Compute loss for actor."""

        batch_state = batch["state"]

        action_distrib = self.policy(batch_state)
        actions = action_distrib.rsample()
        log_prob = action_distrib.log_prob(actions)
        q1 = self.q_func1((batch_state, actions))
        q2 = self.q_func2((batch_state, actions))
        q = torch.min(q1, q2)

        entropy_term = self.temperature * log_prob[..., None]
        assert q.shape == entropy_term.shape
        loss = torch.mean(entropy_term - q)

        self.policy_optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            clip_l2_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy_optimizer.step()

        self.n_policy_updates += 1

        if self.entropy_target is not None:
            self.update_temperature(log_prob.detach())

        # Record entropy
        with torch.no_grad():
            try:
                self.entropy_record.extend(
                    action_distrib.entropy().detach().cpu().numpy()
                )
            except NotImplementedError:
                # Record - log p(x) instead
                self.entropy_record.extend(-log_prob.detach().cpu().numpy())

    def update(self, experiences, errors_out=None):
        """Update the model from experiences"""
        batch = batch_experiences(experiences, self.device, self.phi, self.gamma)
        self.update_q_func(batch)
        self.update_policy_and_temperature(batch)
        self.sync_target_network()

    def batch_select_greedy_action(self, batch_obs, deterministic=False):
        with torch.no_grad(), pfrl.utils.evaluating(self.policy):
            batch_xs = self.batch_states(batch_obs, self.device, self.phi)
            policy_out = self.policy(batch_xs)
            mypolicy =list(self.policy.children())
            if deterministic:
                batch_action = mode_of_distribution(policy_out).cpu().numpy()                
                keep = mypolicy[0](batch_xs)
                keep = mypolicy[1](keep)
                keep = mypolicy[2](keep)
                keep = mypolicy[3](keep)
                keep = mypolicy[4](keep)
            else:
                batch_action = policy_out.sample().cpu().numpy()
            
            action = torch.tensor(batch_action)
            action = action.to('cuda:0')

            # q1 = self.target_q_func1((batch_xs, action))
            # q2 = self.target_q_func2((batch_xs, action))
            # q = torch.min(q1, q2)
            # [q] = q.cpu().numpy()
            # [q] = q
            # # print("Q", q)
            # bathcact = keep.cpu().numpy()
            # [bt] = bathcact
            # data_dir = '/home/spyd66/tSNE/Pullover/Pullover_RHP7_HARD_DR'
            # np.savez(join(data_dir, 'step{}'.format(self.countuh)),
            #          q_values=np.array(q),
            #          hidden_states=np.array(bt))
            # self.countuh += 1
            # f = open('/home/spyd66/tSNE/PUBASEDR_RHP7_TSNEQHS.txt','a') # Prints Q with HS Values
            # f.write(str(q))
            # f.write(" ")
            # for hidden in bt:
            #   f.write(str(hidden))
            #   f.write(" ")
            # f.write("\n")
            # f.close()

            # f = open('/home/spyd66/tSNE/FINAL2DRTSNEAHS.txt','a') # Prints Action with HS Values
            # [ba] = batch_action
            # f.write(str(q))
            # f.write(" ")
            # for act in ba:
            #   f.write(str(act))
            #   f.write(" ")
            # f.write("\n")
            # f.close()
            
        return batch_action

    def batch_act(self, batch_obs):
        if self.training:
            return self._batch_act_train(batch_obs)
        else:
            return self._batch_act_eval(batch_obs)

    def batch_observe(self, batch_obs, batch_reward, batch_done, batch_reset):
        if self.training:
            self._batch_observe_train(batch_obs, batch_reward, batch_done, batch_reset)

    def _batch_act_eval(self, batch_obs):
        assert not self.training
        return self.batch_select_greedy_action(
            batch_obs, deterministic=self.act_deterministically
        )

    def _batch_act_train(self, batch_obs):
        assert self.training
        if self.burnin_action_func is not None and self.n_policy_updates == 0:
            batch_action = [self.burnin_action_func() for _ in range(len(batch_obs))]
        else:
            batch_action = self.batch_select_greedy_action(batch_obs)
        self.batch_last_obs = list(batch_obs)
        self.batch_last_action = list(batch_action)

        return batch_action

    def _batch_observe_train(self, batch_obs, batch_reward, batch_done, batch_reset):
        assert self.training
        for i in range(len(batch_obs)):
            self.t += 1
            if self.batch_last_obs[i] is not None:
                assert self.batch_last_action[i] is not None
                # Add a transition to the replay buffer
                self.replay_buffer.append(
                    state=self.batch_last_obs[i],
                    action=self.batch_last_action[i],
                    reward=batch_reward[i],
                    next_state=batch_obs[i],
                    next_action=None,
                    is_state_terminal=batch_done[i],
                    env_id=i,
                )
                if batch_reset[i] or batch_done[i]:
                    self.batch_last_obs[i] = None
                    self.batch_last_action[i] = None
                    self.replay_buffer.stop_current_episode(env_id=i)
            self.replay_updater.update_if_necessary(self.t)

    def get_statistics(self):
        return [
            ("average_q1", _mean_or_nan(self.q1_record)),
            ("average_q2", _mean_or_nan(self.q2_record)),
            ("average_q_func1_loss", _mean_or_nan(self.q_func1_loss_record)),
            ("average_q_func2_loss", _mean_or_nan(self.q_func2_loss_record)),
            ("n_updates", self.n_policy_updates),
            ("average_entropy", _mean_or_nan(self.entropy_record)),
            ("temperature", self.temperature),
        ]

class MTSoftActorCritic(AttributeSavingMixin, BatchAgent):
    """Soft Actor-Critic (SAC).

    See https://arxiv.org/abs/1812.05905

    Args:
        policy (Policy): Policy.
        q_func1 (Module): First Q-function that takes state-action pairs as input
            and outputs predicted Q-values.
        q_func2 (Module): Second Q-function that takes state-action pairs as
            input and outputs predicted Q-values.
        policy_optimizer (Optimizer): Optimizer setup with the policy
        q_func1_optimizer (Optimizer): Optimizer setup with the first
            Q-function.
        q_func2_optimizer (Optimizer): Optimizer setup with the second
            Q-function.
        replay_buffer (ReplayBuffer): Replay buffer
        gamma (float): Discount factor
        gpu (int): GPU device id if not None nor negative.
        replay_start_size (int): if the replay buffer's size is less than
            replay_start_size, skip update
        minibatch_size (int): Minibatch size
        update_interval (int): Model update interval in step
        phi (callable): Feature extractor applied to observations
        soft_update_tau (float): Tau of soft target update.
        logger (Logger): Logger used
        batch_states (callable): method which makes a batch of observations.
            default is `pfrl.utils.batch_states.batch_states`
        burnin_action_func (callable or None): If not None, this callable
            object is used to select actions before the model is updated
            one or more times during training.
        initial_temperature (float): Initial temperature value. If
            `entropy_target` is set to None, the temperature is fixed to it.
        entropy_target (float or None): If set to a float, the temperature is
            adjusted during training to match the policy's entropy to it.
        temperature_optimizer_lr (float): Learning rate of the temperature
            optimizer. If set to None, Adam with default hyperparameters
            is used.
        act_deterministically (bool): If set to True, choose most probable
            actions in the act method instead of sampling from distributions.
    """

    saved_attributes = (
        "policy1",
        "shared_q_critic",
        "shared_layer_critic",
        "shared_q_actor",
        "shared_layer_actor",
        "q_func1_T1",        
        "q_func2_T1",        
        "target_q_func1_T1",
        "target_q_func2_T1",        
        "policy_optimizer1",        
        "shared_q_optimizer_critic",
        "shared_q_optimizer_actor",
        "q_func1_optimizer1",        
        "q_func2_optimizer1",        
        "temperature_holder1",        
        "temperature_optimizer1",        
    )

    def __init__(
        self,
        policy1,        
        shared_q_critic,
        shared_layer_critic,
        shared_q_actor,
        shared_layer_actor,
        q_func1_T1,
        q_func2_T1,        
        policy_optimizer1,        
        shared_q_optimizer_critic,
        shared_q_optimizer_actor,
        q_func1_optimizer1,        
        q_func2_optimizer1,        
        replay_buffer,
        gamma,
        seq_len,
        gpu=None,
        replay_start_size=10000,
        minibatch_size=100,
        update_interval=1,
        phi=lambda x: x,
        soft_update_tau=5e-3,
        max_grad_norm=None,
        logger=getLogger(__name__),
        batch_states=batch_states,
        burnin_action_func=None,
        initial_temperature=1.0,
        entropy_target=None,
        temperature_optimizer_lr=None,
        act_deterministically=True,
    ):

        self.policy1 = policy1        
        self.shared_q_critic = shared_q_critic
        self.shared_layer_critic = shared_layer_critic
        self.shared_q_actor = shared_q_actor
        self.shared_layer_actor = shared_layer_actor
        self.q_func1_T1 = q_func1_T1
        self.q_func2_T1 = q_func2_T1        
        self.countuh = 0
        self.recurrent = True

        self.train_recurrent_states_critic: Any = None
        self.train_prev_recurrent_states_critic: Any = None
        self.test_recurrent_states_critic: Any = None

        self.train_recurrent_states_actor: Any = None
        self.train_prev_recurrent_states_actor: Any = None
        self.test_recurrent_states_actor: Any = None
        
        if gpu is not None and gpu >= 0:
            assert torch.cuda.is_available()
            self.device = torch.device("cuda:{}".format(gpu))
            # self.device = xm.xla_device()
            self.policy1.to(self.device)            
            self.shared_q_critic.to(self.device)
            self.shared_layer_critic.to(self.device)
            self.shared_q_actor.to(self.device)
            self.shared_layer_actor.to(self.device)
            self.q_func1_T1.to(self.device)
            self.q_func2_T1.to(self.device)            
        else:
            self.device = torch.device("cpu")

        self.replay_buffer = replay_buffer
        self.gamma = gamma
        self.gpu = gpu
        self.phi = phi
        self.soft_update_tau = soft_update_tau
        self.logger = logger
        self.policy_optimizer1 = policy_optimizer1        
        self.shared_q_optimizer_critic = shared_q_optimizer_critic
        self.shared_q_optimizer_actor = shared_q_optimizer_actor
        self.q_func1_optimizer1 = q_func1_optimizer1        
        self.q_func2_optimizer1 = q_func2_optimizer1        
        self.replay_updater = ReplayUpdater(
            replay_buffer=replay_buffer,
            update_func=self.update,
            batchsize=minibatch_size,
            episodic_update=True,
            n_times_update=1,
            replay_start_size=replay_start_size,
            update_interval=update_interval,
            episodic_update_len=seq_len,
        )
        self.max_grad_norm = max_grad_norm
        self.batch_states = batch_states
        self.burnin_action_func = burnin_action_func
        self.initial_temperature = initial_temperature
        self.entropy_target = entropy_target
        if self.entropy_target is not None:
            self.temperature_holder1 = TemperatureHolder(
                initial_log_temperature=np.log(initial_temperature)
            )            
            if temperature_optimizer_lr is not None:
                self.temperature_optimizer1 = torch.optim.Adam(
                    self.temperature_holder1.parameters(), lr=temperature_optimizer_lr
                )            
            else:
                self.temperature_optimizer1 = torch.optim.Adam(
                    self.temperature_holder1.parameters()
                )                
            if gpu is not None and gpu >= 0:
                self.temperature_holder1.to(self.device)                
        else:
            self.temperature_holder1 = None            
            self.temperature_optimizer1 = None            
        self.act_deterministically = act_deterministically

        self.t = 0

        # Target model
        self.target_q_func_shared = copy.deepcopy(self.shared_q_critic).eval().requires_grad_(False)
        self.target_q_func_shared_layer = copy.deepcopy(self.shared_layer_critic).eval().requires_grad_(False)

        self.target_q_func1_T1 = copy.deepcopy(self.q_func1_T1).eval().requires_grad_(False)        
        self.target_q_func2_T1 = copy.deepcopy(self.q_func2_T1).eval().requires_grad_(False)        

        # Statistics
        self.q1_record_T1 = collections.deque(maxlen=1000)        
        self.q2_record_T1 = collections.deque(maxlen=1000)        
        self.entropy_record1 = collections.deque(maxlen=1000)        
        self.q_func1_loss_T1_record = collections.deque(maxlen=100)        
        self.q_func2_loss_T1_record = collections.deque(maxlen=100)       
        self.n_policy_updates = 0
        
        self.seq_len = seq_len
        self.minibatch_size = minibatch_size
        
        self.scaler = torch.cuda.amp.GradScaler(enabled=True)

    @property
    def temperature(self):
        if self.entropy_target is None:
            return self.initial_temperature #, self.initial_temperature, self.initial_temperature
        else:
            with torch.no_grad():
                return float(self.temperature_holder1())#, float(self.temperature_holder2()), float(self.temperature_holder3())

    def sync_target_network(self):
        """Synchronize target network with current network."""
        synchronize_parameters(
            src=self.q_func1_T1,
            dst=self.target_q_func1_T1,
            method="soft",
            tau=self.soft_update_tau,
        )        
        synchronize_parameters(
            src=self.q_func2_T1,
            dst=self.target_q_func2_T1,
            method="soft",
            tau=self.soft_update_tau,
        )        
        synchronize_parameters(
            src=self.shared_q_critic,
            dst=self.target_q_func_shared,
            method="soft",
            tau=self.soft_update_tau,
        )
        synchronize_parameters(
            src=self.shared_layer_critic,
            dst=self.target_q_func_shared_layer,
            method="soft",
            tau=self.soft_update_tau,
        )

    def update_q_func(self, batch):        
        """Compute loss for a given Q-function."""
        with torch.autograd.profiler.profile(use_cuda=True) as prof:
            batch_next_state = batch["next_state"]
            batch_rewards1 = batch["reward"]
            batch_terminal1 = batch["is_state_terminal"]
            batch_state = batch["state"]
            batch_actions = batch["action"]
            batch_next_actions = batch_actions
            batch_discount1 = batch["discount"]
            batch_next_recurrent_state_critic = batch["next_recurrent_state_critic"]
            batch_next_recurrent_state_actor = batch["next_recurrent_state_actor"]
            batch_recurrent_state_critic = batch["recurrent_state_critic"]
            batch_recurrent_state_actor = batch["recurrent_state_actor"]            
            batch_next_state = [tensor.to(self.device) for tensor in batch_next_state]
            batch_state = [tensor.to(self.device) for tensor in batch_state]
            
            ep_len_actual = [len(tensor) for tensor in batch_state]
            # ep_len_actual_sum1 = np.cumsum(ep_len_actual)
            # ep_len_actual_sum2 = [ep_len_actual_sum - ep_len_actual for ep_len_actual_sum, ep_len_actual in zip(ep_len_actual_sum1, ep_len_actual)]
                        
            demo_batch_actions = torch.split(batch_actions, ep_len_actual, dim=0)
            demo_batch_actions = [demo_batch_actions[:-1] for demo_batch_actions in demo_batch_actions]
            demo_batch_actions = [torch.cat((torch.zeros(1,27).to(self.device), demo_batch_actions), dim=0) for demo_batch_actions in demo_batch_actions]
            demo_batch_actions = torch.cat(demo_batch_actions)            
            
            # get the indices for episodes for each task
            # indicesA = [i for i, tensor in enumerate(batch_next_state)]            
            
            # get indices for every step in each episode and get 30 indices that are in order anywhere within the episode length from each episode
            # indicesAA = []            
            # splitter = []
            
            # for j in range(len(ep_len_actual_sum2)): 
            #     if j in indicesA:
            #         if (ep_len_actual_sum1[j] - self.seq_len) > ep_len_actual_sum2[j]:
            #             random_indexA = random.randint(ep_len_actual_sum2[j], ep_len_actual_sum1[j] - self.seq_len)
            #             numbers = [i for i in range(random_indexA, random_indexA + self.seq_len)]
            #             indicesAA = np.append(indicesAA, numbers)
            #         else:
            #             random_indexA = random.randint(ep_len_actual_sum2[j], ep_len_actual_sum1[j])
            #             numbers = [i for i in range(random_indexA, ep_len_actual_sum1[j])]
            #             indicesAA = np.append(indicesAA, numbers)
            #         splitter = np.append(splitter, len(numbers))                
            # splitter = [int(splitter) for splitter in splitter]
            # self.indicesAA = torch.tensor(indicesAA, dtype=torch.long).to(self.device)
            # splitter = tuple(splitter)
            # ndcsAA = torch.split(self.indicesAA, splitter)
            # self.ndcsAA = torch.tensor([ndcsAA[-1] for ndcsAA in ndcsAA])            

            batch_next_state = torch.cat(batch_next_state)
            # batch_next_state = batch_next_state[self.indicesAA]

            batch_state = torch.cat(batch_state)
            # batch_state = batch_state[self.indicesAA]
            
            # batch_actions = demo_batch_actions[self.indicesAA]
            # batch_next_actions = batch_next_actions[self.indicesAA]
                              
            batch_next_state = nn.utils.rnn.pad_sequence(batch_next_state, batch_first=True, padding_value=0)
            # if len(batch_next_state) < (self.seq_len * self.minibatch_size):
            #     zero_tensor = torch.zeros(((self.seq_len * self.minibatch_size), batch_next_state.shape[1])).to(self.device)
            #     zero_tensor[:batch_next_state.shape[0], :] = batch_next_state
            #     batch_next_state = zero_tensor
            batch_next_state = torch.split(batch_next_state, self.seq_len, dim=0)
            batch_next_state = [t.squeeze(0) for t in batch_next_state]
                
            batch_state = nn.utils.rnn.pad_sequence(batch_state, batch_first=True, padding_value=0)
            # if len(batch_state) < (self.seq_len * self.minibatch_size):
            #     zero_tensor1 = torch.zeros(((self.seq_len * self.minibatch_size), batch_state.shape[1])).to(self.device)
            #     zero_tensor1[:batch_state.shape[0], :] = batch_state
            #     batch_state = zero_tensor1
            batch_state = torch.split(batch_state, self.seq_len, dim=0)
            batch_state = [t.squeeze(0) for t in batch_state]
            
            batch_actions = nn.utils.rnn.pad_sequence(demo_batch_actions, batch_first=True, padding_value=0)
            # if len(batch_actions) < (self.seq_len * self.minibatch_size):
            #     zero_tensor2 = torch.zeros(((self.seq_len * self.minibatch_size), batch_actions.shape[1])).to(self.device)
            #     zero_tensor2[:batch_actions.shape[0], :] = batch_actions
            #     batch_actions = zero_tensor2
            batch_actions = torch.split(batch_actions, self.seq_len, dim=0)
            batch_actions = [t.squeeze(0) for t in batch_actions]
                      
            batch_next_actions = nn.utils.rnn.pad_sequence(batch_next_actions, batch_first=True, padding_value=0)
            # if len(batch_next_actions) < (self.seq_len * self.minibatch_size):
            #     zero_tensor3 = torch.zeros(((self.seq_len * self.minibatch_size), batch_next_actions.shape[1])).to(self.device)
            #     zero_tensor3[:batch_next_actions.shape[0], :] = batch_next_actions
            #     batch_next_actions = zero_tensor3
            batch_next_actions = torch.split(batch_next_actions, self.seq_len, dim=0)
            batch_next_actions = [t.squeeze(0) for t in batch_next_actions]
            
            batch_input_state = [torch.cat((batch_state, batch_actions), dim = 1).to(torch.float32) for batch_state, batch_actions in zip(batch_state, batch_actions)]
            # batch_input_next_state = [torch.cat((batch_next_state, batch_next_actions), dim = 1).to(torch.float32) for batch_next_state, batch_next_actions in zip(batch_next_state, batch_next_actions)]
            
            # batch_rewards1 = batch_rewards1[self.ndcsAA]
            # batch_discount1 = batch_discount1[self.ndcsAA]
            # batch_terminal1 = batch_terminal1[self.ndcsAA]
            
            batch_rewards1 = batch_rewards1[(self.seq_len - 1)::self.seq_len]
            batch_discount1 = batch_discount1[(self.seq_len - 1)::self.seq_len]
            batch_terminal1 = batch_terminal1[(self.seq_len - 1)::self.seq_len]
            
            batch_actions1 = batch_actions
            batch_actions1 = [batch_actions1[1:,:] for batch_actions1 in batch_actions1]
            
            # batch_actions = batch_actions[(self.seq_len - 1)::self.seq_len]
            # batch_actions1 = batch_actions.clone().detach().to(self.device)
            last_action = torch.cat(batch_actions).to(self.device)
            last_action = last_action[(self.seq_len - 1)::self.seq_len]
            
            #### TASK 1 #### Figure out what pfrl.utils.evaluating does
            
            with torch.no_grad(), pfrl.utils.evaluating(self.policy1), pfrl.utils.evaluating(
                self.target_q_func1_T1
            ), pfrl.utils.evaluating(self.target_q_func2_T1), pfrl.utils.evaluating(
                self.shared_q_critic), pfrl.utils.evaluating(self.shared_q_actor
            ), pfrl.utils.evaluating(self.shared_layer_critic), pfrl.utils.evaluating(self.shared_layer_actor):
                with torch.cuda.amp.autocast(device_type='cuda', dtype=torch.float16):

                    self.shared_q_actor.flatten_parameters()                
                    _, actor_recurrent_state = pack_and_forward(self.shared_q_actor, batch_next_state, batch_next_recurrent_state_actor)                
                    batch_input_next_state_actor1 = self.shared_layer_actor(actor_recurrent_state[-1])

                    self.shared_q_critic.flatten_parameters()
                    _, critic_recurrent_state = pack_and_forward(self.shared_q_critic, batch_input_state, batch_recurrent_state_critic)                
                    batch_input_state1 = self.shared_layer_critic(critic_recurrent_state[-1])

                    temp1 = self.temperature

                    next_action_distrib1 = self.policy1(batch_input_next_state_actor1)
                    next_actions1 = next_action_distrib1.sample()
                    next_log_prob1 = next_action_distrib1.log_prob(next_actions1)                               

                    # for i, ele in zip(range(len(batch_actions1)), batch_actions1):
                    #     ele = ele[1:, :]
                    #     aaa = next_actions1[i].unsqueeze(0)            
                    #     ele = torch.cat((ele, aaa), dim=0) 

                    # batch_actions1 = [torch.cat((batch_actions1, next_actions1[i].unsqueeze(0)), dim=0) for batch_actions1,i in zip(batch_actions1, range(len(next_actions1)))]                
                    batch_input_next_state = [torch.cat((batch_next_state, batch_next_actions), dim = 1).to(torch.float32) for batch_next_state, batch_next_actions in zip(batch_next_state, batch_next_actions)]

                    self.target_q_func_shared.flatten_parameters()
                    _, next_critic_recurrent_state = pack_and_forward(self.target_q_func_shared, batch_input_next_state, batch_next_recurrent_state_critic)                
                    batch_input_next_state_critic1 = self.target_q_func_shared_layer(next_critic_recurrent_state[-1])

                    next_q1T1 = self.target_q_func1_T1((batch_input_next_state_critic1, next_actions1))
                    next_q2T1 = self.target_q_func2_T1((batch_input_next_state_critic1, next_actions1))
                
                next_qT1 = torch.min(next_q1T1, next_q2T1)
                entropy_term_1 = temp1 * next_log_prob1[..., None]
                assert next_qT1.shape == entropy_term_1.shape                

                target_q_T1 = batch_rewards1 + batch_discount1 * (
                    1.0 - batch_terminal1
                ) * torch.flatten(next_qT1 - entropy_term_1)            
                
            n = 1
            
            with torch.cuda.amp.autocast(device_type='cuda', dtype=torch.float16):
                predict_q1_T1 = torch.flatten(self.q_func1_T1((batch_input_state1, last_action)))
                predict_q2_T1 = torch.flatten(self.q_func2_T1((batch_input_state1, last_action)))
                loss1_T1 = 0.5 * F.mse_loss(target_q_T1, predict_q1_T1)
                loss2_T1 = 0.5 * F.mse_loss(target_q_T1, predict_q2_T1)          

                #### NOT USED for Sep Optimizer 1, used for Sep Optimizer 2 and Shared Q ####
                loss1 = (loss1_T1)
                loss2 = (loss2_T1)
                loss = (loss1 + loss2) / 2.0
            # Update stats
            if batch_input_state1.numel() > 0:
                self.q1_record_T1.extend(predict_q1_T1.detach().cpu().numpy())
                self.q2_record_T1.extend(predict_q2_T1.detach().cpu().numpy())
                self.q_func1_loss_T1_record.append(float(loss1_T1))
                self.q_func2_loss_T1_record.append(float(loss2_T1))
            
            self.shared_q_optimizer_critic.zero_grad()
            self.scaler.scale(loss).backward(retain_graph=True)
            self.scaler.unscale_(self.shared_q_optimizer_critic)
            self.scaler.step(self.shared_q_optimizer_critic)
            # self.shared_q_optimizer_critic.step()
            # xm.mark_step()
            
            self.q_func1_optimizer1.zero_grad()
            self.scaler.scale(loss1_T1).backward()
            # loss1_T1.backward()
            self.scaler.step(self.q_func1_optimizer1)
            # self.q_func1_optimizer1.step()            
            # xm.mark_step()

            self.q_func2_optimizer1.zero_grad()
            self.scaler.scale(loss2_T1).backward()
            self.scaler.step(self.q_func2_optimizer1)
            # xm.mark_step()
            
            self.scaler.update()

    def update_temperature(self, log_prob1):        
        assert not log_prob1.requires_grad
        
        with torch.cuda.amp.autocast(device_type='cuda', dtype=torch.float16):
            loss1 = -torch.mean(self.temperature_holder1() * (log_prob1 + self.entropy_target))
        self.temperature_optimizer1.zero_grad()
        self.scaler.scale(loss1).backward()
        if self.max_grad_norm is not None:
            clip_l2_grad_norm_(self.temperature_holder1.parameters(), self.max_grad_norm)
        self.scaler.step(self.temperature_optimizer1)
        self.scaler.update()
        # xm.mark_step()

    def update_policy_and_temperature(self, batch):        
        """Compute loss for actor."""
        batch_state = batch["state"]
        batch_actions = batch["action"]        
        batch_recurrent_state_critic = batch["recurrent_state_critic"]
        batch_recurrent_state_actor = batch["recurrent_state_actor"]
        batch_rewards = batch["reward"]
        
        ep_len_actual = [len(tensor) for tensor in batch_state]

        batch_state = torch.cat(batch_state)
        # batch_state = batch_state[self.indicesAA]
        
        demo_batch_actions = torch.split(batch_actions, ep_len_actual, dim=0)
        demo_batch_actions = [demo_batch_actions[:-1] for demo_batch_actions in demo_batch_actions]
        demo_batch_actions = [torch.cat((torch.zeros(1,27).to(self.device), demo_batch_actions), dim=0) for demo_batch_actions in demo_batch_actions]
        demo_batch_actions = torch.cat(demo_batch_actions)

        # batch_actions = demo_batch_actions[self.indicesAA]
        # batch_next_actions = batch_next_actions[self.indicesAA]      
      
        batch_state = nn.utils.rnn.pad_sequence(batch_state, batch_first=True, padding_value=0)
        # if len(batch_state) < (self.seq_len * self.minibatch_size):
        #     zero_tensor1 = torch.zeros(((self.seq_len * self.minibatch_size), batch_state.shape[1])).to(self.device)
        #     zero_tensor1[:batch_state.shape[0], :] = batch_state
        #     batch_state = zero_tensor1        
        batch_state = torch.split(batch_state, self.seq_len, dim=0)
        batch_state = [t.squeeze(0) for t in batch_state]
        
        batch_actions = nn.utils.rnn.pad_sequence(batch_actions, batch_first=True, padding_value=0)
        # if len(batch_actions) < (self.seq_len * self.minibatch_size):
        #     zero_tensor2 = torch.zeros(((self.seq_len * self.minibatch_size), batch_actions.shape[1])).to(self.device)
        #     zero_tensor2[:batch_actions.shape[0], :] = batch_actions
        #     batch_actions = zero_tensor2        
        batch_actions = torch.split(batch_actions, self.seq_len, dim=0)        
        batch_actions = [t.squeeze(0) for t in batch_actions]      
        
        with torch.cuda.amp.autocast(device_type='cuda', dtype=torch.float16):
            self.shared_q_actor.flatten_parameters()
            _, actor_recurrent_state = pack_and_forward(self.shared_q_actor, batch_state, batch_recurrent_state_actor)                
            batch_input_state_actor1 = self.shared_layer_actor(actor_recurrent_state[-1])        
            temp1 = self.temperature
            n = 1

            action_distrib1 = self.policy1(batch_input_state_actor1)
            actions1 = action_distrib1.rsample()
            log_prob1 = action_distrib1.log_prob(actions1).to(self.device)        
        
        # for i, ele in zip(range(len(actions1)), batch_actions):
        #     ele = ele[:-1, :]
        #     aaa = actions1[i].unsqueeze(0)            
        #     ele = torch.cat((ele, aaa), dim=0)       
                                
        batch_input_state = [torch.cat((batch_s, batch_a), dim = 1).to(torch.float32) for batch_s, batch_a in zip(batch_state, batch_actions)]
        
        with torch.cuda.amp.autocast(device_type='cuda', dtype=torch.float16):
            self.shared_q_critic.flatten_parameters()
            _, critic_recurrent_state = pack_and_forward(self.shared_q_critic, batch_input_state, batch_recurrent_state_critic)        
            batch_input_state_critic1 = self.shared_layer_critic(critic_recurrent_state[-1])       

            # actions = torch.cat(batch_actions).to(self.device)
            # actions = actions[(self.seq_len - 1)::self.seq_len]

            q1_T1 = self.q_func1_T1((batch_input_state_critic1, actions1))
            q2_T1 = self.q_func2_T1((batch_input_state_critic1, actions1))
            q_T1 = torch.min(q1_T1, q2_T1)
            entropy_term1 = temp1 * log_prob1[..., None]
            assert q_T1.shape == entropy_term1.shape
            loss1 = torch.mean(entropy_term1 - q_T1)
            
        self.shared_q_optimizer_actor.zero_grad()
        self.scaler.scale(loss1).backward()
        self.scaler.step(self.shared_q_optimizer_actor)
        self.scaler.update()
        # xm.mark_step()

        self.n_policy_updates += 1

        if self.entropy_target is not None:
            self.update_temperature(log_prob1.detach())

        # Record entropy
        with torch.no_grad():
            try:                
                self.entropy_record1.extend(
                    action_distrib1.entropy().detach().cpu().numpy()
                )                
            except NotImplementedError:
                # Record - log p(x) instead
                self.entropy_record1.extend(-log_prob1.detach().cpu().numpy())                

    def update(self, experiences, errors_out=None):
        """Update the model from experiences"""        
        experiences = sorted(experiences, key=len, reverse=True)
        batch = batch_recurrent_experiences(experiences, self.device, self.phi, self.gamma)
        self.update_q_func(batch)
        self.update_policy_and_temperature(batch)
        self.sync_target_network()

    def batch_select_greedy_action(self, batch_obs, batch_acts, deterministic=False):        
        with torch.no_grad(), pfrl.utils.evaluating(self.policy1):#, pfrl.utils.evaluating(self.policy2), pfrl.utils.evaluating(self.policy3):
            batch_xs = self.batch_states(batch_obs, self.device, self.phi)            
                        
            if self.recurrent:
                if self.training:  
                    self.train_prev_recurrent_states_actor = self.train_recurrent_states_actor
                    with torch.cuda.amp.autocast(device_type='cuda', dtype=torch.float16):
                        self.shared_q_actor.flatten_parameters()
                        _, self.train_recurrent_states_actor = one_step_forward(
                            self.shared_q_actor, batch_xs, self.train_recurrent_states_actor
                        )                                        

                        batch_input_actor = self.shared_layer_actor(self.train_recurrent_states_actor[-1])

                        policy_out1 = self.policy1(batch_input_actor)

                        if deterministic:
                            batch_action = mode_of_distribution(policy_out1).cpu().numpy()                
                        else:
                            batch_action = policy_out1.sample().cpu().numpy()                

                        action = torch.tensor(batch_action)
                        action = action.to('cuda:0')

                        batch_input = torch.cat((batch_xs, action), dim=1)
                        self.train_prev_recurrent_states_critic = self.train_recurrent_states_critic
                        self.shared_q_critic.flatten_parameters()
                        _, self.train_recurrent_states_critic = one_step_forward(
                            self.shared_q_critic, batch_input, self.train_recurrent_states_critic
                        )
                                        
                else:                    
                    with torch.cuda.amp.autocast(device_type='cuda', dtype=torch.float16):
                        self.shared_q_actor.flatten_parameters()
                        _, self.test_recurrent_states_actor = one_step_forward(
                            self.shared_q_actor, batch_xs, self.test_recurrent_states_actor
                        )                                       

                        batch_input_actor = self.shared_layer_actor(self.test_recurrent_states_actor[-1])

                        policy_out1 = self.policy1(batch_input_actor)

                        if deterministic:
                            batch_action = mode_of_distribution(policy_out1).cpu().numpy()                
                        else:
                            batch_action = policy_out1.sample().cpu().numpy()                

                        action = torch.tensor(batch_action)
                        action = action.to('cuda:0')

                        batch_input = torch.cat((batch_xs, action), dim=1)
                        self.shared_q_critic.flatten_parameters()
                        _, self.test_recurrent_states_critic = one_step_forward(
                            self.shared_q_critic, batch_input, self.test_recurrent_states_critic
                        )
                                        
        return batch_action

    def batch_act(self, batch_obs, batch_acts):        
        if self.training:
            return self._batch_act_train(batch_obs, batch_acts)
        else:
            return self._batch_act_eval(batch_obs, batch_acts)

    def batch_observe(self, batch_obs, batch_acts, batch_reward, batch_done, batch_reset):        
        if self.training:
            self._batch_observe_train(batch_obs, batch_acts, batch_reward, batch_done, batch_reset)
        else:
            if self.recurrent:                
                self.test_recurrent_states_critic = (
                    self._batch_reset_recurrent_states_when_episodes_end(  # NOQA
                        batch_done=batch_done,
                        batch_reset=batch_reset,
                        recurrent_states=self.test_recurrent_states_critic,
                    )
                )
                
                self.test_recurrent_states_actor = (
                    self._batch_reset_recurrent_states_when_episodes_end(  # NOQA
                        batch_done=batch_done,
                        batch_reset=batch_reset,
                        recurrent_states=self.test_recurrent_states_actor,
                    )
                )               
                
    def _batch_act_eval(self, batch_obs, batch_acts):        
        assert not self.training
        return self.batch_select_greedy_action(
            batch_obs, batch_acts, deterministic=self.act_deterministically
        )

    def _batch_act_train(self, batch_obs, batch_acts):        
        assert self.training
        with torch.no_grad(), pfrl.utils.evaluating(self.policy1), pfrl.utils.evaluating(self.shared_q_actor), pfrl.utils.evaluating(self.shared_layer_actor):
            if self.burnin_action_func is not None and self.n_policy_updates == 0:
                batch_action = [self.burnin_action_func() for _ in range(len(batch_obs))]
                if self.recurrent:
                    batch_xs = self.batch_states(batch_obs, self.device, self.phi)
                    if batch_acts[0].all() == 0:
                        batch_acts = []
                        for b in range(6):
                            batch_acts.append(np.zeros(27))
                    
                    batch_axs = self.batch_states(batch_action, self.device, self.phi)
                    batch_input = torch.cat((batch_xs, batch_axs), dim=1)
                    self.train_prev_recurrent_states_critic = self.train_recurrent_states_critic
                    with torch.cuda.amp.autocast(device_type='cuda', dtype=torch.float16):
                        self.shared_q_critic.flatten_parameters()
                        _, self.train_recurrent_states_critic = one_step_forward(
                            self.shared_q_critic, batch_input, self.train_recurrent_states_critic
                        )                                       

                        self.shared_q_actor.flatten_parameters()
                        self.train_prev_recurrent_states_actor = self.train_recurrent_states_actor                    
                        _, self.train_recurrent_states_actor = one_step_forward(
                            self.shared_q_actor, batch_xs, self.train_recurrent_states_actor
                        )                   
                    
            else:
                batch_action = self.batch_select_greedy_action(batch_obs, batch_acts)
            self.batch_last_obs = list(batch_obs)
            self.batch_last_action = list(batch_action)

        return batch_action

    def _batch_reset_recurrent_states_when_episodes_end(self,
        batch_done: Sequence[bool], batch_reset: Sequence[bool], recurrent_states: Any
    ) -> Any:
        """Reset recurrent states when episodes end.

        Args:
            batch_done (array-like of bool): True iff episodes are terminal.
            batch_reset (array-like of bool): True iff episodes will be reset.
            recurrent_states (object): Recurrent state.

        Returns:
            object: New recurrent states.
        """
        indices_that_ended = [
            i
            for i, (done, reset) in enumerate(zip(batch_done, batch_reset))
            if done or reset
        ]
        if indices_that_ended:
            return mask_recurrent_state_at(recurrent_states, indices_that_ended)
        else:
            return recurrent_states

    def _batch_observe_train(self, batch_obs, batch_acts, batch_reward, batch_done, batch_reset):        
        assert self.training
        for i in range(len(batch_obs)):
            self.t += 1
            if self.batch_last_obs[i] is not None:
                assert self.batch_last_action[i] is not None
                # Add a transition to the replay buffer
                transition = {
                    "state": self.batch_last_obs[i],
                    "action": self.batch_last_action[i],
                    "reward": batch_reward[i],
                    "next_state": batch_obs[i],
                    "next_action": batch_acts[i],
                    "is_state_terminal": batch_done[i],
                }
                if self.recurrent:                                        
                    transition["recurrent_state_critic"] = recurrent_state_as_numpy(
                        get_recurrent_state_at(
                            self.train_prev_recurrent_states_critic, i, detach=True
                        )
                    )
                    transition["next_recurrent_state_critic"] = recurrent_state_as_numpy(
                        get_recurrent_state_at(
                            self.train_recurrent_states_critic, i, detach=True
                        )
                    )

                    transition["recurrent_state_actor"] = recurrent_state_as_numpy(
                        get_recurrent_state_at(
                            self.train_prev_recurrent_states_actor, i, detach=True
                        )
                    )
                    transition["next_recurrent_state_actor"] = recurrent_state_as_numpy(
                        get_recurrent_state_at(
                            self.train_recurrent_states_actor, i, detach=True
                        )
                    )                    
                 
                self.replay_buffer.append(env_id=i, **transition)
                if batch_reset[i] or batch_done[i]:
                    self.batch_last_obs[i] = None
                    self.batch_last_action[i] = None
                    self.replay_buffer.stop_current_episode(env_id=i)
            self.replay_updater.update_if_necessary(self.t)
        
        if self.recurrent:
            # Reset recurrent states when episodes end
            self.train_prev_recurrent_states_critic = None
            self.train_prev_recurrent_states_actor = None            
            
            self.train_recurrent_states_critic = (
                self._batch_reset_recurrent_states_when_episodes_end(  # NOQA
                    batch_done=batch_done,
                    batch_reset=batch_reset,
                    recurrent_states=self.train_recurrent_states_critic,
                )
            )

            self.train_recurrent_states_actor = (
                self._batch_reset_recurrent_states_when_episodes_end(  # NOQA
                    batch_done=batch_done,
                    batch_reset=batch_reset,
                    recurrent_states=self.train_recurrent_states_actor,
                )
            )                        

    def get_statistics(self):
        temp1 = self.temperature
        return [
            ("average_q1_T1", _mean_or_nan(self.q1_record_T1)),            
            ("average_q2_T1", _mean_or_nan(self.q2_record_T1)),            
            ("average_q_func1_loss_T1", _mean_or_nan(self.q_func1_loss_T1_record)),            
            ("average_q_func2_loss_T1", _mean_or_nan(self.q_func2_loss_T1_record)),            
            ("n_updates", self.n_policy_updates),            
            ("average_entropy1", _mean_or_nan(self.entropy_record1)),            
            ("temperature1", temp1),            
        ]
