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
        "policy2",
        "policy3",
        "shared_policy",        
        "q_func1_T1",
        "q_func2_T1",
        "q_func1_T2",
        "q_func2_T2",
        "q_func1_T3",
        "q_func2_T3",
        "target_q_func1_T1",
        "target_q_func2_T1",
        "target_q_func1_T2",
        "target_q_func2_T2",
        "target_q_func1_T3",
        "target_q_func2_T3",
        "policy_optimizer1",
        "policy_optimizer2",
        "policy_optimizer3",
        "shared_policy_optimizer",
        "q_func1_optimizer1",
        "q_func2_optimizer1",
        "q_func1_optimizer2",
        "q_func2_optimizer2",
        "q_func1_optimizer3",
        "q_func2_optimizer3",
        "temperature_holder1",
        "temperature_optimizer1",
        "temperature_holder2",
        "temperature_optimizer2",
        "temperature_holder3",
        "temperature_optimizer3",
    )

    def __init__(
        self,
        policy1,
        policy2,
        policy3,
        shared_policy,
        q_func1_T1,
        q_func2_T1,
        q_func1_T2,
        q_func2_T2,
        q_func1_T3,
        q_func2_T3,
        policy_optimizer1,
        policy_optimizer2,
        policy_optimizer3,
        shared_policy_optimizer,
        q_func1_optimizer1,
        q_func1_optimizer2,
        q_func1_optimizer3,
        q_func2_optimizer1,
        q_func2_optimizer2,
        q_func2_optimizer3,
        replay_buffer,
        gamma,
        update_interval=1,
        replay_start_size=10000,
        gpu=None,
        phi=lambda x: x,
        minibatch_size=100,
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
        self.policy2 = policy2
        self.policy3 = policy3
        self.shared_policy = shared_policy
        
        self.q_func1_T1 = q_func1_T1
        self.q_func2_T1 = q_func2_T1
        self.q_func1_T2 = q_func1_T2
        self.q_func2_T2 = q_func2_T2
        self.q_func1_T3 = q_func1_T3
        self.q_func2_T3 = q_func2_T3
        
        self.countuh = 0        
                
        if gpu is not None and gpu >= 0:
            assert torch.cuda.is_available()
            self.device = torch.device("cuda:{}".format(gpu))
            
            self.policy1.to(self.device)
            self.policy2.to(self.device)
            self.policy3.to(self.device)
            self.shared_policy.to(self.device)
            
            self.q_func1_T1.to(self.device)
            self.q_func2_T1.to(self.device)
            self.q_func1_T2.to(self.device)
            self.q_func2_T2.to(self.device)
            self.q_func1_T3.to(self.device)
            self.q_func2_T3.to(self.device)
                        
        else:
            self.device = torch.device("cpu")

        self.replay_buffer = replay_buffer
        self.gamma = gamma
        self.gpu = gpu
        self.phi = phi
        self.soft_update_tau = soft_update_tau
        self.logger = logger
        
        self.policy_optimizer1 = policy_optimizer1
        self.policy_optimizer2 = policy_optimizer2
        self.policy_optimizer3 = policy_optimizer3
        self.shared_policy_optimizer = shared_policy_optimizer
        
        self.q_func1_optimizer1 = q_func1_optimizer1
        self.q_func2_optimizer1 = q_func2_optimizer1
        
        self.q_func1_optimizer2 = q_func1_optimizer2
        self.q_func2_optimizer2 = q_func2_optimizer2
        
        self.q_func1_optimizer3 = q_func1_optimizer3
        self.q_func2_optimizer3 = q_func2_optimizer3
        
        self.replay_updater = ReplayUpdater(
            replay_buffer=replay_buffer,
            update_func=self.update,
            batchsize=minibatch_size,
            episodic_update=False,
            n_times_update=1,
            replay_start_size=replay_start_size,
            update_interval=update_interval,            
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
            self.temperature_holder2 = TemperatureHolder(
                initial_log_temperature=np.log(initial_temperature)
            )
            self.temperature_holder3 = TemperatureHolder(
                initial_log_temperature=np.log(initial_temperature)
            )
            if temperature_optimizer_lr is not None:
                self.temperature_optimizer1 = torch.optim.Adam(
                    self.temperature_holder1.parameters(), lr=temperature_optimizer_lr
                )
            if temperature_optimizer_lr is not None:
                self.temperature_optimizer2 = torch.optim.Adam(
                    self.temperature_holder2.parameters(), lr=temperature_optimizer_lr
                )
            if temperature_optimizer_lr is not None:
                self.temperature_optimizer3 = torch.optim.Adam(
                    self.temperature_holder3.parameters(), lr=temperature_optimizer_lr
                )
            else:
                self.temperature_optimizer1 = torch.optim.Adam(
                    self.temperature_holder1.parameters()
                )
                self.temperature_optimizer2 = torch.optim.Adam(
                    self.temperature_holder2.parameters()
                )
                self.temperature_optimizer3 = torch.optim.Adam(
                    self.temperature_holder3.parameters()
                )
            if gpu is not None and gpu >= 0:
                self.temperature_holder1.to(self.device)
                self.temperature_holder2.to(self.device)
                self.temperature_holder3.to(self.device)
        else:
            self.temperature_holder1 = None
            self.temperature_optimizer1 = None
            self.temperature_holder2 = None
            self.temperature_optimizer2 = None
            self.temperature_holder3 = None
            self.temperature_optimizer3 = None
            
        self.act_deterministically = act_deterministically

        self.t = 0

        # Target model       
        self.target_q_func1_T1 = copy.deepcopy(self.q_func1_T1).eval().requires_grad_(False)
        self.target_q_func2_T1 = copy.deepcopy(self.q_func2_T1).eval().requires_grad_(False)
        self.target_q_func1_T2 = copy.deepcopy(self.q_func1_T2).eval().requires_grad_(False)
        self.target_q_func2_T2 = copy.deepcopy(self.q_func2_T2).eval().requires_grad_(False)
        self.target_q_func1_T3 = copy.deepcopy(self.q_func1_T3).eval().requires_grad_(False)
        self.target_q_func2_T3 = copy.deepcopy(self.q_func2_T3).eval().requires_grad_(False)

        # Statistics
        self.q1_record_T1 = collections.deque(maxlen=1000)
        self.q2_record_T1 = collections.deque(maxlen=1000)
        self.entropy_record1 = collections.deque(maxlen=1000)
        self.q_func1_loss_T1_record = collections.deque(maxlen=100)
        self.q_func2_loss_T1_record = collections.deque(maxlen=100)
        
        self.q1_record_T2 = collections.deque(maxlen=1000)
        self.q2_record_T2 = collections.deque(maxlen=1000)
        self.entropy_record2 = collections.deque(maxlen=1000)
        self.q_func1_loss_T2_record = collections.deque(maxlen=100)
        self.q_func2_loss_T2_record = collections.deque(maxlen=100)
        
        self.q1_record_T3 = collections.deque(maxlen=1000)
        self.q2_record_T3 = collections.deque(maxlen=1000)
        self.entropy_record3 = collections.deque(maxlen=1000)
        self.q_func1_loss_T3_record = collections.deque(maxlen=100)
        self.q_func2_loss_T3_record = collections.deque(maxlen=100)
        
        self.n_policy_updates1 = 0
        self.n_policy_updates2 = 0
        self.n_policy_updates3 = 0
        
        self.minibatch_size = minibatch_size        
        self.scaler = torch.cuda.amp.GradScaler(enabled=True)

        self.n_tasks = 3
        self.alpha = 1.5
        self.weights = torch.ones((self.n_tasks, ), requires_grad=True, device=self.device)
        self.init_losses = None

    @property
    def temperature(self):
        if self.entropy_target is None:
            return self.initial_temperature, self.initial_temperature, self.initial_temperature
        else:
            with torch.no_grad():
                return float(self.temperature_holder1()), float(self.temperature_holder2()), float(self.temperature_holder3())

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
            src=self.q_func1_T2,
            dst=self.target_q_func1_T2,
            method="soft",
            tau=self.soft_update_tau,
        )        
        synchronize_parameters(
            src=self.q_func2_T2,
            dst=self.target_q_func2_T2,
            method="soft",
            tau=self.soft_update_tau,
        )
        synchronize_parameters(
            src=self.q_func1_T3,
            dst=self.target_q_func1_T3,
            method="soft",
            tau=self.soft_update_tau,
        )        
        synchronize_parameters(
            src=self.q_func2_T3,
            dst=self.target_q_func2_T3,
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
        
        batch_actions = batch_actions.to(torch.float32)
        
        ##### Divide into three #####
        self.mask1 = torch.all(batch_next_state[:, -3:] == torch.tensor([1, 0, 0]).to(self.device), dim=1)
        self.mask2 = torch.all(batch_next_state[:, -3:] == torch.tensor([0, 1, 0]).to(self.device), dim=1)
        self.mask3 = torch.all(batch_next_state[:, -3:] == torch.tensor([0, 0, 1]).to(self.device), dim=1)

        batch_next_state1 = batch_next_state.clone().detach()
        batch_next_state2 = batch_next_state.clone().detach()
        batch_next_state3 = batch_next_state.clone().detach()
        
        batch_state1 = batch_state.clone().detach()
        batch_state2 = batch_state.clone().detach()
        batch_state3 = batch_state.clone().detach()
        
        batch_actions1 = batch_actions.clone().detach()
        batch_actions2 = batch_actions.clone().detach()
        batch_actions3 = batch_actions.clone().detach()
        
        batch_rewards1 = batch_rewards.clone().detach()
        batch_rewards2 = batch_rewards.clone().detach()
        batch_rewards3 = batch_rewards.clone().detach()
        
        batch_terminal1 = batch_terminal.clone().detach()
        batch_terminal2 = batch_terminal.clone().detach()
        batch_terminal3 = batch_terminal.clone().detach()
        
        batch_discount1 = batch_discount.clone().detach()
        batch_discount2 = batch_discount.clone().detach()
        batch_discount3 = batch_discount.clone().detach()
        
        batch_next_state1[~self.mask1] = 0
        batch_next_state2[~self.mask2] = 0
        batch_next_state3[~self.mask3] = 0
        
        batch_state1[~self.mask1] = 0
        batch_state2[~self.mask2] = 0
        batch_state3[~self.mask3] = 0
        
        batch_actions1[~self.mask1] = 0
        batch_actions2[~self.mask2] = 0
        batch_actions3[~self.mask3] = 0
        
        batch_rewards1[~self.mask1] = 0
        batch_rewards2[~self.mask2] = 0
        batch_rewards3[~self.mask3] = 0
        
        batch_terminal1[~self.mask1] = 0
        batch_terminal2[~self.mask2] = 0
        batch_terminal3[~self.mask3] = 0
        
        batch_discount1[~self.mask1] = 0
        batch_discount2[~self.mask2] = 0
        batch_discount3[~self.mask3] = 0                

        with torch.no_grad(), pfrl.utils.evaluating(self.shared_policy), pfrl.utils.evaluating(self.policy1), pfrl.utils.evaluating(self.policy2), pfrl.utils.evaluating(self.policy3), pfrl.utils.evaluating(self.target_q_func1_T1), pfrl.utils.evaluating(self.target_q_func2_T1), pfrl.utils.evaluating(self.target_q_func1_T2), pfrl.utils.evaluating(self.target_q_func2_T2), pfrl.utils.evaluating(self.target_q_func1_T3), pfrl.utils.evaluating(self.target_q_func2_T3):            
            temp1, temp2, temp3 = self.temperature
            batch_next_state_shared = self.shared_policy(batch_next_state)            
            ##### Divide into three #####
            batch_next_state_shared1 = batch_next_state_shared.clone().detach()
            batch_next_state_shared2 = batch_next_state_shared.clone().detach()
            batch_next_state_shared3 = batch_next_state_shared.clone().detach()
            
            batch_next_state_shared1[~self.mask1] = 0
            batch_next_state_shared2[~self.mask2] = 0
            batch_next_state_shared3[~self.mask3] = 0            
            N = 0
            if batch_next_state1.numel() > 0:
                next_action_distrib1 = self.policy1(batch_next_state_shared1)
                next_actions1 = next_action_distrib1.sample()
                next_log_prob1 = next_action_distrib1.log_prob(next_actions1)
                next_q1_T1 = self.target_q_func1_T1((batch_next_state1, next_actions1))
                next_q2_T1 = self.target_q_func2_T1((batch_next_state1, next_actions1))
                next_q_T1 = torch.min(next_q1_T1, next_q2_T1)
                entropy_term1 = temp1 * next_log_prob1[..., None]
                assert next_q_T1.shape == entropy_term1.shape

                target_q_T1 = batch_rewards1 + batch_discount1 * (
                    1.0 - batch_terminal1
                ) * torch.flatten(next_q_T1 - entropy_term1)
                
                N += 1
            
            if batch_next_state2.numel() > 0:
                next_action_distrib2 = self.policy2(batch_next_state_shared2)
                next_actions2 = next_action_distrib2.sample()
                next_log_prob2 = next_action_distrib2.log_prob(next_actions2)
                next_q1_T2 = self.target_q_func1_T2((batch_next_state2, next_actions2))
                next_q2_T2 = self.target_q_func2_T2((batch_next_state2, next_actions2))
                next_q_T2 = torch.min(next_q1_T2, next_q2_T2)
                entropy_term2 = temp2 * next_log_prob2[..., None]
                assert next_q_T2.shape == entropy_term2.shape

                target_q_T2 = batch_rewards2 + batch_discount2 * (
                    1.0 - batch_terminal2
                ) * torch.flatten(next_q_T2 - entropy_term2)
                
                N += 1
            
            if batch_next_state3.numel() > 0:
                next_action_distrib3 = self.policy3(batch_next_state_shared3)
                next_actions3 = next_action_distrib3.sample()
                next_log_prob3 = next_action_distrib3.log_prob(next_actions3)
                next_q1_T3 = self.target_q_func1_T3((batch_next_state3, next_actions3))
                next_q2_T3 = self.target_q_func2_T3((batch_next_state3, next_actions3))
                next_q_T3 = torch.min(next_q1_T3, next_q2_T3)
                entropy_term3 = temp3 * next_log_prob3[..., None]
                assert next_q_T3.shape == entropy_term3.shape

                target_q_T3 = batch_rewards3 + batch_discount3 * (
                    1.0 - batch_terminal3
                ) * torch.flatten(next_q_T3 - entropy_term3)
                
                N += 1
        
        if batch_next_state1.numel() > 0:
            predict_q1_T1 = torch.flatten(self.q_func1_T1((batch_state1, batch_actions1)))
            predict_q2_T1 = torch.flatten(self.q_func2_T1((batch_state1, batch_actions1)))

            loss1_T1 = 0.5 * F.mse_loss(target_q_T1, predict_q1_T1)
            loss2_T1 = 0.5 * F.mse_loss(target_q_T1, predict_q2_T1)

            # Update stats
            self.q1_record_T1.extend(predict_q1_T1.detach().cpu().numpy())
            self.q2_record_T1.extend(predict_q2_T1.detach().cpu().numpy())
            self.q_func1_loss_T1_record.append(float(loss1_T1))
            self.q_func2_loss_T1_record.append(float(loss2_T1))

            self.q_func1_optimizer1.zero_grad()
            loss1_T1.backward()
            if self.max_grad_norm is not None:
                clip_l2_grad_norm_(self.q_func1_T1.parameters(), self.max_grad_norm)
            self.q_func1_optimizer1.step()

            self.q_func2_optimizer1.zero_grad()
            loss2_T1.backward()
            if self.max_grad_norm is not None:
                clip_l2_grad_norm_(self.q_func2_T1.parameters(), self.max_grad_norm)
            self.q_func2_optimizer1.step() 
        
        if batch_next_state2.numel() > 0:
            predict_q1_T2 = torch.flatten(self.q_func1_T2((batch_state2, batch_actions2)))
            predict_q2_T2 = torch.flatten(self.q_func2_T2((batch_state2, batch_actions2)))

            loss1_T2 = 0.5 * F.mse_loss(target_q_T2, predict_q1_T2)
            loss2_T2 = 0.5 * F.mse_loss(target_q_T2, predict_q2_T2)

            # Update stats
            self.q1_record_T2.extend(predict_q1_T2.detach().cpu().numpy())
            self.q2_record_T2.extend(predict_q2_T2.detach().cpu().numpy())
            self.q_func1_loss_T2_record.append(float(loss1_T2))
            self.q_func2_loss_T2_record.append(float(loss2_T2))

            self.q_func1_optimizer2.zero_grad()
            loss1_T2.backward()
            if self.max_grad_norm is not None:
                clip_l2_grad_norm_(self.q_func1_T2.parameters(), self.max_grad_norm)
            self.q_func1_optimizer2.step()

            self.q_func2_optimizer2.zero_grad()
            loss2_T2.backward()
            if self.max_grad_norm is not None:
                clip_l2_grad_norm_(self.q_func2_T2.parameters(), self.max_grad_norm)
            self.q_func2_optimizer2.step() 
        
        if batch_next_state3.numel() > 0:
            predict_q1_T3 = torch.flatten(self.q_func1_T3((batch_state3, batch_actions3)))
            predict_q2_T3 = torch.flatten(self.q_func2_T3((batch_state3, batch_actions3)))

            loss1_T3 = 0.5 * F.mse_loss(target_q_T3, predict_q1_T3)
            loss2_T3 = 0.5 * F.mse_loss(target_q_T3, predict_q2_T3)

            # Update stats
            self.q1_record_T3.extend(predict_q1_T3.detach().cpu().numpy())
            self.q2_record_T3.extend(predict_q2_T3.detach().cpu().numpy())
            self.q_func1_loss_T3_record.append(float(loss1_T3))
            self.q_func2_loss_T3_record.append(float(loss2_T3))

            self.q_func1_optimizer3.zero_grad()
            loss1_T3.backward()
            if self.max_grad_norm is not None:
                clip_l2_grad_norm_(self.q_func1_T3.parameters(), self.max_grad_norm)
            self.q_func1_optimizer3.step()

            self.q_func2_optimizer3.zero_grad()
            loss2_T3.backward()
            if self.max_grad_norm is not None:
                clip_l2_grad_norm_(self.q_func2_T3.parameters(), self.max_grad_norm)
            self.q_func2_optimizer3.step() 

    def update_temperature(self, log_prob1, log_prob2, log_prob3):
        assert not log_prob1.requires_grad
        assert not log_prob2.requires_grad
        assert not log_prob3.requires_grad
        
        if log_prob1.numel() > 0:
            loss1 = -torch.mean(self.temperature_holder1() * (log_prob1 + self.entropy_target))
            self.temperature_optimizer1.zero_grad()
            loss1.backward()
            if self.max_grad_norm is not None:
                clip_l2_grad_norm_(self.temperature_holder1.parameters(), self.max_grad_norm)
            self.temperature_optimizer1.step()
        
        if log_prob2.numel() > 0:
            loss2 = -torch.mean(self.temperature_holder2() * (log_prob2 + self.entropy_target))
            self.temperature_optimizer2.zero_grad()
            loss2.backward()
            if self.max_grad_norm is not None:
                clip_l2_grad_norm_(self.temperature_holder2.parameters(), self.max_grad_norm)
            self.temperature_optimizer2.step()
            
        if log_prob3.numel() > 0:
            loss3 = -torch.mean(self.temperature_holder3() * (log_prob3 + self.entropy_target))
            self.temperature_optimizer3.zero_grad()
            loss3.backward()
            if self.max_grad_norm is not None:
                clip_l2_grad_norm_(self.temperature_holder3.parameters(), self.max_grad_norm)
            self.temperature_optimizer3.step()

    def update_policy_and_temperature(self, batch):
        """Compute loss for actor."""

        batch_state = batch["state"]
        #### Divide into three ####
        batch_state1 = batch_state.clone().detach()
        batch_state2 = batch_state.clone().detach()
        batch_state3 = batch_state.clone().detach()
        
        batch_state1[~self.mask1] = 0
        batch_state2[~self.mask2] = 0
        batch_state3[~self.mask3] = 0        
        
        batch_state_shared = self.shared_policy(batch_state)        
        #### Divide into three ####
        batch_state_shared1 = batch_state_shared.clone().detach()
        batch_state_shared2 = batch_state_shared.clone().detach()
        batch_state_shared3 = batch_state_shared.clone().detach()

        batch_state_shared1[~self.mask1] = 0
        batch_state_shared2[~self.mask2] = 0
        batch_state_shared3[~self.mask3] = 0        
        
        temp1, temp2, temp3 = self.temperature
                
        self.shared_policy_optimizer.zero_grad()
        self.policy_optimizer1.zero_grad()
        self.policy_optimizer2.zero_grad()
        self.policy_optimizer3.zero_grad()
        
        N = 0
        if batch_state1.numel() > 0:
            action_distrib1 = self.policy1(batch_state_shared1)
            actions1 = action_distrib1.rsample()
            log_prob1 = action_distrib1.log_prob(actions1)
            q1_T1 = self.q_func1_T1((batch_state1, actions1))
            q2_T1 = self.q_func2_T1((batch_state1, actions1))
            q_T1 = torch.min(q1_T1, q2_T1)

            entropy_term1 = temp1 * log_prob1[..., None]
            assert q_T1.shape == entropy_term1.shape
            loss_T1 = torch.mean(entropy_term1 - q_T1)
            
            N += 1
        else:
            loss_T1 = torch.tensor([0.0], requires_grad = True).to(self.device)
            log_prob1 = torch.empty(1).to(self.device)
        loss_T1_clone = loss_T1.clone()
        last_layer_params = self.shared_policy[-1].parameters()
        dlidW = torch.autograd.grad(loss_T1_clone, last_layer_params, retain_graph=True)[0]
        print("YOYOY", dlidW)
        
        if batch_state2.numel() > 0:
            action_distrib2 = self.policy2(batch_state_shared2)
            actions2 = action_distrib2.rsample()
            log_prob2 = action_distrib2.log_prob(actions2)
            q1_T2 = self.q_func1_T2((batch_state2, actions2))
            q2_T2 = self.q_func2_T2((batch_state2, actions2))
            q_T2 = torch.min(q1_T2, q2_T2)

            entropy_term2 = temp2 * log_prob2[..., None]
            assert q_T2.shape == entropy_term2.shape
            loss_T2 = torch.mean(entropy_term2 - q_T2)
            
            N += 1
        else:
            loss_T2 = torch.tensor([0.0], requires_grad = True).to(self.device)
            log_prob2 = torch.empty(1).to(self.device)
        
        if batch_state3.numel() > 0:
            action_distrib3 = self.policy3(batch_state_shared3)
            actions3 = action_distrib3.rsample()
            log_prob3 = action_distrib3.log_prob(actions3)
            q1_T3 = self.q_func1_T3((batch_state3, actions3))
            q2_T3 = self.q_func2_T3((batch_state3, actions3))
            q_T3 = torch.min(q1_T3, q2_T3)

            entropy_term3 = temp3 * log_prob3[..., None]
            assert q_T3.shape == entropy_term3.shape
            loss_T3 = torch.mean(entropy_term3 - q_T3)
            
            N += 1
        else:
            loss_T3 = torch.tensor([0.0], requires_grad = True).to(self.device)
            log_prob3 = torch.empty(1).to(self.device)              
                        
        losses = [loss_T1 ,loss_T2, loss_T3]
        # loss_T1_clone = loss_T1.clone()
        losses = torch.stack(losses)        
        # total_weighted_loss = torch.dot(self.weights, losses)
        
        # self.shared_backward(losses, last_layer_params)

        loss = (loss_T1 + loss_T2 + loss_T3) / N
        loss.backward(retain_graph=True)        
        # self.weights.grad.zero_()

        if self.init_losses is None:
            self.init_losses = losses.detach_().data

        norms = []
        
        norms.append(torch.norm(w_i * dlidW))
        norms = torch.stack(norms)
            
        self.shared_policy_optimizer.step()
        
        loss_T1.backward()
        if self.max_grad_norm is not None:
            clip_l2_grad_norm_(self.policy1.parameters(), self.max_grad_norm)
        self.policy_optimizer1.step()
        self.n_policy_updates1 += 1
        
        loss_T2.backward()
        if self.max_grad_norm is not None:
            clip_l2_grad_norm_(self.policy2.parameters(), self.max_grad_norm)
        self.policy_optimizer2.step()
        self.n_policy_updates2 += 1
        
        loss_T3.backward()
        if self.max_grad_norm is not None:
            clip_l2_grad_norm_(self.policy3.parameters(), self.max_grad_norm)
        self.policy_optimizer3.step()
        self.n_policy_updates3 += 1
        
        if self.entropy_target is not None:
            self.update_temperature(log_prob1.detach(), log_prob2.detach(), log_prob3.detach())

        # Record entropy
        with torch.no_grad():
            try:
                if batch_state1.numel() > 0:
                    self.entropy_record1.extend(
                        action_distrib1.entropy().detach().cpu().numpy()
                    )
                if batch_state2.numel() > 0:
                    self.entropy_record2.extend(
                        action_distrib2.entropy().detach().cpu().numpy()
                    )
                if batch_state3.numel() > 0:
                    self.entropy_record3.extend(
                        action_distrib3.entropy().detach().cpu().numpy()
                    )
            except NotImplementedError:
                # Record - log p(x) instead
                if batch_state1.numel() > 0:
                    self.entropy_record1.extend(-log_prob1.detach().cpu().numpy())
                if batch_state2.numel() > 0:
                    self.entropy_record2.extend(-log_prob2.detach().cpu().numpy())
                if batch_state3.numel() > 0:
                    self.entropy_record3.extend(-log_prob3.detach().cpu().numpy())

    def shared_backward(self, losses, last_shared_params):
        """Update gradients of the weights.

        :param losses:
        :param last_shared_params:
        :param returns:
        :return:
        """        
            
        with torch.no_grad():
            # loss ratios
            loss_ratios = losses / self.init_losses
            # inverse training rate r(t)
            inverse_train_rates = loss_ratios / loss_ratios.mean()
            constant_term = norms.mean() * (inverse_train_rates ** self.alpha)

        grad_norm_loss = (norms - constant_term).abs().sum()
        self.weights.grad = torch.autograd.grad(grad_norm_loss, self.weights)[0]

        # make sure sum_i w_i = T, where T is the number of tasks
        with torch.no_grad():
            renormalize_coeff = self.n_tasks / self.weights.sum()
            self.weights *= renormalize_coeff

        if returns:
            return total_weighted_loss

    def update(self, experiences, errors_out=None):
        """Update the model from experiences"""        
        with torch.autograd.profiler.profile(use_cuda=True) as prof:
            batch = batch_experiences(experiences, self.device, self.phi, self.gamma)
            self.update_q_func(batch)
            self.update_policy_and_temperature(batch)
            self.sync_target_network()
        # print(prof)

    def batch_select_greedy_action(self, batch_obs, deterministic=False):        
        with torch.no_grad(), pfrl.utils.evaluating(self.shared_policy), pfrl.utils.evaluating(self.policy1), pfrl.utils.evaluating(self.policy2), pfrl.utils.evaluating(self.policy3):
            batch_xs = self.batch_states(batch_obs, self.device, self.phi)            
            shared_policy_out = self.shared_policy(batch_xs)
            
            mask1 = torch.all(batch_xs[:, -3:] == torch.tensor([1, 0, 0]).to(self.device), dim=1)
            mask2 = torch.all(batch_xs[:, -3:] == torch.tensor([0, 1, 0]).to(self.device), dim=1)
            mask3 = torch.all(batch_xs[:, -3:] == torch.tensor([0, 0, 1]).to(self.device), dim=1)
            
            indicesA = torch.where(mask1)[0]
            indicesB = torch.where(mask2)[0]
            indicesC = torch.where(mask3)[0]
            
            shared_policy_out1 = shared_policy_out.clone().detach()
            shared_policy_out2 = shared_policy_out.clone().detach()
            shared_policy_out3 = shared_policy_out.clone().detach()
            
            shared_policy_out1[~mask1] = 0
            shared_policy_out2[~mask2] = 0
            shared_policy_out3[~mask3] = 0            
            
            policy_out1 = self.policy1(shared_policy_out1)
            policy_out2 = self.policy2(shared_policy_out2)
            policy_out3 = self.policy3(shared_policy_out3)
                        
            batch_action = np.empty((9,23))
            if deterministic:
                batch_action1 = mode_of_distribution(policy_out1).cpu().numpy()
                batch_action2 = mode_of_distribution(policy_out2).cpu().numpy()
                batch_action3 = mode_of_distribution(policy_out3).cpu().numpy()
            else:
                batch_action1 = policy_out1.sample().cpu().numpy()
                batch_action2 = policy_out2.sample().cpu().numpy()
                batch_action3 = policy_out3.sample().cpu().numpy()
            
            for index in range(9):
                if torch.any(indicesA == index):
                    batch_action[index] = batch_action1[index%9]
                elif torch.any(indicesB == index):
                    batch_action[index] = batch_action2[index%9]
                elif torch.any(indicesC == index):
                    batch_action[index] = batch_action3[index%9]
                        
            action = torch.tensor(batch_action)            
            action = action.to('cuda:0')            
                                        
        return batch_action

    def batch_act(self, batch_obs, batch_acts):        
        if self.training:
            return self._batch_act_train(batch_obs)
        else:
            return self._batch_act_eval(batch_obs)

    def batch_observe(self, batch_obs, batch_acts, batch_reward, batch_done, batch_reset):        
        if self.training:
            self._batch_observe_train(batch_obs, batch_acts, batch_reward, batch_done, batch_reset)        
                
    def _batch_act_eval(self, batch_obs):
        assert not self.training
        return self.batch_select_greedy_action(
            batch_obs, deterministic=self.act_deterministically
        )

    def _batch_act_train(self, batch_obs):
        assert self.training
        with torch.no_grad(), pfrl.utils.evaluating(self.shared_policy), pfrl.utils.evaluating(self.policy1), pfrl.utils.evaluating(self.policy2), pfrl.utils.evaluating(self.policy3):
            if self.burnin_action_func is not None and self.n_policy_updates1 == 0 and self.n_policy_updates2 == 0 and self.n_policy_updates3 == 0:
                batch_action = [self.burnin_action_func() for _ in range(len(batch_obs))]
            else:
                batch_action = self.batch_select_greedy_action(batch_obs)            
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
                    "next_action": None,
                    "is_state_terminal": batch_done[i],
                }                
                 
                self.replay_buffer.append(env_id=i, **transition)
                if batch_reset[i] or batch_done[i]:
                    self.batch_last_obs[i] = None
                    self.batch_last_action[i] = None
                    self.replay_buffer.stop_current_episode(env_id=i)
            self.replay_updater.update_if_necessary(self.t)                     

    def get_statistics(self):
        temp1, temp2, temp3 = self.temperature
        return [
            ("average_q1_T1", _mean_or_nan(self.q1_record_T1)),
            ("average_q2_T1", _mean_or_nan(self.q2_record_T1)),
            ("average_q_func1_loss_T1", _mean_or_nan(self.q_func1_loss_T1_record)),
            ("average_q_func2_loss_T1", _mean_or_nan(self.q_func2_loss_T1_record)),
            ("n_updates1", self.n_policy_updates1),
            ("average_entropy1", _mean_or_nan(self.entropy_record1)),
            ("temperature1", temp1),
            ("average_q1_T2", _mean_or_nan(self.q1_record_T2)),
            ("average_q2_T2", _mean_or_nan(self.q2_record_T2)),
            ("average_q_func1_loss_T2", _mean_or_nan(self.q_func1_loss_T2_record)),
            ("average_q_func2_loss_T2", _mean_or_nan(self.q_func2_loss_T2_record)),
            ("n_updates2", self.n_policy_updates2),
            ("average_entropy2", _mean_or_nan(self.entropy_record2)),
            ("temperature2", temp2),
            ("average_q1_T3", _mean_or_nan(self.q1_record_T3)),
            ("average_q2_T3", _mean_or_nan(self.q2_record_T3)),
            ("average_q_func1_loss_T3", _mean_or_nan(self.q_func1_loss_T3_record)),
            ("average_q_func2_loss_T3", _mean_or_nan(self.q_func2_loss_T3_record)),
            ("n_updates3", self.n_policy_updates3),
            ("average_entropy3", _mean_or_nan(self.entropy_record3)),
            ("temperature3", temp3),
        ]
