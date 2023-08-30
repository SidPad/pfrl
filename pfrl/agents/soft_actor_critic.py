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
        "policy1fhalf",
        "policy1shalf",
        "policy2fhalf",
        "policy2shalf",
        "policy3",
        "q_func1_T1fhalf",
        "q_func1_T1shalf",
        "q_func2_T1fhalf",
        "q_func2_T1shalf",
        "q_func1_T2fhalf",
        "q_func1_T2shalf",
        "q_func2_T2fhalf",
        "q_func2_T2shalf",
        "q_func1_T3",        
        "q_func2_T3",
        "target_q_func1_T1fhalf",
        "target_q_func1_T1shalf",
        "target_q_func2_T1fhalf",
        "target_q_func2_T1shalf",
        "target_q_func1_T2fhalf",
        "target_q_func1_T2shalf",
        "target_q_func2_T2fhalf",
        "target_q_func2_T2shalf",
        "target_q_func1_T3",
        "target_q_func2_T3",
        "policy_optimizer1",
        "policy_optimizer2",
        "policy_optimizer3",        
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
        policy1fhalf,
        policy1shalf,
        policy2fhalf,
        policy2shalf,
        policy3,
        q_func1_T1fhalf,
        q_func1_T1shalf,
        q_func2_T1fhalf,
        q_func2_T1shalf,
        q_func1_T2fhalf,
        q_func1_T2shalf,
        q_func2_T2fhalf,
        q_func2_T2shalf,
        q_func1_T3,
        q_func2_T3,
        policy_optimizer1,
        policy_optimizer2,
        policy_optimizer3,        
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

        self.policy1fhalf = policy1fhalf
        self.policy1shalf = policy1shalf
        self.policy2fhalf = policy2fhalf
        self.policy2shalf = policy2shalf
        self.policy3 = policy3        
        
        self.q_func1_T1fhalf = q_func1_T1fhalf
        self.q_func1_T1shalf = q_func1_T1shalf
        self.q_func2_T1fhalf = q_func2_T1fhalf
        self.q_func2_T1shalf = q_func2_T1shalf
        self.q_func1_T2fhalf = q_func1_T2fhalf
        self.q_func1_T2shalf = q_func1_T2shalf
        self.q_func2_T2fhalf = q_func2_T2fhalf
        self.q_func2_T2shalf = q_func2_T2shalf
        self.q_func1_T3 = q_func1_T3
        self.q_func2_T3 = q_func2_T3
        
        self.countuh = 0        
                
        if gpu is not None and gpu >= 0:
            assert torch.cuda.is_available()
            self.device = torch.device("cuda:{}".format(gpu))
            
            self.policy1fhalf.to(self.device)
            self.policy1shalf.to(self.device)
            self.policy2fhalf.to(self.device)
            self.policy2shalf.to(self.device)
            self.policy3.to(self.device)            
            
            self.q_func1_T1fhalf.to(self.device)
            self.q_func1_T1shalf.to(self.device)
            self.q_func2_T1fhalf.to(self.device)
            self.q_func2_T1shalf.to(self.device)
            self.q_func1_T2fhalf.to(self.device)
            self.q_func1_T2shalf.to(self.device)
            self.q_func2_T2fhalf.to(self.device)
            self.q_func2_T2shalf.to(self.device)
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
                self.temperature_optimizer2 = torch.optim.Adam(
                    self.temperature_holder2.parameters(), lr=temperature_optimizer_lr
                )            
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
        self.T = 0

        # Target model       
        self.target_q_func1_T1fhalf = copy.deepcopy(self.q_func1_T1fhalf).eval().requires_grad_(False)
        self.target_q_func1_T1shalf = copy.deepcopy(self.q_func1_T1shalf).eval().requires_grad_(False)
        self.target_q_func2_T1fhalf = copy.deepcopy(self.q_func2_T1fhalf).eval().requires_grad_(False)
        self.target_q_func2_T1shalf = copy.deepcopy(self.q_func2_T1shalf).eval().requires_grad_(False)
        self.target_q_func1_T2fhalf = copy.deepcopy(self.q_func1_T2fhalf).eval().requires_grad_(False)
        self.target_q_func1_T2shalf = copy.deepcopy(self.q_func1_T2shalf).eval().requires_grad_(False)
        self.target_q_func2_T2fhalf = copy.deepcopy(self.q_func2_T2fhalf).eval().requires_grad_(False)
        self.target_q_func2_T2shalf = copy.deepcopy(self.q_func2_T2shalf).eval().requires_grad_(False)
        self.target_q_func1_T3 = copy.deepcopy(self.q_func1_T3).eval().requires_grad_(False)
        self.target_q_func2_T3 = copy.deepcopy(self.q_func2_T3).eval().requires_grad_(False)

        # Statistics
        self.q1_record = collections.deque(maxlen=1000)
        self.q2_record = collections.deque(maxlen=1000)
        self.entropy_record1 = collections.deque(maxlen=1000)
        self.entropy_record2 = collections.deque(maxlen=1000)
        self.entropy_record3 = collections.deque(maxlen=1000)
        self.q_func1_loss_record = collections.deque(maxlen=100)
        self.q_func2_loss_record = collections.deque(maxlen=100)                
        
        self.n_policy_updates = 0        
        
        self.minibatch_size = minibatch_size        
        self.scaler = torch.cuda.amp.GradScaler(enabled=True)

    @property
    def temperature(self):
        if self.entropy_target is None:
            return self.initial_temperature, self.initial_temperature, self.initial_temperature
        else:
            with torch.no_grad():
                return float(self.temperature_holder1()), float(self.temperature_holder2()), float(self.temperature_holder3())

    def sync_target_network(self, t):
        """Synchronize target network with current network."""
        if t == 1:
            synchronize_parameters(
                src=self.q_func1_T1fhalf,
                dst=self.target_q_func1_T1fhalf,
                method="soft",
                tau=self.soft_update_tau,
            )
            synchronize_parameters(
                src=self.q_func1_T1shalf,
                dst=self.target_q_func1_T1shalf,
                method="soft",
                tau=self.soft_update_tau,
            )
            synchronize_parameters(
                src=self.q_func2_T1fhalf,
                dst=self.target_q_func2_T1fhalf,
                method="soft",
                tau=self.soft_update_tau,
            )
            synchronize_parameters(
                src=self.q_func2_T1shalf,
                dst=self.target_q_func2_T1shalf,
                method="soft",
                tau=self.soft_update_tau,
            )
        elif t == 2:
            synchronize_parameters(
                src=self.q_func1_T2fhalf,
                dst=self.target_q_func1_T2fhalf,
                method="soft",
                tau=self.soft_update_tau,
            )
            synchronize_parameters(
                src=self.q_func1_T2shalf,
                dst=self.target_q_func1_T2shalf,
                method="soft",
                tau=self.soft_update_tau,
            )
            synchronize_parameters(
                src=self.q_func2_T2fhalf,
                dst=self.target_q_func2_T2fhalf,
                method="soft",
                tau=self.soft_update_tau,
            )
            synchronize_parameters(
                src=self.q_func2_T2shalf,
                dst=self.target_q_func2_T2shalf,
                method="soft",
                tau=self.soft_update_tau,
            )
        elif t == 3:
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
        self.mask1 = torch.any(torch.all(batch_next_state[:, -3:] == torch.tensor([1, 0, 0]).to(self.device), dim=1))
        self.mask2 = torch.any(torch.all(batch_next_state[:, -3:] == torch.tensor([0, 1, 0]).to(self.device), dim=1))
        self.mask3 = torch.any(torch.all(batch_next_state[:, -3:] == torch.tensor([0, 0, 1]).to(self.device), dim=1))

        if self.mask1:
            t = 1
            print(t)
        elif self.mask2:
            t = 2
            print(t)
        elif self.mask3:
            t = 3
            print(t)

        ##### separate task depedent info #####
        with torch.no_grad():
            temp1, temp2, temp3 = self.temperature
            
            batch_next_state_ind = batch_next_state[:, :55]
            batch_next_state_d = batch_next_state[:, -6:]            
            if self.mask1:
                with pfrl.utils.evaluating(self.policy1fhalf), pfrl.utils.evaluating(self.policy1shalf), pfrl.utils.evaluating(self.target_q_func1_T1fhalf), pfrl.utils.evaluating(self.target_q_func1_T1shalf), pfrl.utils.evaluating(self.target_q_func2_T1fhalf), pfrl.utils.evaluating(self.target_q_func2_T1shalf):
                    next_action_distrib = self.policy1shalf(self.policy1fhalf(batch_next_state))
                    next_actions = next_action_distrib.sample()
                    next_log_prob = next_action_distrib.log_prob(next_actions)
                    
                    target_q1_mid = self.target_q_func1_T1fhalf((batch_next_state, next_actions))
                    next_q1 = self.target_q_func1_T1shalf(target_q1_mid)
                    
                    target_q2_mid = self.target_q_func2_T1fhalf((batch_next_state, next_actions))
                    next_q2 = self.target_q_func2_T1shalf(target_q2_mid)
                    
                    next_q = torch.min(next_q1, next_q2)
                    entropy_term = temp1 * next_log_prob[..., None]
                    assert next_q.shape == entropy_term.shape
    
                    target_q = batch_rewards + batch_discount * (
                        1.0 - batch_terminal
                    ) * torch.flatten(next_q - entropy_term)
    
                    self.T = 1                    
            
            elif self.mask2:
                with pfrl.utils.evaluating(self.policy2fhalf), pfrl.utils.evaluating(self.policy2shalf), pfrl.utils.evaluating(self.target_q_func1_T2fhalf), pfrl.utils.evaluating(self.target_q_func1_T2shalf), pfrl.utils.evaluating(self.target_q_func2_T2fhalf), pfrl.utils.evaluating(self.target_q_func2_T2shalf):
                    next_action_distrib = self.policy2shalf(self.policy2fhalf(batch_next_state))
                    next_actions = next_action_distrib.sample()
                    next_log_prob = next_action_distrib.log_prob(next_actions)
                    
                    target_q1_mid = self.target_q_func1_T2fhalf((batch_next_state, next_actions))
                    next_q1 = self.target_q_func1_T2shalf(target_q1_mid)
                    
                    target_q2_mid = self.target_q_func2_T2fhalf((batch_next_state, next_actions))
                    next_q2 = self.target_q_func2_T2shalf(target_q2_mid)
                    
                    next_q = torch.min(next_q1, next_q2)
                    entropy_term = temp2 * next_log_prob[..., None]
                    assert next_q.shape == entropy_term.shape
    
                    target_q = batch_rewards + batch_discount * (
                        1.0 - batch_terminal
                    ) * torch.flatten(next_q - entropy_term)
    
                    self.T = 2                    
            
            elif self.mask3:
                with pfrl.utils.evaluating(self.policy3), pfrl.utils.evaluating(self.target_q_func1_T3), pfrl.utils.evaluating(self.target_q_func2_T3):
                    policy3_input = torch.cat((self.policy1fhalf(batch_next_state), self.policy2fhalf(batch_next_state)), dim = 1)
                    next_action_distrib = self.policy3((policy3_input))
                    next_actions = next_action_distrib.sample()
                    next_log_prob = next_action_distrib.log_prob(next_actions)
                    
                    target_q1_mid1 = self.target_q_func1_T1fhalf((batch_next_state, next_actions))
                    target_q1_mid2 = self.target_q_func1_T2fhalf((batch_next_state, next_actions))
    
                    q3_input_1 = torch.cat(target_q1_mid1, target_q1_mid2, dim = 1)
                    next_q1 = self.target_q_func1_T3(q3_input_1)
                    
                    target_q2_mid1 = self.target_q_func2_T1fhalf((batch_next_state, next_actions))    
                    target_q2_mid2 = self.target_q_func2_T2fhalf((batch_next_state, next_actions))
    
                    q3_input_2 = torch.cat(target_q2_mid1, target_q2_mid2, dim = 1)
                    next_q2 = self.target_q_func2_T3(q3_input_2)
                    
                    next_q = torch.min(next_q1, next_q2)
                    entropy_term = temp3 * next_log_prob[..., None]
                    assert next_q.shape == entropy_term.shape
    
                    target_q = batch_rewards + batch_discount * (
                        1.0 - batch_terminal
                    ) * torch.flatten(next_q - entropy_term)
    
                    self.T = 3                    

        batch_state_ind = batch_state[:, :55]
        batch_state_d = batch_state[:, -6:]
        
        if self.mask1:            
            predict_q1 = torch.flatten(self.q_func1_T1shalf(self.q_func1_T1fhalf((batch_state, batch_actions))))
            predict_q2 = torch.flatten(self.q_func2_T1shalf(self.q_func2_T1fhalf((batch_state, batch_actions))))

            loss1 = 0.5 * F.mse_loss(target_q, predict_q1)
            loss2 = 0.5 * F.mse_loss(target_q, predict_q2)

            # Update stats
            self.q1_record.extend(predict_q1.detach().cpu().numpy())
            self.q2_record.extend(predict_q2.detach().cpu().numpy())
            self.q_func1_loss_record.append(float(loss1))
            self.q_func2_loss_record.append(float(loss2))

            self.q_func1_optimizer1.zero_grad()
            loss1.backward()
            if self.max_grad_norm is not None:
                clip_l2_grad_norm_(self.q_func1.parameters(), self.max_grad_norm)
            self.q_func1_optimizer1.step()

            self.q_func2_optimizer1.zero_grad()
            loss2.backward()
            if self.max_grad_norm is not None:
                clip_l2_grad_norm_(self.q_func2.parameters(), self.max_grad_norm)
            self.q_func2_optimizer1.step()
        
        elif self.mask2:            
            predict_q1 = torch.flatten(self.q_func1_T2shalf(self.q_func1_T2fhalf((batch_state, batch_actions))))
            predict_q2 = torch.flatten(self.q_func2_T2shalf(self.q_func2_T2fhalf((batch_state, batch_actions))))

            loss1 = 0.5 * F.mse_loss(target_q, predict_q1)
            loss2 = 0.5 * F.mse_loss(target_q, predict_q2)

            # Update stats
            self.q1_record.extend(predict_q1.detach().cpu().numpy())
            self.q2_record.extend(predict_q2.detach().cpu().numpy())
            self.q_func1_loss_record.append(float(loss1))
            self.q_func2_loss_record.append(float(loss2))

            self.q_func1_optimizer2.zero_grad()
            loss1.backward()
            if self.max_grad_norm is not None:
                clip_l2_grad_norm_(self.q_func1.parameters(), self.max_grad_norm)
            self.q_func1_optimizer2.step()

            self.q_func2_optimizer2.zero_grad()
            loss2.backward()
            if self.max_grad_norm is not None:
                clip_l2_grad_norm_(self.q_func2.parameters(), self.max_grad_norm)
            self.q_func2_optimizer2.step()
        
        elif self.mask3:            
            q3_input = torch.cat((self.q_func1_T1fhalf((batch_state, batch_actions)), self.q_func2_T2fhalf((batch_state, batch_actions))), dim = 1)
            predict_q1 = torch.flatten(self.q_func1_T3(q3_input))
            predict_q2 = torch.flatten(self.q_func2_T3(q3_input))

            loss1 = 0.5 * F.mse_loss(target_q, predict_q1)
            loss2 = 0.5 * F.mse_loss(target_q, predict_q2)

            # Update stats
            self.q1_record.extend(predict_q1.detach().cpu().numpy())
            self.q2_record.extend(predict_q2.detach().cpu().numpy())
            self.q_func1_loss_record.append(float(loss1))
            self.q_func2_loss_record.append(float(loss2))

            self.q_func1_optimizer3.zero_grad()
            loss1.backward()
            if self.max_grad_norm is not None:
                clip_l2_grad_norm_(self.q_func1.parameters(), self.max_grad_norm)
            self.q_func1_optimizer3.step()

            self.q_func2_optimizer3.zero_grad()
            loss2.backward()
            if self.max_grad_norm is not None:
                clip_l2_grad_norm_(self.q_func2.parameters(), self.max_grad_norm)
            self.q_func2_optimizer3.step()
            

    def update_temperature(self, log_prob, t):
        assert not log_prob.requires_grad        
        
        if t == 1:
            loss1 = -torch.mean(self.temperature_holder1() * (log_prob + self.entropy_target))
            self.temperature_optimizer1.zero_grad()
            loss1.backward()
            if self.max_grad_norm is not None:
                clip_l2_grad_norm_(self.temperature_holder1.parameters(), self.max_grad_norm)
            self.temperature_optimizer1.step()            
        
        elif t == 2:
            loss2 = -torch.mean(self.temperature_holder2() * (log_prob + self.entropy_target))
            self.temperature_optimizer2.zero_grad()
            loss2.backward()
            if self.max_grad_norm is not None:
                clip_l2_grad_norm_(self.temperature_holder2.parameters(), self.max_grad_norm)
            self.temperature_optimizer2.step()            
            
        elif t == 3:
            loss3 = -torch.mean(self.temperature_holder3() * (log_prob + self.entropy_target))
            self.temperature_optimizer3.zero_grad()
            loss3.backward()
            if self.max_grad_norm is not None:
                clip_l2_grad_norm_(self.temperature_holder3.parameters(), self.max_grad_norm)
            self.temperature_optimizer3.step()
            

    def update_policy_and_temperature(self, batch):
        """Compute loss for actor."""

        batch_state = batch["state"]
        batch_state_ind = batch_state[:, :55]
        batch_state_d = batch_state[:, -6:]

        ##### separate task depedent info #####
        
        temp1, temp2, temp3 = self.temperature
                
        self.policy_optimizer1.zero_grad()
        self.policy_optimizer2.zero_grad()
        self.policy_optimizer3.zero_grad()
        if self.mask1:            
            action_distrib1 = self.policy1shalf(self.policy1fhalf(batch_state))
            actions = action_distrib1.rsample()
            log_prob = action_distrib1.log_prob(actions)
            q1 = self.q_func1_T1shalf(self.q_func1_T1fhalf((batch_state, actions)))
            q2 = self.q_func2_T1shalf(self.q_func2_T1fhalf((batch_state, actions)))
            q = torch.min(q1, q2)

            entropy_term1 = temp1 * log_prob[..., None]
            assert q.shape == entropy_term1.shape
            loss = torch.mean(entropy_term1 - q)
            
            loss.backward()
            if self.max_grad_norm is not None:
                clip_l2_grad_norm_(self.policy1.parameters(), self.max_grad_norm)
            self.policy_optimizer1.step()
        
        elif self.mask2> 0:            
            action_distrib2 = self.policy2shalf(self.policy2fhalf(batch_state))
            actions = action_distrib2.rsample()
            log_prob = action_distrib2.log_prob(actions)
            q1 = self.q_func1_T2shalf(self.q_func1_T2fhalf((batch_state, actions)))
            q2 = self.q_func2_T2shalf(self.q_func2_T2fhalf((batch_state, actions)))
            q = torch.min(q1, q2)

            entropy_term2 = temp2 * log_prob[..., None]
            assert q.shape == entropy_term2.shape
            loss = torch.mean(entropy_term2 - q)
            
            loss.backward()
            if self.max_grad_norm is not None:
                clip_l2_grad_norm_(self.policy2.parameters(), self.max_grad_norm)
            self.policy_optimizer2.step()
        
        elif self.mask3> 0:            
            with torch.no_grad(), pfrl.utils.evaluating(self.policy1fhalf), pfrl.utils.evaluating(self.policy2fhalf):
                policymid1, policymid2 = self.policy1fhalf(batch_state), self.policy2fhalf(batch_state)
            p3_input = torch.cat((policymid1, policymid2), dim = 1)
            action_distrib3 = self.policy3(p3_input)
            actions3 = action_distrib3.rsample()
            log_prob = action_distrib3.log_prob(actions3)
            q1 = self.q_func1_T3shalf(self.q_func1_T3fhalf((batch_state, actions)))
            q2 = self.q_func2_T3shalf(self.q_func2_T3fhalf((batch_state, actions)))
            q = torch.min(q1, q2)

            entropy_term3 = temp3 * log_prob[..., None]
            assert q.shape == entropy_term3.shape
            loss = torch.mean(entropy_term3 - q)
                  
            loss.backward()
            if self.max_grad_norm is not None:
                clip_l2_grad_norm_(self.policy3.parameters(), self.max_grad_norm)
            self.policy_optimizer3.step()
        
        self.n_policy_updates += 1

        print("T", self.T)

        if self.entropy_target is not None:
            self.update_temperature(log_prob.detach(), self.T)#, log_prob2.detach(), log_prob3.detach())

        # Record entropy
        with torch.no_grad():
            try:
                if self.mask1:
                    self.entropy_record1.extend(
                        action_distrib1.entropy().detach().cpu().numpy()
                    )
                if self.mask2:
                    self.entropy_record2.extend(
                        action_distrib2.entropy().detach().cpu().numpy()
                    )
                if self.mask3:
                    self.entropy_record3.extend(
                        action_distrib3.entropy().detach().cpu().numpy()
                    )
            except NotImplementedError:
                # Record - log p(x) instead
                if self.mask1:
                    self.entropy_record1.extend(-log_prob.detach().cpu().numpy())
                if self.mask2:
                    self.entropy_record2.extend(-log_prob.detach().cpu().numpy())
                if self.mask3:
                    self.entropy_record3.extend(-log_prob.detach().cpu().numpy())

    def update(self, experiences, errors_out=None):
        """Update the model from experiences"""        
        # with torch.autograd.profiler.profile(use_cuda=True) as prof:
        batch = batch_experiences(experiences, self.device, self.phi, self.gamma)
        self.update_q_func(batch)
        self.update_policy_and_temperature(batch)
        self.sync_target_network(self.T)
        # print(prof)        

    def batch_select_greedy_action(self, batch_obs, deterministic=False):        
        with torch.no_grad(), pfrl.utils.evaluating(self.policy1fhalf), pfrl.utils.evaluating(self.policy1shalf), pfrl.utils.evaluating(self.policy2fhalf), pfrl.utils.evaluating(self.policy2shalf), pfrl.utils.evaluating(self.policy3):
            batch_xs = self.batch_states(batch_obs, self.device, self.phi)
            batch_xs_ind = batch_xs[:, :55]
            batch_xs_d = batch_xs[:, -6:]
            ##### separate task depedent info #####
            
            mask1 = torch.any(torch.all(batch_xs[:, -3:] == torch.tensor([1, 0, 0]).to(self.device), dim=1))
            mask2 = torch.any(torch.all(batch_xs[:, -3:] == torch.tensor([0, 1, 0]).to(self.device), dim=1))
            mask3 = torch.any(torch.all(batch_xs[:, -3:] == torch.tensor([0, 0, 1]).to(self.device), dim=1))
                  
            if mask1:
                policy_out = self.policy1shalf(self.policy1fhalf(batch_xs))
            elif mask2:
                policy_out = self.policy2shalf(self.policy2fhalf(batch_xs))
            elif mask3:
                p3_input = torch.cat((self.policy1fhalf(batch_xs), self.policy2fhalf(batch_xs)), dim = 1)
                policy_out = self.policy3(p3_input)
                
            if deterministic:
                batch_action = mode_of_distribution(policy_out).cpu().numpy()                
            else:
                batch_action = policy_out.sample().cpu().numpy()                        
                        
            action = torch.tensor(batch_action)            
            action = action.to('cuda:0')            
                                        
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
        with torch.no_grad(), pfrl.utils.evaluating(self.policy1fhalf), pfrl.utils.evaluating(self.policy1shalf), pfrl.utils.evaluating(self.policy2fhalf), pfrl.utils.evaluating(self.policy2shalf), pfrl.utils.evaluating(self.policy3):
            if self.burnin_action_func is not None and self.n_policy_updates == 0:
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

    def _batch_observe_train(self, batch_obs, batch_reward, batch_done, batch_reset):        
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
            ("average_q1", _mean_or_nan(self.q1_record)),
            ("average_q2", _mean_or_nan(self.q2_record)),
            ("average_q_func1_loss", _mean_or_nan(self.q_func1_loss_record)),
            ("average_q_func2_loss", _mean_or_nan(self.q_func2_loss_record)),
            ("n_updates", self.n_policy_updates),
            ("average_entropy1", _mean_or_nan(self.entropy_record1)),
            ("average_entropy2", _mean_or_nan(self.entropy_record2)),
            ("average_entropy3", _mean_or_nan(self.entropy_record3)),
            ("temperature1", temp1),
            ("temperature2", temp2),
            ("temperature3", temp3),
        ]
