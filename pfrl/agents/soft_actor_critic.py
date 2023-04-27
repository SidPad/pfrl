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
        # "policy2",
        # "policy3",
        "shared_q_critic",
        "shared_layer_critic",
        "shared_q_actor",
        "shared_layer_actor",
        "q_func1_T1",
        # "q_func1_T2",
        # "q_func1_T3",
        "q_func2_T1",
        # "q_func2_T2",
        # "q_func2_T3",
        "target_q_func1_T1",
        "target_q_func2_T1",
        # "target_q_func1_T2",
        # "target_q_func2_T2",
        # "target_q_func1_T3",
        # "target_q_func2_T3",
        "policy_optimizer1",
        # "policy_optimizer2",
        # "policy_optimizer3",
        "shared_q_optimizer_critic",
        "shared_q_optimizer_actor",
        "q_func1_optimizer1",
        # "q_func1_optimizer2",
        # "q_func1_optimizer3",
        "q_func2_optimizer1",
        # "q_func2_optimizer2",
        # "q_func2_optimizer3",
        "temperature_holder1",
        # "temperature_holder2",
        # "temperature_holder3",
        "temperature_optimizer1",
        # "temperature_optimizer2",
        # "temperature_optimizer3",
    )

    def __init__(
        self,
        policy1,
        # policy2,
        # policy3,
        shared_q_critic,
        shared_layer_critic,
        shared_q_actor,
        shared_layer_actor,
        q_func1_T1,
        q_func2_T1,
        # q_func1_T2,
        # q_func2_T2,
        # q_func1_T3,
        # q_func2_T3,
        policy_optimizer1,
        # policy_optimizer2,
        # policy_optimizer3,
        shared_q_optimizer_critic,
        shared_q_optimizer_actor,
        q_func1_optimizer1,
        # q_func1_optimizer2,
        # q_func1_optimizer3,
        q_func2_optimizer1,
        # q_func2_optimizer2,
        # q_func2_optimizer3,
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

        self.policy1 = policy1
        # self.policy2 = policy2
        # self.policy3 = policy3
        self.shared_q_critic = shared_q_critic
        self.shared_layer_critic = shared_layer_critic
        self.shared_q_actor = shared_q_actor
        self.shared_layer_actor = shared_layer_actor
        self.q_func1_T1 = q_func1_T1
        self.q_func2_T1 = q_func2_T1
        # self.q_func1_T2 = q_func1_T2
        # self.q_func2_T2 = q_func2_T2
        # self.q_func1_T3 = q_func1_T3
        # self.q_func2_T3 = q_func2_T3
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
            self.policy1.to(self.device)
            # self.policy2.to(self.device)
            # self.policy3.to(self.device)
            self.shared_q_critic.to(self.device)
            self.shared_layer_critic.to(self.device)
            self.shared_q_actor.to(self.device)
            self.shared_layer_actor.to(self.device)
            self.q_func1_T1.to(self.device)
            self.q_func2_T1.to(self.device)
            # self.q_func1_T2.to(self.device)
            # self.q_func2_T2.to(self.device)
            # self.q_func1_T3.to(self.device)
            # self.q_func2_T3.to(self.device)
        else:
            self.device = torch.device("cpu")

        self.replay_buffer = replay_buffer
        self.gamma = gamma
        self.gpu = gpu
        self.phi = phi
        self.soft_update_tau = soft_update_tau
        self.logger = logger
        self.policy_optimizer1 = policy_optimizer1
        # self.policy_optimizer2 = policy_optimizer2
        # self.policy_optimizer3 = policy_optimizer3
        self.shared_q_optimizer_critic = shared_q_optimizer_critic
        self.shared_q_optimizer_actor = shared_q_optimizer_actor
        self.q_func1_optimizer1 = q_func1_optimizer1
        # self.q_func1_optimizer2 = q_func1_optimizer2
        # self.q_func1_optimizer3 = q_func1_optimizer3
        self.q_func2_optimizer1 = q_func2_optimizer1
        # self.q_func2_optimizer2 = q_func2_optimizer2
        # self.q_func2_optimizer3 = q_func2_optimizer3
        self.replay_updater = ReplayUpdater(
            replay_buffer=replay_buffer,
            update_func=self.update,
            batchsize=minibatch_size,
            episodic_update=True,
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
            # self.temperature_holder2 = TemperatureHolder(
            #     initial_log_temperature=np.log(initial_temperature)
            # )
            # self.temperature_holder3 = TemperatureHolder(
            #     initial_log_temperature=np.log(initial_temperature)
            # )
            # temp_params = chain(self.temperature_holder1.parameters(), self.temperature_holder2.parameters(), self.temperature_holder3.parameters())
            if temperature_optimizer_lr is not None:
                self.temperature_optimizer1 = torch.optim.Adam(
                    self.temperature_holder1.parameters(), lr=temperature_optimizer_lr
                )
            # if temperature_optimizer_lr is not None:
            #     self.temperature_optimizer2 = torch.optim.Adam(
            #         self.temperature_holder2.parameters(), lr=temperature_optimizer_lr
            #     )
            # if temperature_optimizer_lr is not None:
            #     self.temperature_optimizer3 = torch.optim.Adam(
            #         self.temperature_holder3.parameters(), lr=temperature_optimizer_lr
            #     )
            else:
                self.temperature_optimizer1 = torch.optim.Adam(
                    self.temperature_holder1.parameters()
                )
                # self.temperature_optimizer2 = torch.optim.Adam(
                #     self.temperature_holder2.parameters()
                # )
                # self.temperature_optimizer3 = torch.optim.Adam(
                #     self.temperature_holder3.parameters()
                # )
            if gpu is not None and gpu >= 0:
                self.temperature_holder1.to(self.device)
                # self.temperature_holder2.to(self.device)
                # self.temperature_holder3.to(self.device)
        else:
            self.temperature_holder1 = None
            # self.temperature_holder2 = None
            # self.temperature_holder3 = None
            self.temperature_optimizer1 = None
            # self.temperature_optimizer2 = None
            # self.temperature_optimizer3 = None
        self.act_deterministically = act_deterministically

        self.t = 0

        # Target model
        self.target_q_func_shared = copy.deepcopy(self.shared_q_critic).eval().requires_grad_(False)
        self.target_q_func_shared_layer = copy.deepcopy(self.shared_layer_critic).eval().requires_grad_(False)

        self.target_q_func1_T1 = copy.deepcopy(self.q_func1_T1).eval().requires_grad_(False)
        # self.target_q_func1_T2 = copy.deepcopy(self.q_func1_T2).eval().requires_grad_(False)
        # self.target_q_func1_T3 = copy.deepcopy(self.q_func1_T3).eval().requires_grad_(False)
        self.target_q_func2_T1 = copy.deepcopy(self.q_func2_T1).eval().requires_grad_(False)
        # self.target_q_func2_T2 = copy.deepcopy(self.q_func2_T2).eval().requires_grad_(False)
        # self.target_q_func2_T3 = copy.deepcopy(self.q_func2_T3).eval().requires_grad_(False)

        # Statistics
        self.q1_record_T1 = collections.deque(maxlen=1000)
        # self.q1_record_T2 = collections.deque(maxlen=1000)
        # self.q1_record_T3 = collections.deque(maxlen=1000)
        self.q2_record_T1 = collections.deque(maxlen=1000)
        # self.q2_record_T2 = collections.deque(maxlen=1000)
        # self.q2_record_T3 = collections.deque(maxlen=1000)
        self.entropy_record1 = collections.deque(maxlen=1000)
        # self.entropy_record2 = collections.deque(maxlen=1000)
        # self.entropy_record3 = collections.deque(maxlen=1000)
        self.q_func1_loss_T1_record = collections.deque(maxlen=100)
        # self.q_func1_loss_T2_record = collections.deque(maxlen=100)
        # self.q_func1_loss_T3_record = collections.deque(maxlen=100)
        self.q_func2_loss_T1_record = collections.deque(maxlen=100)
        # self.q_func2_loss_T2_record = collections.deque(maxlen=100)
        # self.q_func2_loss_T3_record = collections.deque(maxlen=100)
        self.n_policy_updates = 0

        self.r_mean = torch.zeros((3,)).to(self.device)
        self.r_var = torch.ones((3,)).to(self.device)
        self.r_std = torch.ones((3,)).to(self.device)
        self.prev_r_mean = None
        self.prev_r_std = None
        
        self.seq_length = 4

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
        # synchronize_parameters(
        #     src=self.q_func1_T2,
        #     dst=self.target_q_func1_T2,
        #     method="soft",
        #     tau=self.soft_update_tau,
        # )
        # synchronize_parameters(
        #     src=self.q_func1_T3,
        #     dst=self.target_q_func1_T3,
        #     method="soft",
        #     tau=self.soft_update_tau,
        # )
        synchronize_parameters(
            src=self.q_func2_T1,
            dst=self.target_q_func2_T1,
            method="soft",
            tau=self.soft_update_tau,
        )
        # synchronize_parameters(
        #     src=self.q_func2_T2,
        #     dst=self.target_q_func2_T2,
        #     method="soft",
        #     tau=self.soft_update_tau,
        # )
        # synchronize_parameters(
        #     src=self.q_func2_T3,
        #     dst=self.target_q_func2_T3,
        #     method="soft",
        #     tau=self.soft_update_tau,
        # )
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
            batch_actions1 = batch["action"]
            batch_discount1 = batch["discount"]
            batch_next_recurrent_state_critic = batch["next_recurrent_state_critic"]
            batch_next_recurrent_state_actor = batch["next_recurrent_state_actor"]
            batch_recurrent_state_critic = batch["recurrent_state_critic"]
            batch_recurrent_state_actor = batch["recurrent_state_actor"]

            # print("HERE")
            # print(batch_state)
            # print(batch_actions)
            # print(batch_next_recurrent_state_critic)
            # print(batch_next_recurrent_state_actor)
            # print(batch_recurrent_state_critic)
            # print(batch_recurrent_state_actor)
            # print(input())

            batch_next_state = [tensor.to(self.device) for tensor in batch_next_state]
            batch_state = [tensor.to(self.device) for tensor in batch_state]
            
            ep_len_actual = [len(tensor) for tensor in batch_state]
            ep_len_actual_sum1 = np.cumsum(ep_len_actual)
            ep_len_actual_sum2 = [ep_len_actual_sum - ep_len_actual for ep_len_actual_sum, ep_len_actual in zip(ep_len_actual_sum1, ep_len_actual)]

            # get the indices for episodes for each task
            indicesA = [i for i, tensor in enumerate(batch_next_state) if torch.all(tensor[:, -3:] == torch.tensor([1.0, 0.0, 0.0]).to(self.device))]
            # indicesB = [i for i, tensor in enumerate(batch_next_state) if torch.all(tensor[:, -3:] == torch.tensor([0.0, 1.0, 0.0]).to(self.device))]
            # indicesC = [i for i, tensor in enumerate(batch_next_state) if torch.all(tensor[:, -3:] == torch.tensor([0.0, 0.0, 1.0]).to(self.device))]
            
            # # get indices for every step in each episode and get 30 indices that are in order anywhere within the episode length from each episode
            indicesAA = []
            # indicesBB = []
            # indicesCC = []
            
            for j in range(len(ep_len_actual_sum2)): 
                if j in indicesA:
                    random_indexA = random.randint(ep_len_actual_sum2[j], ep_len_actual_sum1[j] - self.seq_length)
                    indicesAA = np.append(indicesAA, [i for i in range(random_indexA, random_indexA + self.seq_length)])
                # if j in indicesB:
                #     random_indexB = random.randint(ep_len_actual_sum2[j], ep_len_actual_sum1[j] - 30)
                #     indicesBB = np.append(indicesBB, [i for i in range(random_indexB, random_indexB + 30)])
                # if j in indicesC:
                #     random_indexC = random.randint(ep_len_actual_sum2[j], ep_len_actual_sum1[j] - 30)
                #     indicesCC = np.append(indicesCC, [i for i in range(random_indexC, random_indexC + 30)])

            self.indicesAA = torch.tensor(indicesAA, dtype=torch.long).to(self.device)
            # self.indicesBB = torch.tensor(indicesBB, dtype=torch.long).to(self.device)
            # self.indicesCC = torch.tensor(indicesCC, dtype=torch.long).to(self.device)
            self.ndcsAA = self.indicesAA[(self.seq_length-1)::self.seq_length]
            # self.indices = torch.cat((self.indicesAA, self.indicesBB, self.indicesCC), dim=0)
            # self.indices = torch.cat((self.indicesAA), dim=0)

            batch_next_state = torch.cat(batch_next_state)
            batch_next_state = batch_next_state[self.indicesAA]

            batch_state = torch.cat(batch_state)
            batch_state = batch_state[self.indicesAA]

            batch_actions1 = batch_actions1[self.indicesAA]

            batch_next_state = nn.utils.rnn.pad_sequence(batch_next_state, batch_first=True, padding_value=0)
            # if len(batch_next_state) < 960:
            #     batch_next_state = torch.cat([batch_next_state, torch.zeros(960-len(batch_next_state), batch_next_state.shape[0]).to(self.device)], dim=1)
            batch_next_state = torch.split(batch_next_state, self.seq_length, dim=0)
            batch_next_state = [t.squeeze(0) for t in batch_next_state]
                
            batch_state = nn.utils.rnn.pad_sequence(batch_state, batch_first=True, padding_value=0)
            # if len(batch_state) < 960:
            #     batch_state = torch.cat([batch_state, torch.zeros(960-len(batch_state), batch_state.shape[0]).to(self.device)], dim=1)
            batch_state = torch.split(batch_state, self.seq_length, dim=0)
            batch_state = [t.squeeze(0) for t in batch_state]
            
            batch_actions1 = nn.utils.rnn.pad_sequence(batch_actions1, batch_first=True, padding_value=0)
            # if len(batch_actions) < 960:
            #     batch_actions = torch.cat([batch_actions, torch.zeros(960-len(batch_actions), batch_actions.shape[0]).to(self.device)], dim=1)
            batch_actions1 = torch.split(batch_actions1, self.seq_length, dim=0)
            batch_actions1 = [t.squeeze(0) for t in batch_actions1]
            
            batch_recurrent_action = []
            for action in batch_actions1:
                new_action = action.clone().detach()
                batch_recurrent_action.append(new_action)
            
            batch_recurrent_action = [tensor.cpu() for tensor in batch_recurrent_action]

            for i in range(len(batch_recurrent_action)):
                batch_recurrent_action[i] = batch_recurrent_action[i][:-1, :]
                zero_row = torch.zeros(1, batch_recurrent_action[i].shape[1])
                batch_recurrent_action[i] = torch.cat([zero_row, batch_recurrent_action[i]], dim=0)
            
            batch_recurrent_action = [tensor.to(self.device) for tensor in batch_recurrent_action]
            
            batch_input_state = [torch.cat((batch_state, batch_recurrent_action), dim = 1).to(torch.float32) for batch_state, batch_recurrent_action in zip(batch_state, batch_recurrent_action)]
            batch_input_next_state = [torch.cat((batch_next_state, batch_actions), dim = 1).to(torch.float32) for batch_next_state, batch_actions in zip(batch_next_state, batch_actions1)]
            
            # batch_rewards1 = batch_rewards.clone().detach().to(self.device)
            batch_rewards1 = batch_rewards1[self.ndcsAA]
            # batch_rewards1 = torch.cat((batch_rewards1, torch.zeros(960 - len(batch_rewards1)).to(self.device)), dim=0)
            # # batch_rewards1_mean = batch_rewards1.mean()
            # # batch_rewards1_var = batch_rewards1.var()
            # batch_rewards2 = batch_rewards_clone[self.indicesBB]
            # batch_rewards2 = torch.cat((batch_rewards2, torch.zeros(960 - len(batch_rewards2)).to(self.device)), dim=0)
            # # batch_rewards2_mean = batch_rewards2.mean()
            # # batch_rewards2_var = batch_rewards2.var()
            # batch_rewards3 = batch_rewards_clone[self.indicesCC]
            # batch_rewards3 = torch.cat((batch_rewards3, torch.zeros(960 - len(batch_rewards3)).to(self.device)), dim=0)
            # # batch_rewards3_mean = batch_rewards3.mean()
            # # batch_rewards3_var = batch_rewards3.var()

            # r_mean = torch.tensor([batch_rewards1_mean, batch_rewards2_mean, batch_rewards3_mean]).to(self.device)
            # r_var = torch.tensor([batch_rewards1_var, batch_rewards2_var, batch_rewards3_var]).to(self.device)
            # r_std = torch.tensor([torch.sqrt(batch_rewards1_var), torch.sqrt(batch_rewards2_var), torch.sqrt(batch_rewards3_var)]).to(self.device)

            # if self.prev_r_mean is None:
            #     self.prev_r_mean = self.r_mean.clone().detach()
            #     self.prev_r_std = self.r_std.clone().detach()

            # self.r_mean += 1e-1 * (r_mean - self.r_mean)
            # self.r_var += 1e-1 * (r_var - self.r_var)
            # self.r_std += 1e-1 * (r_std - self.r_std)

            # popart_factor = self.r_std / self.prev_r_std
            # batch_rewards1 = batch_rewards1 * popart_factor[0]
            # batch_rewards2 = batch_rewards2 * popart_factor[1]
            # batch_rewards3 = batch_rewards3 * popart_factor[2]

            # self.prev_r_mean = self.r_mean.clone().detach()
            # self.prev_r_std = self.r_std.clone().detach()

            # batch_discount_clone = batch_discount.clone().detach().to(self.device)
            batch_discount1 = batch_discount1[self.ndcsAA]
            # batch_discount1 = torch.cat((batch_discount1, torch.zeros(960 - len(batch_discount1)).to(self.device)), dim=0)
            # batch_discount2 = batch_discount_clone[self.indicesBB]
            # batch_discount2 = torch.cat((batch_discount2, torch.zeros(960 - len(batch_discount2)).to(self.device)), dim=0)
            # batch_discount3 = batch_discount_clone[self.indicesCC]
            # batch_discount3 = torch.cat((batch_discount3, torch.zeros(960 - len(batch_discount3)).to(self.device)), dim=0)

            # batch_terminal_clone = batch_terminal.clone().detach().to(self.device)
            batch_terminal1 = batch_terminal1[self.ndcsAA]
            # batch_terminal1 = torch.cat((batch_terminal1, torch.zeros(960 - len(batch_terminal1)).to(self.device)), dim=0)
            # batch_terminal2 = batch_terminal_clone[self.indicesBB]
            # batch_terminal2 = torch.cat((batch_terminal2, torch.zeros(960 - len(batch_terminal2)).to(self.device)), dim=0)
            # batch_terminal3 = batch_terminal_clone[self.indicesCC]
            # batch_terminal3 = torch.cat((batch_terminal3, torch.zeros(960 - len(batch_terminal3)).to(self.device)), dim=0)

            # batch_actions_clone = [batch_actions for batch_actions in batch_actions]
            batch_actions1 = torch.cat(batch_actions1).to(self.device)
            batch_actions1 = batch_actions1[(self.seq_length-1)::self.seq_length]
            # batch_actions1, batch_actions2, batch_actions3 = torch.split(batch_actions_clone, [len(self.indicesAA), len(self.indicesBB), len(self.indicesCC)])
            # batch_actions1 = torch.cat((batch_actions1, torch.zeros(960-len(self.indicesAA), 23).to(self.device))).to(torch.float32)
            # batch_actions2 = torch.cat((batch_actions2, torch.zeros(960-len(self.indicesBB), 23).to(self.device))).to(torch.float32)
            # batch_actions3 = torch.cat((batch_actions3, torch.zeros(960-len(self.indicesCC), 23).to(self.device))).to(torch.float32)
            
            #### TASK 1 #### Figure out what pfrl.utils.evaluating does
            with torch.no_grad(), pfrl.utils.evaluating(self.policy1), pfrl.utils.evaluating(
                self.target_q_func1_T1
            ), pfrl.utils.evaluating(self.target_q_func2_T1), pfrl.utils.evaluating(
                self.shared_q_critic), pfrl.utils.evaluating(self.shared_q_actor
            ), pfrl.utils.evaluating(self.shared_layer_critic), pfrl.utils.evaluating(self.shared_layer_actor):
                
                # # # batch_input_next_state_critic1, _ = pack_and_forward(self.shared_q_critic, batch_input_next_state, batch_next_recurrent_state_critic)
                # # # batch_input_next_state_critic1 = self.shared_layer_critic(batch_input_next_state_critic1)

                _, batch_input_next_state_critic1 = pack_and_forward(self.shared_q_critic, batch_input_next_state, batch_next_recurrent_state_critic)
                # batch_input_next_state_critic1 = torch.squeeze(batch_input_next_state_critic1)
                batch_input_next_state_critic1 = self.shared_layer_critic(batch_input_next_state_critic1[-1])

                # # # batch_input_next_state_actor1, _ = pack_and_forward(self.shared_q_actor, batch_input_next_state, batch_next_recurrent_state_actor)
                # # # batch_input_next_state_actor1 = self.shared_layer_actor(batch_input_next_state_actor1)

                _, batch_input_next_state_actor1 = pack_and_forward(self.shared_q_actor, batch_input_next_state, batch_next_recurrent_state_actor)
                # batch_input_next_state_actor1 = torch.squeeze(batch_input_next_state_actor1)
                batch_input_next_state_actor1 = self.shared_layer_actor(batch_input_next_state_actor1[-1])
                
                # # # batch_input_state1, _ = pack_and_forward(self.shared_q_critic, batch_input_state, batch_recurrent_state_critic)
                # # # batch_input_state1 = self.shared_layer_critic(batch_input_state1)

                _, batch_input_state1 = pack_and_forward(self.shared_q_critic, batch_input_state, batch_recurrent_state_critic)
                # batch_input_state1 = torch.squeeze(batch_input_state1)
                batch_input_state1 = self.shared_layer_critic(batch_input_state1[-1])

                # batch_input_next_state_critic_clone = batch_input_next_state_critic.clone().detach()
                # batch_input_next_state_critic1, batch_input_next_state_critic2, batch_input_next_state_critic3 = torch.split(batch_input_next_state_critic_clone, [len(self.indicesAA), len(self.indicesBB), len(self.indicesCC)])
                # batch_input_next_state_critic1 = torch.cat((batch_input_next_state_critic1, torch.zeros(960-len(self.indicesAA), 61).to(self.device))).to(torch.float32)
                # batch_input_next_state_critic2 = torch.cat((batch_input_next_state_critic2, torch.zeros(960-len(self.indicesBB), 61).to(self.device))).to(torch.float32)
                # batch_input_next_state_critic3 = torch.cat((batch_input_next_state_critic3, torch.zeros(960-len(self.indicesCC), 61).to(self.device))).to(torch.float32)

                # batch_input_next_state_actor_clone = batch_input_next_state_actor.clone().detach()
                # batch_input_next_state_actor1, batch_input_next_state_actor2, batch_input_next_state_actor3 = torch.split(batch_input_next_state_actor_clone, [len(self.indicesAA), len(self.indicesBB), len(self.indicesCC)])
                # batch_input_next_state_actor1 = torch.cat((batch_input_next_state_actor1, torch.zeros(960-len(self.indicesAA), 61).to(self.device))).to(torch.float32)
                # batch_input_next_state_actor2 = torch.cat((batch_input_next_state_actor2, torch.zeros(960-len(self.indicesBB), 61).to(self.device))).to(torch.float32)
                # batch_input_next_state_actor3 = torch.cat((batch_input_next_state_actor3, torch.zeros(960-len(self.indicesCC), 61).to(self.device))).to(torch.float32)

                # batch_input_state_clone = batch_input_state.clone().detach()
                # batch_input_state1, batch_input_state2, batch_input_state3 = torch.split(batch_input_state_clone, [len(self.indicesAA), len(self.indicesBB), len(self.indicesCC)])
                # batch_input_state1 = torch.cat((batch_input_state1, torch.zeros(960-len(self.indicesAA), 61).to(self.device))).to(torch.float32)
                # batch_input_state2 = torch.cat((batch_input_state2, torch.zeros(960-len(self.indicesBB), 61).to(self.device))).to(torch.float32)
                # batch_input_state3 = torch.cat((batch_input_state3, torch.zeros(960-len(self.indicesCC), 61).to(self.device))).to(torch.float32)
                
                # temp1, temp2, temp3 = self.temperature
                temp1 = self.temperature
                
                if batch_input_next_state_actor1.numel() > 0:
                    next_action_distrib1 = self.policy1(batch_input_next_state_actor1)
                    next_actions1 = next_action_distrib1.sample()
                    next_log_prob1 = next_action_distrib1.log_prob(next_actions1)
                    next_q1T1 = self.target_q_func1_T1((batch_input_next_state_critic1, next_actions1))
                    next_q2T1 = self.target_q_func2_T1((batch_input_next_state_critic1, next_actions1))
                    next_qT1 = torch.min(next_q1T1, next_q2T1)
                    entropy_term_1 = temp1 * next_log_prob1[..., None]
                    assert next_qT1.shape == entropy_term_1.shape

                    target_q_T1 = batch_rewards1 + batch_discount1 * (
                        1.0 - batch_terminal1
                    ) * torch.flatten(next_qT1 - entropy_term_1)
            
                # if batch_input_next_state_actor2.numel() > 0:
                #     next_action_distrib2 = self.policy2(batch_input_next_state_actor2)
                #     next_actions2 = next_action_distrib2.sample()
                #     next_log_prob2 = next_action_distrib2.log_prob(next_actions2)
                #     next_q1T2 = self.target_q_func1_T2((batch_input_next_state_critic2, next_actions2))
                #     next_q2T2 = self.target_q_func2_T2((batch_input_next_state_critic2, next_actions2))
                #     next_qT2 = torch.min(next_q1T2, next_q2T2)
                #     entropy_term_2 = temp2 * next_log_prob2[..., None]
                #     assert next_qT2.shape == entropy_term_2.shape

                #     target_q_T2 = batch_rewards2 + batch_discount2 * (
                #         1.0 - batch_terminal2
                #     ) * torch.flatten(next_qT2 - entropy_term_2)

                # if batch_input_next_state_actor3.numel() > 0:
                #     next_action_distrib3 = self.policy3(batch_input_next_state_actor3)
                #     next_actions3 = next_action_distrib3.sample()
                #     next_log_prob3 = next_action_distrib3.log_prob(next_actions3)
                #     next_q1T3 = self.target_q_func1_T3((batch_input_next_state_critic3, next_actions3))
                #     next_q2T3 = self.target_q_func2_T3((batch_input_next_state_critic3, next_actions3))
                #     next_qT3 = torch.min(next_q1T3, next_q2T3)
                #     entropy_term_3 = temp3 * next_log_prob3[..., None]
                #     assert next_qT3.shape == entropy_term_3.shape

                #     target_q_T3 = batch_rewards3 + batch_discount3 * (
                #         1.0 - batch_terminal3
                #     ) * torch.flatten(next_qT3 - entropy_term_3)
            n = 0
            if batch_input_state1.numel() > 0:
                predict_q1_T1 = torch.flatten(self.q_func1_T1((batch_input_state1, batch_actions1)))
                predict_q2_T1 = torch.flatten(self.q_func2_T1((batch_input_state1, batch_actions1)))
                loss1_T1 = 0.5 * F.mse_loss(target_q_T1, predict_q1_T1)
                loss2_T1 = 0.5 * F.mse_loss(target_q_T1, predict_q2_T1)
                n += 1
            else:
                loss1_T1 = torch.tensor([0], requires_grad = False).to(self.device)
                loss2_T1 = torch.tensor([0], requires_grad = False).to(self.device)
        
            # if batch_input_state2.numel() > 0:
            #     predict_q1_T2 = torch.flatten(self.q_func1_T2((batch_input_state2, batch_actions2)))
            #     predict_q2_T2 = torch.flatten(self.q_func2_T2((batch_input_state2, batch_actions2)))
            #     loss1_T2 = 0.5 * F.mse_loss(target_q_T2, predict_q1_T2)
            #     loss2_T2 = 0.5 * F.mse_loss(target_q_T2, predict_q2_T2)
            #     n += 1
            # else:
            #     loss1_T2 = torch.tensor([0], requires_grad = False).to(self.device)
            #     loss2_T2 = torch.tensor([0], requires_grad = False).to(self.device)
        
            # if batch_input_state3.numel() > 0:
            #     predict_q1_T3 = torch.flatten(self.q_func1_T3((batch_input_state3, batch_actions3)))
            #     predict_q2_T3 = torch.flatten(self.q_func2_T3((batch_input_state3, batch_actions3)))
            #     loss1_T3 = 0.5 * F.mse_loss(target_q_T3, predict_q1_T3)
            #     loss2_T3 = 0.5 * F.mse_loss(target_q_T3, predict_q2_T3)
            #     n += 1
            # else:
            #     loss1_T3 = torch.tensor([0], requires_grad = False).to(self.device)
            #     loss2_T3 = torch.tensor([0], requires_grad = False).to(self.device)

            #### NOT USED for Sep Optimizer 1, used for Sep Optimizer 2 and Shared Q ####
            loss1 = (loss1_T1)# + loss1_T2 + loss1_T3) / n
            loss2 = (loss2_T1)# + loss2_T2 + loss2_T3) / n
            loss = (loss1 + loss2) / 2.0
            # Update stats
            if batch_input_state1.numel() > 0:
                self.q1_record_T1.extend(predict_q1_T1.detach().cpu().numpy())
                self.q2_record_T1.extend(predict_q2_T1.detach().cpu().numpy())
                self.q_func1_loss_T1_record.append(float(loss1_T1))
                self.q_func2_loss_T1_record.append(float(loss2_T1))
        
            # if batch_input_state2.numel() > 0:
            #     self.q1_record_T2.extend(predict_q1_T2.detach().cpu().numpy())
            #     self.q2_record_T2.extend(predict_q2_T2.detach().cpu().numpy())
            #     self.q_func1_loss_T2_record.append(float(loss1_T2))
            #     self.q_func2_loss_T2_record.append(float(loss2_T2))
        
            # if batch_input_state3.numel() > 0:
            #     self.q1_record_T3.extend(predict_q1_T3.detach().cpu().numpy())
            #     self.q2_record_T3.extend(predict_q2_T3.detach().cpu().numpy())
            #     self.q_func1_loss_T3_record.append(float(loss1_T3))
            #     self.q_func2_loss_T3_record.append(float(loss2_T3))
        
            self.shared_q_optimizer_critic.zero_grad()
            loss.backward(retain_graph=True)
            self.shared_q_optimizer_critic.step()

            if loss1_T1.item() != 0:
                self.q_func1_optimizer1.zero_grad()
                loss1_T1.backward()
                self.q_func1_optimizer1.step()

                self.q_func2_optimizer1.zero_grad()
                loss2_T1.backward()
                self.q_func2_optimizer1.step()

            # if loss1_T2.item() != 0:
            #     self.q_func1_optimizer2.zero_grad()
            #     loss1_T2.backward()
            #     self.q_func1_optimizer2.step()

            #     self.q_func2_optimizer2.zero_grad()
            #     loss2_T2.backward()
            #     self.q_func2_optimizer2.step()

            # if loss1_T3.item() != 0:
            #     self.q_func1_optimizer3.zero_grad()
            #     loss1_T3.backward()
            #     self.q_func1_optimizer3.step()

            #     self.q_func2_optimizer3.zero_grad()
            #     loss2_T3.backward()
            #     self.q_func2_optimizer3.step()
        # print(prof)

    def update_temperature(self, log_prob1):#, log_prob2, log_prob3):
        assert not log_prob1.requires_grad
        # assert not log_prob2.requires_grad
        # assert not log_prob3.requires_grad

        if log_prob1.numel() > 0:
            loss1 = -torch.mean(self.temperature_holder1() * (log_prob1 + self.entropy_target))
            self.temperature_optimizer1.zero_grad()
            loss1.backward()
            if self.max_grad_norm is not None:
                clip_l2_grad_norm_(self.temperature_holder1.parameters(), self.max_grad_norm)
            self.temperature_optimizer1.step()

        # if log_prob2.numel() > 0:
        #     loss2 = -torch.mean(self.temperature_holder2() * (log_prob2 + self.entropy_target))
        #     self.temperature_optimizer2.zero_grad()
        #     loss2.backward()
        #     if self.max_grad_norm is not None:
        #         clip_l2_grad_norm_(self.temperature_holder2.parameters(), self.max_grad_norm)
        #     self.temperature_optimizer2.step()

        # if log_prob3.numel() > 0:
        #     loss3 = -torch.mean(self.temperature_holder3() * (log_prob3 + self.entropy_target))
        #     self.temperature_optimizer3.zero_grad()
        #     loss3.backward()
        #     if self.max_grad_norm is not None:
        #         clip_l2_grad_norm_(self.temperature_holder3.parameters(), self.max_grad_norm)
        #     self.temperature_optimizer3.step()

        # loss = loss1 + loss2 + loss3

    def update_policy_and_temperature(self, batch):
        """Compute loss for actor."""
        batch_state = batch["state"]
        batch_actions = batch["action"]
        batch_recurrent_state_critic = batch["recurrent_state_critic"]
        batch_recurrent_state_actor = batch["recurrent_state_actor"]
        batch_rewards = batch["reward"]
        
        ep_len_actual = [len(tensor) for tensor in batch_state]

        batch_state = torch.cat(batch_state)
        batch_state = batch_state[self.indicesAA]

        batch_actions = batch_actions[self.indicesAA]

        batch_state = nn.utils.rnn.pad_sequence(batch_state, batch_first=True, padding_value=0)
        if len(batch_state) < (minibatch_size * self.seq_length):
            batch_state = torch.cat([batch_state, torch.zeros((minibatch_size * self.seq_length)-len(batch_state), batch_state.shape[0]).to(self.device)], dim=1)
        batch_state = torch.split(batch_state, self.seq_length, dim=0)
        batch_state = [t.squeeze(0) for t in batch_state]
        
        batch_actions = nn.utils.rnn.pad_sequence(batch_actions, batch_first=True, padding_value=0)
        if len(batch_actions) < (minibatch_size * self.seq_length):
            batch_actions = torch.cat([batch_actions, torch.zeros((minibatch_size * self.seq_length)-len(batch_actions), batch_actions.shape[0]).to(self.device)], dim=1)
        batch_actions = torch.split(batch_actions, self.seq_length, dim=0)
        batch_actions = [t.squeeze(0) for t in batch_actions]

        batch_recurrent_action = []
        for action in batch_actions:
            new_action = action.clone().detach()
            batch_recurrent_action.append(new_action)
        
        for i in range(len(batch_recurrent_action)):
            batch_recurrent_action[i] = batch_recurrent_action[i][:-1, :]
            zero_row = torch.zeros(1, batch_recurrent_action[i].shape[1]).to(self.device)
            batch_recurrent_action[i] = torch.cat([zero_row, batch_recurrent_action[i]], dim=0)
        
        batch_input_state = [torch.cat((batch_state, batch_recurrent_action), dim = 1).to(torch.float32) for batch_state, batch_recurrent_action in zip(batch_state, batch_recurrent_action)]

        # # # batch_input_state_critic1, _ = pack_and_forward(self.shared_q_critic, batch_input_state, batch_recurrent_state_critic)
        # # # batch_input_state_critic1 = self.shared_layer_critic(batch_input_state_critic1)

        _, batch_input_state_critic1 = pack_and_forward(self.shared_q_critic, batch_input_state, batch_recurrent_state_critic)
        # batch_input_state_critic1 = torch.squeeze(batch_input_state_critic1)
        batch_input_state_critic1 = self.shared_layer_critic(batch_input_state_critic1[-1])

        # # # batch_input_state_actor1, _ = pack_and_forward(self.shared_q_actor, batch_input_state, batch_recurrent_state_actor)
        # # # batch_input_state_actor1 = self.shared_layer_actor(batch_input_state_actor1)

        _, batch_input_state_actor1 = pack_and_forward(self.shared_q_actor, batch_input_state, batch_recurrent_state_actor)
        # batch_input_state_actor1 = torch.squeeze(batch_input_state_actor1)
        batch_input_state_actor1 = self.shared_layer_actor(batch_input_state_actor1[-1])

        # batch_input_state_critic_clone = batch_input_state_critic.clone().detach()
        # batch_input_state_critic1, batch_input_state_critic2, batch_input_state_critic3 = torch.split(batch_input_state_critic_clone, [len(self.indicesAA), len(self.indicesBB), len(self.indicesCC)])
        # batch_input_state_critic1 = torch.cat((batch_input_state_critic1, torch.zeros(960-len(self.indicesAA), 61).to(self.device))).to(torch.float32)
        # batch_input_state_critic2 = torch.cat((batch_input_state_critic2, torch.zeros(960-len(self.indicesBB), 61).to(self.device))).to(torch.float32)
        # batch_input_state_critic3 = torch.cat((batch_input_state_critic3, torch.zeros(960-len(self.indicesCC), 61).to(self.device))).to(torch.float32)

        # batch_input_state_actor_clone = batch_input_state_actor.clone().detach()
        # batch_input_state_actor1, batch_input_state_actor2, batch_input_state_actor3 = torch.split(batch_input_state_actor_clone, [len(self.indicesAA), len(self.indicesBB), len(self.indicesCC)])
        # batch_input_state_actor1 = torch.cat((batch_input_state_actor1, torch.zeros(960-len(self.indicesAA), 61).to(self.device))).to(torch.float32)
        # batch_input_state_actor2 = torch.cat((batch_input_state_actor2, torch.zeros(960-len(self.indicesBB), 61).to(self.device))).to(torch.float32)
        # batch_input_state_actor3 = torch.cat((batch_input_state_actor3, torch.zeros(960-len(self.indicesCC), 61).to(self.device))).to(torch.float32)

        # temp1, temp2, temp3 = self.temperature
        temp1 = self.temperature
        n = 0
        if self.indicesAA.numel() > 0:
            action_distrib1 = self.policy1(batch_input_state_actor1)
            actions1 = action_distrib1.rsample()
            log_prob1 = action_distrib1.log_prob(actions1).to(self.device)
            q1_T1 = self.q_func1_T1((batch_input_state_critic1, actions1))
            q2_T1 = self.q_func2_T1((batch_input_state_critic1, actions1))
            q_T1 = torch.min(q1_T1, q2_T1)
            entropy_term1 = temp1 * log_prob1[..., None]
            assert q_T1.shape == entropy_term1.shape
            loss1 = torch.mean(entropy_term1 - q_T1)
            
            self.shared_q_optimizer_actor.zero_grad()
            loss1.backward(retain_graph=True)
            self.shared_q_optimizer_actor.step()
            
            # self.policy_optimizer1.zero_grad()
            # loss1.backward()
            # if self.max_grad_norm is not None:
            #     clip_l2_grad_norm_(self.policy1.parameters(), self.max_grad_norm)
            # self.policy_optimizer1.step()
            
            n += 1
        else:
            log_prob1 = torch.empty(1).to(self.device)
            loss1 = torch.tensor([0.0], requires_grad = True).to(self.device)

        # if self.indicesBB.numel() > 0:
        #     action_distrib2 = self.policy2(batch_input_state_actor2)
        #     actions2 = action_distrib2.rsample()
        #     log_prob2 = action_distrib2.log_prob(actions2).to(self.device)
        #     q1_T2 = self.q_func1_T2((batch_input_state_critic2, actions2))
        #     q2_T2 = self.q_func2_T2((batch_input_state_critic2, actions2))
        #     q_T2 = torch.min(q1_T2, q2_T2)
        #     entropy_term2 = temp2 * log_prob2[..., None]
        #     assert q_T2.shape == entropy_term2.shape
        #     loss2 = torch.mean(entropy_term2 - q_T2)
        #     self.policy_optimizer2.zero_grad()
        #     loss2.backward()
        #     if self.max_grad_norm is not None:
        #         clip_l2_grad_norm_(self.policy2.parameters(), self.max_grad_norm)
        #     self.policy_optimizer2.step()
        #     n += 1
        # else:
        #     log_prob2 = torch.empty(1).to(self.device)
        #     loss2 = torch.tensor([0.0], requires_grad = True).to(self.device)

        # if self.indicesCC.numel() > 0:
        #     action_distrib3 = self.policy3(batch_input_state_actor3)
        #     actions3 = action_distrib3.rsample()
        #     log_prob3 = action_distrib3.log_prob(actions3).to(self.device)
        #     q1_T3 = self.q_func1_T3((batch_input_state_critic3, actions3))
        #     q2_T3 = self.q_func2_T3((batch_input_state_critic3, actions3))
        #     q_T3 = torch.min(q1_T3, q2_T3)
        #     entropy_term3 = temp3 * log_prob3[..., None]
        #     assert q_T3.shape == entropy_term3.shape
        #     loss3 = torch.mean(entropy_term3 - q_T3)
        #     n += 1
        #     self.policy_optimizer3.zero_grad()
        #     loss3.backward()
        #     if self.max_grad_norm is not None:
        #         clip_l2_grad_norm_(self.policy3.parameters(), self.max_grad_norm)
        #     self.policy_optimizer3.step()
        # else:
        #     log_prob3 = torch.empty(1).to(self.device)
        #     loss3 = torch.tensor([0.0], requires_grad = True).to(self.device)
            
        # loss = (loss1)# + loss2 + loss3) / n

        self.n_policy_updates += 1

        if self.entropy_target is not None:
            self.update_temperature(log_prob1.detach())#, log_prob2.detach(), log_prob3.detach())

        # Record entropy
        with torch.no_grad():
            try:
                # if self.indicesAA.numel() > 0:
                self.entropy_record1.extend(
                    action_distrib1.entropy().detach().cpu().numpy()
                )
                # if self.indicesBB.numel() > 0:
                #     self.entropy_record2.extend(
                #         action_distrib2.entropy().detach().cpu().numpy()
                #     )
                # if self.indicesCC.numel() > 0:
                #     self.entropy_record3.extend(
                #         action_distrib3.entropy().detach().cpu().numpy()
                #     )
            except NotImplementedError:
                # Record - log p(x) instead
                # if self.indicesAA.numel() > 0:
                self.entropy_record1.extend(-log_prob1.detach().cpu().numpy())
                # if self.indicesBB.numel() > 0:
                #     self.entropy_record2.extend(-log_prob2.detach().cpu().numpy())
                # if self.indicesCC.numel() > 0:
                #     self.entropy_record3.extend(-log_prob3.detach().cpu().numpy())

    def update(self, experiences, errors_out=None):
        """Update the model from experiences"""
        # batch = batch_experiences(experiences, self.device, self.phi, self.gamma)
        experiences = sorted(experiences, key=len, reverse=True)
        batch = batch_recurrent_experiences(experiences, self.device, self.phi, self.gamma)
        self.update_q_func(batch)
        self.update_policy_and_temperature(batch)
        self.sync_target_network()

    def batch_select_greedy_action(self, batch_obs, batch_acts, deterministic=False):
        with torch.no_grad(), pfrl.utils.evaluating(self.policy1):#, pfrl.utils.evaluating(self.policy2), pfrl.utils.evaluating(self.policy3):
            batch_xs = self.batch_states(batch_obs, self.device, self.phi)
            batch_axs = self.batch_states(batch_acts, self.device, self.phi)
            
            # mask1 = torch.all(batch_xs[:, -3:] == torch.tensor([1.0, 0.0, 0.0]).to(self.device), dim = 1)
            # mask2 = torch.all(batch_xs[:, -3:] == torch.tensor([0.0, 1.0, 0.0]).to(self.device), dim = 1)
            # mask3 = torch.all(batch_xs[:, -3:] == torch.tensor([0.0, 0.0, 1.0]).to(self.device), dim = 1)

            # indicesA = torch.where(mask1)[0]
            # otherindicesA = torch.where(~mask1)
            # indicesB = torch.where(mask2)[0]
            # otherindicesB = torch.where(~mask2)
            # indicesC = torch.where(mask3)[0]
            # otherindicesC = torch.where(~mask3)

            batch_input = torch.cat((batch_xs, batch_axs), dim=1).to(torch.float32)

            if self.recurrent:
                if self.training:
                    self.train_prev_recurrent_states_critic = self.train_recurrent_states_critic
                    _, self.train_recurrent_states_critic = one_step_forward(
                        self.shared_q_critic, batch_input, self.train_recurrent_states_critic
                    )
                    self.train_prev_recurrent_states_actor = self.train_recurrent_states_actor
                    batch_input_actor, self.train_recurrent_states_actor = one_step_forward(
                        self.shared_q_actor, batch_input, self.train_recurrent_states_actor
                    )
                    # train_recurrent_states_actor = torch.squeeze(self.train_recurrent_states_actor)
                    batch_input_actor = self.shared_layer_actor(self.train_recurrent_states_actor[-1])
                else:
                    _, self.test_recurrent_states_critic = one_step_forward(
                        self.shared_q_critic, batch_input, self.test_recurrent_states_critic
                    )
                    batch_input_actor, self.test_recurrent_states_actor = one_step_forward(
                        self.shared_q_actor, batch_input, self.test_recurrent_states_actor
                    )
                    # test_recurrent_states_actor = torch.squeeze(self.test_recurrent_states_actor)
                    batch_input_actor = self.shared_layer_actor(self.test_recurrent_states_actor[-1])
            
            # batch_xs1 = batch_input_actor.clone().detach()
            # batch_xs2 = batch_input_actor.clone().detach()
            # batch_xs3 = batch_input_actor.clone().detach()

            # batch_xs1 = batch_xs1[indicesA]
            # batch_xs2 = batch_xs2[indicesB]
            # batch_xs3 = batch_xs3[indicesC]
            policy_out1 = self.policy1(batch_input_actor)
            # policy_out2 = self.policy2(batch_xs2)
            # policy_out3 = self.policy3(batch_xs3)

            # batch_action = np.empty((9,23))
            # batch_action = np.empty((3,23))

            # mypolicy =list(self.policy.children())
            
            if deterministic:
                batch_action = mode_of_distribution(policy_out1).cpu().numpy()
                # batch_action2 = mode_of_distribution(policy_out2).cpu().numpy()
                # batch_action3 = mode_of_distribution(policy_out3).cpu().numpy()

                # for index in range(3):
                #     if torch.any(indicesA == index):
                #         batch_action[index] = batch_action1[index%3]
                    # elif torch.any(indicesB == index):
                    #     batch_action[index] = batch_action2[index%3]
                    # elif torch.any(indicesC == index):
                    #     batch_action[index] = batch_action3[index%3]

                # keep = mypolicy[0](batch_xs)
                # keep = mypolicy[1](keep)
                # keep = mypolicy[2](keep)
                # keep = mypolicy[3](keep)
                # keep = mypolicy[4](keep)
            else:
                batch_action = policy_out1.sample().cpu().numpy()
                # batch_action2 = policy_out2.sample().cpu().numpy()
                # batch_action3 = policy_out3.sample().cpu().numpy()

                # for index in range(3):
                #     if torch.any(indicesA == index):
                #         batch_action[index] = batch_action1[index%3]
                    # elif torch.any(indicesB == index):
                    #     batch_action[index] = batch_action2[index%3]
                    # elif torch.any(indicesC == index):
                    #     batch_action[index] = batch_action3[index%3]
            
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

    def batch_act(self, batch_obs, batch_acts):
        if self.training:
            return self._batch_act_train(batch_obs, batch_acts)
        else:
            return self._batch_act_eval(batch_obs, batch_acts)

    def batch_observe(self, batch_obs, batch_reward, batch_done, batch_reset):
        if self.training:
            self._batch_observe_train(batch_obs, batch_reward, batch_done, batch_reset)
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
        if self.burnin_action_func is not None and self.n_policy_updates == 0:
            batch_action = [self.burnin_action_func() for _ in range(len(batch_obs))]
            if self.recurrent:
                batch_xs = self.batch_states(batch_obs, self.device, self.phi)
                if batch_acts[0].all() == 0:
                    batch_acts = []
                    for b in range(6):
                        batch_acts.append(np.zeros(23))
                batch_axs = self.batch_states(batch_acts, self.device, self.phi)
                
                # mask1 = torch.all(batch_xs[:, -3:] == torch.tensor([1.0, 0.0, 0.0]).to(self.device), dim = 1)
                # mask2 = torch.all(batch_xs[:, -3:] == torch.tensor([0.0, 1.0, 0.0]).to(self.device), dim = 1)
                # mask3 = torch.all(batch_xs[:, -3:] == torch.tensor([0.0, 0.0, 1.0]).to(self.device), dim = 1)

                # indicesA = torch.where(mask1)[0]
                # otherindicesA = torch.where(~mask1)
                # indicesB = torch.where(mask2)[0]
                # otherindicesB = torch.where(~mask2)
                # indicesC = torch.where(mask3)[0]
                # otherindicesC = torch.where(~mask3)

                batch_input = torch.cat((batch_xs, batch_axs), dim=1).to(torch.float32)

                self.train_prev_recurrent_states_critic = self.train_recurrent_states_critic
                _, self.train_recurrent_states_critic = one_step_forward(
                    self.shared_q_critic, batch_input, self.train_recurrent_states_critic
                )
                
                self.train_prev_recurrent_states_actor = self.train_recurrent_states_actor
                _, self.train_recurrent_states_actor = one_step_forward(
                    self.shared_q_actor, batch_input, self.train_recurrent_states_actor
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
                
                # print(self.train_recurrent_states_actor.shape)
                # print(self.train_recurrent_states_critic.shape)
                # self.replay_buffer.append(
                #     state=self.batch_last_obs[i],
                #     action=self.batch_last_action[i],
                #     reward=batch_reward[i],
                #     next_state=batch_obs[i],
                #     next_action=None,
                #     is_state_terminal=batch_done[i],
                #     env_id=i,
                # )
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
        # temp1, temp2, temp3 = self.temperature
        temp1 = self.temperature
        return [
            ("average_q1_T1", _mean_or_nan(self.q1_record_T1)),
            # ("average_q1_T2", _mean_or_nan(self.q1_record_T2)),
            # ("average_q1_T3", _mean_or_nan(self.q1_record_T3)),
            ("average_q2_T1", _mean_or_nan(self.q2_record_T1)),
            # ("average_q2_T2", _mean_or_nan(self.q2_record_T2)),
            # ("average_q2_T3", _mean_or_nan(self.q2_record_T3)),
            ("average_q_func1_loss_T1", _mean_or_nan(self.q_func1_loss_T1_record)),
            # ("average_q_func1_loss_T2", _mean_or_nan(self.q_func1_loss_T2_record)),
            # ("average_q_func1_loss_T3", _mean_or_nan(self.q_func1_loss_T3_record)),
            ("average_q_func2_loss_T1", _mean_or_nan(self.q_func2_loss_T1_record)),
            # ("average_q_func2_loss_T2", _mean_or_nan(self.q_func2_loss_T2_record)),
            # ("average_q_func2_loss_T3", _mean_or_nan(self.q_func2_loss_T3_record)),
            # ("average_q_func_shared_1_loss", _mean_or_nan(self.q_func_shared_loss_1_record)),
            # ("average_q_func_shared_2_loss", _mean_or_nan(self.q_func_shared_loss_2_record)),
            ("n_updates", self.n_policy_updates),
            # ("n_updates2", self.n_policy_updates_T2),
            # ("n_updates3", self.n_policy_updates_T3),
            ("average_entropy1", _mean_or_nan(self.entropy_record1)),
            # ("average_entropy2", _mean_or_nan(self.entropy_record2)),
            # ("average_entropy3", _mean_or_nan(self.entropy_record3)),
            ("temperature1", temp1),
            # ("temperature2", temp2),
            # ("temperature3", temp3),
        ]
