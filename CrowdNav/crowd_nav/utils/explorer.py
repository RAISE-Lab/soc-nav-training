import logging
import copy
import torch
from crowd_sim.envs.utils.info import *
from crowd_sim.envs.utils.utils import write_results, get_env_code
import numpy as np
import time
import os

np.seterr(all='raise')

class Explorer(object):
    def __init__(self, env, robot, device, memory=None, gamma=None, target_policy=None):
        self.env = env
        self.robot = robot
        self.device = device
        self.memory = memory
        self.gamma = gamma
        self.target_policy = target_policy
        self.target_model = None

    def update_target_model(self, target_model):
        self.target_model = copy.deepcopy(target_model)

    # @profile
    def run_k_episodes(self, k, phase, update_memory=False, imitation_learning=False, episode=None,
                       print_test=True, results_dir=None):
        self.robot.policy.set_phase(phase)
        success_times = []
        collision_times = []
        timeout_times = []
        success = 0
        collision = 0
        timeout = 0
        too_close = 0
        min_dist = []
        cumulative_rewards = []
        collision_cases = []
        timeout_cases = []
        for i in range(k):
            epoch_start = time.perf_counter()
            ob = self.env.reset(phase)
            done = False
            states = []
            actions = []
            rewards = []
            #logging.debug(f'i: {i}')
            while not done:
                action = self.robot.act(ob)
                ob, reward, done, info = self.env.step(action)
                states.append(self.robot.policy.last_state)
                actions.append(action)
                rewards.append(reward)
                if isinstance(info, Danger):
                    too_close += 1
                    min_dist.append(info.min_dist)

            if isinstance(info, ReachGoal):
                success += 1
                success_times.append(self.env.global_time)
            elif isinstance(info, Collision):
                collision += 1
                collision_cases.append(i)
                collision_times.append(self.env.global_time)
            elif isinstance(info, Timeout):
                timeout += 1
                timeout_cases.append(i)
                timeout_times.append(self.env.time_limit)
            else:
                raise ValueError('Invalid end signal from environment')

            if update_memory:
                if isinstance(info, ReachGoal) or isinstance(info, Collision):
                    # only add positive(success) or negative(collision) experience in experience set
                    self.update_memory(states, actions, rewards, imitation_learning)

            cumulative_rewards.append(sum([pow(self.gamma, t * self.robot.time_step * self.robot.v_pref)
                                           * reward for t, reward in enumerate(rewards)]))

            # print first 10, then every 10% milestone, then last 3
            if (
                    print_test and
                    phase.lower() == 'test' and
                    (
                        i < 100 or
                        i % (k//10) == 0 or
                        i > k-4
                    )
                ):
                ep_str = str(i).zfill(int(np.log10(k-1))+1)
                epoch_diff = time.perf_counter() - epoch_start
                logging.debug(f'Test Episode {ep_str}/{k-1}    Time: [{time.strftime("%H:%M:%S", time.gmtime(epoch_diff))}]')

        success_rate = success / k

        collision_rate = collision / k

        assert success + collision + timeout == k

        avg_nav_time = sum(success_times) / len(success_times) if success_times else self.env.time_limit

        # was checking why this wasn't working
        # logging.debug(f"success times: {success_times}")
        # Turns out, there might be *no* success times, so we condition:
        if len(success_times) > 0:
            std_nav_time = np.std(success_times)
        else:
            std_nav_time = 0


        std_cumul_rewards = np.std(cumulative_rewards)


        extra_info = '' if episode is None else 'in episode {} '.format(episode)
        logging.info('{:<5} {}has success rate: {:.2f}, collision rate: {:.2f}, nav time: {:.2f}±{:1f}, total reward: {:.4f})'.
                     format(phase.upper(), extra_info, success_rate, collision_rate, avg_nav_time, std_nav_time,
                            average(cumulative_rewards)))


        if phase in ['val', 'test']:
            num_step = sum(success_times + collision_times + timeout_times) / self.robot.time_step
            logging.info('Frequency of being in danger: %.2f and average min separate distance in danger: %.2f±%.1f', too_close / num_step, average(min_dist), np.std(min_dist))

        # array of [env, succ, coll, time, std_time, rew, std_rew, disc_freq,
        #           danger_d_min, std_danger, d_min_overall, std_overall]
        if results_dir is not None:
            results_path = os.path.join(results_dir, 'results.csv')
            logging.info(f"results_path: {results_path}")
            env_code = get_env_code(self.env, phase)
            results = [
                env_code,
                success_rate,
                collision_rate,
                avg_nav_time,
                std_nav_time,
                average(cumulative_rewards),
                std_cumul_rewards,
                too_close / num_step,
                average(min_dist),
                np.std(min_dist)
            ]
            write_results(results_path, results)

    def update_memory(self, states, actions, rewards, imitation_learning=False):
        if self.memory is None or self.gamma is None:
            raise ValueError('Memory or gamma value is not set!')

        for i, state in enumerate(states):
            reward = rewards[i]

            # VALUE UPDATE
            if imitation_learning:
                # define the value of states in IL as cumulative discounted rewards, which is the same in RL
                state = self.target_policy.transform(state)
                # value = pow(self.gamma, (len(states) - 1 - i) * self.robot.time_step * self.robot.v_pref)
                value = sum([pow(self.gamma, max(t - i, 0) * self.robot.time_step * self.robot.v_pref) * reward
                             * (1 if t >= i else 0) for t, reward in enumerate(rewards)])
            else:
                if i == len(states) - 1:
                    # terminal state
                    value = reward
                else:
                    next_state = states[i + 1]
                    gamma_bar = pow(self.gamma, self.robot.time_step * self.robot.v_pref)
                    value = reward + gamma_bar * self.target_model(next_state.unsqueeze(0)).data.item()
            value = torch.Tensor([value]).to(self.device)

            # # transform state of different human_num into fixed-size tensor
            # if len(state.size()) == 1:
            #     human_num = 1
            #     feature_size = state.size()[0]
            # else:
            #     human_num, feature_size = state.size()
            # if human_num != 5:
            #     padding = torch.zeros((5 - human_num, feature_size))
            #     state = torch.cat([state, padding])
            self.memory.push((state, value))


def average(input_list):
    #logging.debug(f'Input list:\n{input_list}')
    if input_list:
        return sum(input_list) / len(input_list)
    else:
        return 0
