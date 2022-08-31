import logging
import argparse
import configparser
import os
import torch
import numpy as np
import gym
from crowd_nav.utils.explorer import Explorer
from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.policy.orca import ORCA
from crowd_sim.envs.policy.sf import SF


def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str, default='configs/env.config')
    parser.add_argument('--policy_config', type=str, default='configs/policy.config')
    parser.add_argument('--policy', type=str, default='orca')
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--il', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--visualize', default=False, action='store_true')
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('--test_case', type=int, default=None)
    parser.add_argument('--square', default=False, action='store_true')
    parser.add_argument('--circle', default=False, action='store_true')
    parser.add_argument('--video_file', type=str, default=None)
    parser.add_argument('--traj', default=False, action='store_true')
    parser.add_argument('--results_dir', type=str, default=None)
    parser.add_argument('--hide_attn', default=False, action='store_true')
    parser.add_argument('--show_border', default=False, action='store_true')
    args = parser.parse_args()


    if args.model_dir is not None:
        # if we have model_dir, but no results_dir
        if args.results_dir is None:
            args.results_dir = os.path.join(args.model_dir, 'eval')
        if args.il:
            model_weights = os.path.join(args.model_dir, 'il_model.pth')
        else:
            if os.path.exists(os.path.join(args.model_dir, 'resumed_rl_model.pth')):
                model_weights = os.path.join(args.model_dir, 'resumed_rl_model.pth')
            else:
                model_weights = os.path.join(args.model_dir, 'rl_model.pth')
    # if outputting results file, create dir if it doesn't exist
    if args.results_dir is not None:
        if not os.path.exists(args.results_dir):
            os.makedirs(args.results_dir)
    env_config_file = args.env_config
    policy_config_file = args.policy_config

    # configure logging and device
    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info('Using device: %s', device)

    # configure policy
    policy = policy_factory[args.policy]()
    policy_config = configparser.RawConfigParser()
    policy_config.read(policy_config_file)
    policy.configure(policy_config)
    if policy.trainable:
        if args.model_dir is None:
            parser.error('Trainable policy must be specified with a model weights directory')
        policy.get_model().load_state_dict(torch.load(model_weights))

    # configure environment
    env_config = configparser.RawConfigParser()
    logging.info(f'Env config file: {env_config_file}')
    env_config.read(env_config_file)
    env = gym.make('CrowdSim-v0')

    env.configure(env_config)
    if args.square:
        env.test_sim = 'square_crossing'
    if args.circle:
        env.test_sim = 'circle_crossing'
    robot = Robot(env_config, 'robot')
    robot.set_policy(policy)
    env.set_robot(robot)
    explorer = Explorer(env, robot, device, gamma=0.9)

    policy.set_phase(args.phase)
    policy.set_device(device)
    # set safety space for ORCA in non-cooperative simulation
    if isinstance(robot.policy, ORCA):
        if robot.visible:
            robot.policy.safety_space = 0
        else:
            # because invisible case breaks the reciprocal assumption
            # adding some safety space improves ORCA performance. Tune this value based on your need.
            robot.policy.safety_space = 0
        logging.info('ORCA robot agent buffer: %f', robot.policy.safety_space)


    policy.set_env(env)
    robot.print_info()
    if args.visualize:
        # XXX: gets the environment set up
        #       - Set px, py, gx, gy, vx, vy, theta for robot and humans
        # ob: observation
        ob = env.reset(args.phase, args.test_case)
        done = False
        last_pos = np.array(robot.get_position())

        # ADAM: step through an episode until end
        np.set_printoptions(precision=2)
        while not done:
            action = robot.act(ob)
            # ob, reward, done, info
            ob, _, done, info = env.step(action)
            current_pos = np.array(robot.get_position())
            logging.debug('Speed: %.2f', np.linalg.norm(current_pos - last_pos) / robot.time_step)
            last_pos = current_pos
        if args.traj:
            env.render('traj', args.video_file, phase=args.phase, show_border=args.show_border)
        else:
            env.render('video', args.video_file, phase=args.phase, hide_attn=args.hide_attn, show_border=args.show_border)

        logging.info('It takes %.2f seconds to finish. Final status is %s', env.global_time, info)
        if robot.visible and info == 'reach goal':
            human_times = env.get_human_times()
            logging.info(f'Average time for humans to reach goal: {sum(human_times) / len(human_times):.2f} (std: {np.std(human_times)})')

    # ADAM: RUNNING TESTING FOR K EPISODES
    else:
        explorer.run_k_episodes(env.case_size[args.phase], args.phase, results_dir=args.results_dir)


if __name__ == '__main__':
    main()
