import numpy as np
import random
from crowd_sim.envs.policy.orca import ORCA
from crowd_sim.envs.policy.sf import SF
import os
import csv

def point_to_segment_dist(x1, y1, x2, y2, x3, y3):
    """
    Calculate the closest distance between point(x3, y3) and a line segment with two endpoints (x1, y1), (x2, y2)

    """
    px = x2 - x1
    py = y2 - y1

    if px == 0 and py == 0:
        return np.linalg.norm((x3-x1, y3-y1))

    u = ((x3 - x1) * px + (y3 - y1) * py) / (px * px + py * py)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    # (x, y) is the closest point to (x3, y3) on the line segment
    x = x1 + u * px
    y = y1 + u * py

    return np.linalg.norm((x - x3, y-y3))

def get_random_policy():
    available_human_policies = ('orca', 'sf')
    return random.choice(available_human_policies)

def get_env_code(env, phase):
    assert env is not None
    assert env.circle_radius is not None
    assert env.square_width is not None
    assert env.human_num is not None
    cr_dict = {
        ( 5, 4): 'bl',
        (12, 6): 'lg',
        (10, 4): 'dn'
    }
    sq_dict = {
        ( 5, 10): 'bl',
        (20, 14): 'lg',
        (20,  10): 'dn'
    }

    if phase == 'test':
        crossing = env.test_sim
    else:
        crossing = env.train_val_sim

    if crossing == 'circle_crossing':
        setting = cr_dict.get((env.human_num, env.circle_radius))
        crossing_code = 'cr'
    else:
        setting = sq_dict.get((env.human_num, env.square_width))
        crossing_code = 'sq'

    return f"{setting}-{crossing_code}"

def write_results(path, results, fuse_std=False, stringify=False):
    """
    results: array of [env, succ, coll, time, std_time, rew, std_rew, disc_freq,
                       danger_d_min, std_danger]#, d_min_overall, std_overall]
    writes: [env, succ, coll, time, rew, disc_freq, danger_d_min]#, d_min_overall]
    """
    assert len(results) == 10

    if fuse_std:
        new = []
        # env
        new.append(results[0])
        # succ
        new.append(f'{results[1]:.2f}')
        # coll
        new.append(f'{results[2]:.2f}')
        # time
        new.append(f'{results[3]:.2f} ± {results[4]:.2f}')
        # reward
        new.append(f'{results[5]:.3f} ± {results[6]:.3f}')
        # disc_freq
        new.append(f'{results[7]:.2f}')
        # danger_d_min
        new.append(f'{results[8]:.2f} ± {results[9]:.2f}')

        if not os.path.exists(path):
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                header = ['env', 'succ', 'coll', 'time', 'rew', 'disc_freq',
                          'danger_d_min']#, 'd_min_overall']
    else:
        if stringify:
            for i in range(len(results)):
                if i == 0:
                    new.append(results[0])
                elif i == 5 or i == 6:
                    new.append(f'{results[i]:.3f}')
                else:
                    new.append(f'{results[i]:.2f}')
        else:
            new = results

        if not os.path.exists(path):
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                header = ['env', 'succ', 'coll', 'time', 'std_time', 'rew',
                          'std_rew', 'disc_freq', 'danger_d_min', 'std_danger']
                writer.writerow(header)

    # write whichever is the new array to the file
    with open(path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(new)
