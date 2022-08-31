# coding=utf-8

"""Synthetic pedestrian behavior according to the Social Force model.

See Helbing and Molnár 1998.
"""

import numpy as np

from .potentials import PedPedPotential
from .fieldofview import FieldOfView
from . import stateutils

MAX_SPEED_MULTIPLIER = 1.3  # with respect to initial speed


class Simulator(object):
    """Simulate social force model.

    Main interface is the state. Every pedestrian is an entry in the state and
    represented by a vector (px, py, vx, vy, gx, gy, [tau]).
    tau is optional in this vector.

    state: np array of size [num_humans, 7]

    ped_space is an instance of PedSpacePotential.
    ped_ped is an instance of PedPedPotential.

    delta_t in seconds.
    tau in seconds: either float or numpy array of shape[n_ped].
        XXX: former default: 0.5

    max_speeds: np.array of floats; shape (n_ped); XXX: *speeds*, so not 2d.
        - Equivalent to preferred speeds
    """
    def __init__(self, initial_state, ped_space=None, ped_ped=None,
                 field_of_view=None, delta_t=0.4, tau=0.5, max_speeds=None):
        self.state = initial_state
        self.initial_speeds = stateutils.speeds(initial_state)

        if max_speeds is None:
            self.max_speeds = MAX_SPEED_MULTIPLIER * self.initial_speeds
        else:
            self.max_speeds = max_speeds

        self.delta_t = delta_t

        if self.state.shape[1] < 7:
            # if tau is a float, not an ndarray
            if not hasattr(tau, 'shape'):
                if tau is None:
                    tau = self.delta_t * np.ones(self.state.shape[0])
                else:
                    tau = tau * np.ones(self.state.shape[0])
            self.state = np.concatenate((self.state, np.expand_dims(tau, -1)), axis=-1)

        # potentials
        self.V = ped_ped or PedPedPotential(self.delta_t)
        self.U = ped_space

        # field of view
        self.w = field_of_view or FieldOfView()

    def set_state(self, new_state):
        if new_state.shape == self.state.shape:
            self.state = new_state
        # sf_state without tau
        elif ((new_state.shape[0] == self.state.shape[0]) and
              (new_state.shape[1] == 6)):
            self.state[:,:6] = new_state
        else:
            raise ValueError("New state of shape", new_state.shape, "shoud be of shape", self.state.shape, "or", (self.state.shape[0], 6) )

    def f_ab(self):
        """Compute f_ab."""
        return -1.0 * self.V.grad_r_ab(self.state)

    def f_aB(self):
        """Compute f_aB."""
        if self.U is None:
            return np.zeros((self.state.shape[0], 0, 2))
        return -1.0 * self.U.grad_r_aB(self.state)

    def capped_velocity(self, desired_velocity):
        """Scale down a desired velocity to its capped speed."""
        desired_speeds = np.linalg.norm(desired_velocity, axis=-1)
        factor = np.minimum(1.0, self.max_speeds / desired_speeds)
        factor[desired_speeds == 0] = 0.0
        return desired_velocity * np.expand_dims(factor, -1)

    def step(self):
        """Do one step in the simulation and update the state in place."""
        # accelerate to desired velocity
        e = stateutils.desired_directions(self.state)
        vel = self.state[:, 2:4]
        tau = self.state[:, 6:7]
        F0 = 1.0 / tau * (np.expand_dims(self.initial_speeds, -1) * e - vel)

        # repulsive terms between pedestrians
        f_ab = self.f_ab()
        w = np.expand_dims(self.w(e, -f_ab), -1)
        F_ab = w * f_ab

        # repulsive terms between pedestrians and boundaries
        F_aB = self.f_aB()

        # social force
        F = F0 + np.sum(F_ab, axis=1) + np.sum(F_aB, axis=1)

        # desired velocity (num. humans, 2)
        #       current velocities   time        overall force = velocity (i guess)
        w = self.state[:, 2:4] + self.delta_t * F #* 5

        # velocity
        v = self.capped_velocity(w)

        # update state
        self.state[:, 0:2] += v * self.delta_t
        self.state[:, 2:4] = v

        return self
