""" This file defines the forward kinematics cost function. """
import copy

import numpy as np

from gps.algorithm.cost.config import COST_DOOR
from gps.algorithm.cost.cost import Cost
from gps.algorithm.cost.cost_utils import get_ramp_multiplier
from gps.proto.gps_pb2 import JOINT_ANGLES, END_EFFECTOR_POINTS, \
        END_EFFECTOR_POINT_JACOBIANS


class CostDoor(Cost):
    """
    Forward kinematics cost function. Used for costs involving the end
    effector position.
    """
    def __init__(self, hyperparams):
        config = copy.deepcopy(COST_DOOR)
        config.update(hyperparams)
        Cost.__init__(self, config)

    def eval(self, sample):
        """
        Evaluate forward kinematics (end-effector penalties) cost.
        Temporary note: This implements the 'joint' penalty type from
            the matlab code, with the velocity/velocity diff/etc.
            penalties removed. (use CostState instead)
        Args:
            sample: A single sample.
        """
        T = sample.T
        dX = sample.dX
        dU = sample.dU

        wpm = get_ramp_multiplier(
            self._hyperparams['ramp_option'], T,
            wp_final_multiplier=self._hyperparams['wp_final_multiplier']
        )
        #wp = self._hyperparams['wp'] * np.expand_dims(wpm, axis=-1)

        # Initialize terms.
        l = np.zeros(T)
        lu = np.zeros((T, dU))
        lx = np.zeros((T, dX))
        luu = np.zeros((T, dU, dU))
        lxx = np.zeros((T, dX, dX))
        lux = np.zeros((T, dU, dX))

        #target_end_effector = np.array([0.695, 0.36, 0.55, 0.695, 0.36, 0.45, 0, 0, 0, 0, 0, 0])
        start = 16
        target_end_effector = sample.get(END_EFFECTOR_POINTS)[:,6:12]
        gripper = sample.get(END_EFFECTOR_POINTS)
        for t in range(T):
            lx[t][16] = (gripper[t][0] - target_end_effector[t][0])
            lx[t][17] = (gripper[t][1] - target_end_effector[t][1])
            lx[t][18] = (gripper[t][2] - target_end_effector[t][2])

            lx[t][19] = (gripper[t][3] - target_end_effector[t][3])
            lx[t][20] = (gripper[t][4] - target_end_effector[t][4])
            lx[t][21] = (gripper[t][5] - target_end_effector[t][5])

            lxx[t][16][16] = 1
            lxx[t][17][17] = 1
            lxx[t][18][18] = 1

            lxx[t][19][19] = 1
            lxx[t][20][20] = 1
            lxx[t][21][21] = 1

            l[t] = (0.5)*np.linalg.norm(gripper[t][0:6] - target_end_effector[t][0:6])**2

        """
        print "door cost"
        print l[5]
        print lx[5]
        #print lxx[5]
        print "\n"
        """


        """
        # Choose target.
        gripper = sample.get(END_EFFECTOR_POINTS)[:,0:3]
        handle = sample.get(END_EFFECTOR_POINTS)[:,3:6]


        #22 23 24 25 26 27

        for t in range(T):
            lx[t][16] = (gripper[t][0] - handle[t][0])
            lx[t][17] = (gripper[t][1] - handle[t][1])
            lx[t][18] = (gripper[t][2] - handle[t][2])

            lx[t][19] = -1*(gripper[t][0] - handle[t][0])
            lx[t][20] = -1*(gripper[t][1] - handle[t][1])
            lx[t][21] = -1*(gripper[t][2] - handle[t][2])


            lxx[t][16][16] = 1
            lxx[t][16][19] = -1
            lxx[t][17][17] = 1
            lxx[t][17][20] = -1
            lxx[t][18][18] = 1
            lxx[t][18][21] = -1

            lxx[t][19][19] = -1
            lxx[t][19][16] = -1
            lxx[t][20][20] = -1
            lxx[t][20][17] = -1
            lxx[t][21][21] = -1
            lxx[t][21][18] = -1

            l[t] = (0.5)*np.linalg.norm(gripper[t] - handle[t])**2
        """

        """
        # TODO - These should be partially zeros so we're not double
        #        counting.
        #        (see pts_jacobian_only in matlab costinfos code)
        jx = sample.get(END_EFFECTOR_POINT_JACOBIANS)

        # Evaluate penalty term. Use estimated Jacobians and no higher
        # order terms.
        jxx_zeros = np.zeros((T, dist.shape[1], jx.shape[2], jx.shape[2]))
        l, ls, lss = self._hyperparams['evalnorm'](
            wp, dist, jx, jxx_zeros, self._hyperparams['l1'],
            self._hyperparams['l2'], self._hyperparams['alpha']
        )
        # Add to current terms.
        sample.agent.pack_data_x(lx, ls, data_types=[JOINT_ANGLES])
        sample.agent.pack_data_x(lxx, lss,
                                 data_types=[JOINT_ANGLES, JOINT_ANGLES])
        """

        return l, lx, lu, lxx, luu, lux

