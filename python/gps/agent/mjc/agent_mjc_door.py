""" This file defines an agent for the MuJoCo simulator environment. """
import copy

import numpy as np

import mujoco_py
from mujoco_py.mjlib import mjlib
from mujoco_py.mjtypes import *

from gps.agent.mjc.agent_mjc import AgentMuJoCo

from gps.agent.agent_utils import generate_noise, setup
from gps.agent.config import AGENT_MUJOCO
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, \
        END_EFFECTOR_POINT_JACOBIANS, ACTION, RGB_IMAGE, RGB_IMAGE_SIZE, \
        CONTEXT_IMAGE, CONTEXT_IMAGE_SIZE

from gps.sample.sample import Sample

class AgentMuJoCoDoor(AgentMuJoCo):
    """
    All communication between the algorithms and MuJoCo is done through
    this class.
    """
    def __init__(self, hyperparams):
        super(AgentMuJoCoDoor, self).__init__(hyperparams)

    def _setup_conditions(self):
        """
        Helper method for setting some hyperparameters that may vary by
        condition.
        """
        conds = self._hyperparams['conditions']
        for field in ('x0', 'x0var', 'pos_body_idx', 'pos_body_offset',
                      'noisy_body_idx', 'noisy_body_var', 'filename'):
            self._hyperparams[field] = setup(self._hyperparams[field], conds)
        
    def _init_sample(self, condition):
        """
        Construct a new sample and fill in the first time step.
        Args:
            condition: Which condition to initialize.
        """
        sample = Sample(self)

        # Initialize world/run kinematics
        self._init(condition)


        #print self._hyperparams['offset']

        # Initialize sample with stuff from _data
        data = self._model[condition].data
        sample.set(JOINT_ANGLES, data.qpos.flatten(), t=0)
        sample.set(JOINT_VELOCITIES, data.qvel.flatten(), t=0)
        eepts = data.site_xpos.flatten()
        


        #print eepts
        #A points
        baseindex = 3
        for i in range(3):
            eepts[i+baseindex] += self._hyperparams['offsetA'][i]


        #B points
        baseindex = 6
        for i in range(3):
            eepts[i+baseindex] += self._hyperparams['offsetB'][i]
        #print eepts
        #print "\n"

        sample.set(END_EFFECTOR_POINTS, eepts, t=0)
        sample.set(END_EFFECTOR_POINT_VELOCITIES, np.zeros_like(eepts), t=0)
        jac = np.zeros([eepts.shape[0], self._model[condition].nq])
        for site in range(eepts.shape[0] // 3):
            idx = site * 3
            temp = np.zeros((3, jac.shape[1]))
            mjlib.mj_jacSite(self._model[condition].ptr, self._model[condition].data.ptr, temp.ctypes.data_as(POINTER(c_double)), 0, site)
            jac[idx:(idx+3), :] = temp
        sample.set(END_EFFECTOR_POINT_JACOBIANS, jac, t=0)

        # save initial image to meta data
        img_string, width, height = self._viewer[condition].get_image()
        img = np.fromstring(img_string, dtype='uint8').reshape(height, width, 3)[::-1,:,:]
        img_data = np.transpose(img, (1, 0, 2)).flatten()

        # if initial image is an observation, replicate it for each time step
        if CONTEXT_IMAGE in self.obs_data_types:
            sample.set(CONTEXT_IMAGE, np.tile(img_data, (self.T, 1)), t=None)
        else:
            sample.set(CONTEXT_IMAGE, img_data, t=None)
        sample.set(CONTEXT_IMAGE_SIZE, np.array([self._hyperparams['image_channels'],
                                                self._hyperparams['image_width'],
                                                self._hyperparams['image_height']]), t=None)
        # only save subsequent images if image is part of observation
        if RGB_IMAGE in self.obs_data_types:
            sample.set(RGB_IMAGE, img_data, t=0)
            sample.set(RGB_IMAGE_SIZE, [self._hyperparams['image_channels'],
                                        self._hyperparams['image_width'],
                                        self._hyperparams['image_height']], t=None)
        return sample

    def _set_sample(self, sample, mj_X, t, condition):
        """
        Set the data for a sample for one time step.
        Args:
            sample: Sample object to set data for.
            mj_X: Data to set for sample.
            t: Time step to set for sample.
            condition: Which condition to set.
        """
        sample.set(JOINT_ANGLES, np.array(mj_X[self._joint_idx]), t=t+1)
        sample.set(JOINT_VELOCITIES, np.array(mj_X[self._vel_idx]), t=t+1)
        curr_eepts = self._data.site_xpos.flatten()


        #print curr_eepts
        #top points
        baseindex = 3
        for i in range(3):
            curr_eepts[i+baseindex] += self._hyperparams['offsetA'][i]


        #bottom points
        baseindex = 6
        for i in range(3):
            curr_eepts[i+baseindex] += self._hyperparams['offsetB'][i]
        #print curr_eepts
        #print "\n"

        sample.set(END_EFFECTOR_POINTS, curr_eepts, t=t+1)
        prev_eepts = sample.get(END_EFFECTOR_POINTS, t=t)
        eept_vels = (curr_eepts - prev_eepts) / self._hyperparams['dt']
        sample.set(END_EFFECTOR_POINT_VELOCITIES, eept_vels, t=t+1)
        jac = np.zeros([curr_eepts.shape[0], self._model[condition].nq])
        for site in range(curr_eepts.shape[0] // 3):
            idx = site * 3
            temp = np.zeros((3, jac.shape[1]))
            mjlib.mj_jacSite(self._model[condition].ptr, self._model[condition].data.ptr, temp.ctypes.data_as(POINTER(c_double)), 0, site)
            jac[idx:(idx+3), :] = temp

        sample.set(END_EFFECTOR_POINT_JACOBIANS, jac, t=t+1)
        if RGB_IMAGE in self.obs_data_types:
            img_string, width, height = self._viewer[condition].get_image()#CHANGES
            img = np.fromstring(img_string, dtype='uint8').reshape(height, width, 3)[::-1,:,:]
            sample.set(RGB_IMAGE, np.transpose(img, (2, 1, 0)).flatten(), t=t+1)
