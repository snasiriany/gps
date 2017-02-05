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


    def _setup_world(self, filename):
        super(AgentMuJoCoDoor, self)._setup_world(filename)
        self.images = []
        self.count = 1


    def sample(self, policy, condition, verbose=True, save=True, noisy=True):
        """
        Runs a trial and constructs a new sample containing information
        about the trial.
        Args:
            policy: Policy to to used in the trial.
            condition: Which condition setup to run.
            verbose: Whether or not to plot the trial.
            save: Whether or not to store the trial into the samples.
            noisy: Whether or not to use noise during sampling.
        """
        # Create new sample, populate first time step.
        new_sample = self._init_sample(condition)
        mj_X = self._hyperparams['x0'][condition]
        U = np.zeros([self.T, self.dU])
        if noisy:
            noise = generate_noise(self.T, self.dU, self._hyperparams)
        else:
            noise = np.zeros((self.T, self.dU))
        if np.any(self._hyperparams['x0var'][condition] > 0):
            x0n = self._hyperparams['x0var'] * \
                    np.random.randn(self._hyperparams['x0var'].shape)
            mj_X += x0n
        noisy_body_idx = self._hyperparams['noisy_body_idx'][condition]
        if noisy_body_idx.size > 0:
            for i in range(len(noisy_body_idx)):
                idx = noisy_body_idx[i]
                var = self._hyperparams['noisy_body_var'][condition][i]
                temp = np.copy(body_pos)  #CHANGES
                temp[idx, :] += var * np.random.randn(1, 3)
                self._model[condition].body_pos = temp

        cam_pos = self._hyperparams['camera_pos']#CHANGES
        self._viewer_main.set_model(self._model[condition])
        self._viewer_bot.set_model(self._model[condition])

        self._viewer_main.distance = 5.0 #TODO mayeb don't hard code this
        self._viewer_bot.cam.lookat[0] = cam_pos[0]
        self._viewer_bot.cam.lookat[1] = cam_pos[1]
        self._viewer_bot.cam.lookat[2] = cam_pos[2]
        self._viewer_bot.cam.distance = cam_pos[3]
        self._viewer_bot.cam.elevation = cam_pos[4]
        self._viewer_bot.cam.azimuth = cam_pos[5]

        # Take the sample.
        for t in range(self.T):
            X_t = new_sample.get_X(t=t)
            obs_t = new_sample.get_obs(t=t)
            mj_U = policy.act(X_t, obs_t, t, noise[t, :])
            U[t, :] = mj_U
            if verbose:
                self._viewer_main.loop_once()
                self._viewer_bot.loop_once()
                self._viewer[condition].loop_once()


            self._store_image(t)

            if (t + 1) < self.T:
                for _ in range(self._hyperparams['substeps']):
                    self._model[condition].data.ctrl = mj_U
                    self._model[condition].step()
                #TODO: Some hidden state stuff will go here.
                mj_X = np.concatenate([self._model[condition].data.qpos, self._model[condition].data.qvel]).flatten()
                self._data = self._model[condition].data
                self._set_sample(new_sample, mj_X, t, condition)
        new_sample.set(ACTION, U)
        if save:
            self._samples[condition].append(new_sample)

        self.save_gif()
        return new_sample

    def RGB2video(self, data, nameFile='video', verbosity=1, indent=0, framerate=24, codec='mpeg4', threads=4):
        '''

        :param data: np.array N x H x W x 3
        :param nameFile:
        :param verbosity:
        :param indent:
        :return:
        '''
        # Write to FFMPEG
        import imageio
        #imageio.plugins.ffmpeg.download()
        from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter as fwv


        extension = '.mp4'  # '.avi'
        fullNameVideo = nameFile + extension
        n_frame = data.shape[0]
        resolution = (data.shape[2], data.shape[1])  # (W, H)
        #print('Resolution: %d x %d fps: %d n_frames: %d' % (resolution[0], resolution[1], framerate, n_frame))
        #print('Saving to file: ' + fullNameVideo)
        a = fwv(filename=fullNameVideo, codec=codec, size=resolution, fps=framerate, preset="slower", threads=threads)

        for i in range(n_frame):
            # frame = np.swapaxes(data[i, :], 1, 2)
            frame = data[i, :].astype('uint8')
            assert np.all(0 <= frame) and np.all(frame <= 255), 'Value of the pixels is not in [0-255]'
            a.write_frame(frame)
            # plt.figure()
            # plt.imshow(frame/255)
            # plt.show
        a.close()
        # rlog.cnd_status(current_verbosity=verbosity, necessary_verbosity=1, f=0)
        # TODO: fix circular  import rlog
        return 0

    def save_gif(self):
        self.RGB2video(np.array(self.images), nameFile=self._hyperparams['trial_dir'] + "video_" + str(self.count), framerate=1/self._hyperparams['dt'])
        self.images = []
        self.count += 1

    def _store_image(self,t):
        """
        store image at time index t
        """

        """
        if self._hyperparams['additional_viewer']:
            self._large_viewer.loop_once()
        """

        img_string, width, height = self._viewer_bot.get_image()
        largeimage = np.fromstring(img_string, dtype='uint8').reshape(AGENT_MUJOCO['image_height'], AGENT_MUJOCO['image_width'], 3)[::-1,:,:]
        self.images.append(largeimage)
        
        """
        self.large_images.append(largeimage)
        """

        ######
        #small viewer:
        """
        self.model_nomarkers.data.qpos = self._model.data.qpos
        self.model_nomarkers.data.qvel = self._model.data.qvel
        self.model_nomarkers.step()
        self._small_viewer.loop_once()

        img_string, width, height = self._small_viewer.get_image()
        img = np.fromstring(img_string, dtype='uint8').reshape((height, width, self._hyperparams['image_channels']))[::-1,:,:]
        """



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


        #C or Center points
        baseindex = 9
        for i in range(3):
            eepts[i+baseindex] += self._hyperparams['offsetC'][i]
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


        #C or Center points
        baseindex = 9
        for i in range(3):
            curr_eepts[i+baseindex] += self._hyperparams['offsetC'][i]
        #print eepts
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
