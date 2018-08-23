"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from https://webdocs.cs.ualberta.ca/~sutton/book/code/pole.c
"""
import os,  inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)

import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import subprocess
import pybullet as p
import pybullet_data
from pkg_resources import parse_version

logger = logging.getLogger(__name__)

class BikeBulletEnvV2(gym.Env):
  metadata = {
    'render.modes': ['human', 'rgb_array'],
    'video.frames_per_second' : 30
  }

  def __init__(self, renders=True):
    # start the bullet physics server
    self._renders = renders
    if (renders):
        p.connect(p.GUI)
    else:
        p.connect(p.DIRECT)
    observation_high = np.array([
        np.finfo(np.float32).max,
        np.finfo(np.float32).max,
        np.finfo(np.float32).max,
        np.finfo(np.float32).max,
        np.finfo(np.float32).max,
        np.finfo(np.float32).max,
        np.finfo(np.float32).max,
        np.finfo(np.float32).max,
        np.finfo(np.float32).max,
        np.finfo(np.float32).max,
        np.finfo(np.float32).max,
        np.finfo(np.float32).max,
        np.finfo(np.float32).max])

    action_high = np.array([
          np.finfo(np.float32).max,
          np.finfo(np.float32).max
          ])


    self.action_space=spaces.Box(-action_high,action_high)
    self.observation_space = spaces.Box(-observation_high, observation_high)

    self.target_V=1
    self.target_degree=0

    self.theta_threshold_radians = 1
    self.x_threshold = 2.4
    self._seed()
#    self.reset()
    self.viewer = None
    self._configure()

  def _configure(self, display=None):
    self.display = display

  def _seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]


  def _step(self, action):
    xposbefore = p.getLinkState(self.bike, 0)[0][0]
    # p.setJointMotorControl2(self.bike, 0, p.VELOCITY_CONTROL, targetVelocity=1,force=bar_theta)



    p.setJointMotorControl2(self.bike, 0, p.POSITION_CONTROL,
                            targetPosition=action[0], force=action[1])
    p.setJointMotorControl2(self.bike, 1, p.VELOCITY_CONTROL, targetVelocity=15, force=0)
    # p.setJointMotorControl2(self.bike, 2, p.VELOCITY_CONTROL, targetVelocity=self.target_V, force=-(back_wheel_v-self.target_V)*6)
    p.setJointMotorControl2(self.bike, 2, p.VELOCITY_CONTROL, targetVelocity=15, force=20)
    p.stepSimulation()
    xposafter = p.getLinkState(self.bike, 0)[0][0]
    ###joint#####
    """
    0 means frame_to_handlebar
    1 means handlebar_to_frontwheel
    2 means frame_to_backwheel
    """

    self.linkstate = p.getLinkState(self.bike, 0)[0:2]
    frame_world_position,frame_world_orientation = self.linkstate
    self.state = p.getJointState(self.bike, 0)[0:2] + p.getJointState(self.bike, 1)[0:2]+p.getJointState(self.bike, 2)[0:2]+frame_world_orientation+frame_world_position
    #print(self.state)
    #print('-------------------------------------------------------')
    #print(action)
    #print(action[0] * p.getJointState(self.bike, 0)[0])

    reward_forward = (xposafter - xposbefore)/self.timeStep
    reward_ctrl = - 0.1 * np.square(action[1])

    # print('degree  ',frame_world_orientation[2])
    done =  abs(frame_world_position[2]) < 0.5 #or abs(p.getJointState(self.bike, 0)[0]) > 2.0 or abs(frame_world_position[2]) > 1.5
    #
    # done = False
    reward = 0
    if not done:
        reward += 1
    reward += reward_forward + reward_ctrl
    return np.array(self.state), reward, done, {}

  def _reset(self):
#    print("-----------reset simulation---------------")

    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    self.plane = p.loadURDF("plane100.urdf", [0, 0, 0])
    perturb_angle = self.np_random.uniform(low=-0.01, high=0.01)
    self.bike = p.loadURDF(os.path.join(pybullet_data.getDataPath(),"bicycle/bike.urdf"),[0,0,1.0], [3*perturb_angle,0,0,1],useFixedBase=False)


    p.changeDynamics(self.plane, -1, mass=20, lateralFriction=1, linearDamping=0, angularDamping=0)
    p.changeDynamics(self.bike, 1, lateralFriction=1, linearDamping=0, angularDamping=0)
    p.changeDynamics(self.bike, 2, lateralFriction=1, linearDamping=0, angularDamping=0)
    self.timeStep = 1.0/2000.
    p.setGravity(0,0, -9.8)
    p.setTimeStep(self.timeStep)
    p.setRealTimeSimulation(1)

    def angle_to_num(angle):
        return angle/180*3.14


    bar_angle = self.np_random.uniform(low=angle_to_num(-20), high=angle_to_num(20))
    p.resetJointState(self.bike, 0, bar_angle)
    p.resetJointState(self.bike, 1, 0)
    p.resetJointState(self.bike, 2, 0)

    p.setJointMotorControl2(self.bike, 0, p.POSITION_CONTROL, targetPosition=-0.5*bar_angle, force=0.01)
    p.setJointMotorControl2(self.bike, 1, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
    # p.setJointMotorControl2(self.bike, 2, p.VELOCITY_CONTROL, targetVelocity=self.target_V, force=-(back_wheel_v-self.target_V)*6)
    p.setJointMotorControl2(self.bike, 2, p.VELOCITY_CONTROL, targetVelocity=15, force=10)
    self.linkstate = p.getLinkState(self.bike, 0)[0:2]
    frame_world_position, frame_world_orientation = self.linkstate
    self.state = p.getJointState(self.bike, 0)[0:2] + p.getJointState(self.bike, 1)[0:2]+p.getJointState(self.bike, 2)[0:2]+frame_world_orientation+frame_world_position

    return np.array(self.state)

  def _render(self, mode='human', close=False):
      distance = 5
      yaw = 0
      humanPos, humanOrn = p.getBasePositionAndOrientation(self.bike)
      humanBaseVel = p.getBaseVelocity(self.bike)
      # print("frame",frame, "humanPos=",humanPos, "humanVel=",humanBaseVel)
      if (True):
          camInfo = p.getDebugVisualizerCamera()
          curTargetPos = camInfo[11]
          distance = camInfo[10]
          yaw = camInfo[8]
          pitch = camInfo[9]
          targetPos = [0.95 * curTargetPos[0] + 0.05 * humanPos[0], 0.95 * curTargetPos[1] + 0.05 * humanPos[1],
                       curTargetPos[2]]
          p.resetDebugVisualizerCamera(distance, yaw, pitch, targetPos);
      return

  if parse_version(gym.__version__)>=parse_version('0.9.6'):
    render = _render
    reset = _reset
    seed = _seed
    step = _step
