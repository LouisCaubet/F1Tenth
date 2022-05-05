import gym
from gym import spaces
import numpy as np
from argparse import Namespace

import os
import yaml

from config_loader import *
from lidar_preprocessing import convert_lidar_to_image, preprocess_image


class F1TenthGymEnv(gym.Env):
    def __init__(self):
        # Define observation and action spaces

        # An observation is a preprocessed LIDAR input, ie. a 3*224*224 tensor of floats in [0,1]
        # self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(3, 224, 224), dtype=np.float64)
        self.observation_space = spaces.Box(low=0.0, high=10.0, shape=(1080,), dtype=np.float64)
        # An action is a couple steering_angle, velocity, with steering angle in [-0.41, 0.41] and velocity in [1, 7]
        # self.action_space = spaces.Box(low=np.array([-0.41, 1.]), high=np.array([0.41, 7.]), dtype=np.float64)
        self.action_space = spaces.Box(low=np.array([-0.41]), high=np.array([0.41]), dtype=np.float64)

        # Create the underlying f110-gym env
        map_conf_path = os.path.join(os.path.dirname(__file__), "gym_maps",
                                     GYM_MAP_NAME, "config_" + GYM_MAP_NAME + ".yaml")
        with open(map_conf_path) as file:
            conf_dict = yaml.load(file, Loader=yaml.FullLoader)
        self.conf = Namespace(**conf_dict)
        map_path = os.path.join(os.path.dirname(__file__), "gym_maps", GYM_MAP_NAME, self.conf.map_path)
        self.env = gym.make('f110_gym:f110-v0', map=map_path, map_ext=self.conf.map_ext, num_agents=1)

    def step(self, action):
        # Execute one time step within the environment
        steering_angle = action[0]
        velocity = 7*(1-2*abs(steering_angle))
        raw_actions = np.array([[steering_angle, velocity]])
        obs, step_reward, done, info = self.env.step(raw_actions)

        # Process the observation
        # preprocessed_obs = convert_lidar_to_image(obs['scans'][0])
        # preprocessed_obs = preprocess_image(preprocessed_obs)

        self.total_time += step_reward
        self.total_dist += step_reward*velocity

        ranges = obs['scans'][0]
        distance_to_wall = min(ranges)
        preprocessed_obs = [max(r, 10) for r in ranges]

        # Compute the reward
        if info['is_crashed'] == 1:
            reward = -1000000
        else:
            coeff = -1 if distance_to_wall < 0.5 else 1
            reward = (0.3-steering_angle) + (velocity-2)/7 + (distance_to_wall-0.5)

        return preprocessed_obs, reward, done, info

    def reset(self):
        # Reset the state of the environment to an initial state
        obs, step_reward, done, info = self.env.reset(np.array([[self.conf.sx, self.conf.sy, self.conf.stheta]]))
        # preprocessed_obs = convert_lidar_to_image(obs['scans'][0])
        # preprocessed_obs = preprocess_image(preprocessed_obs)
        preprocessed_obs = [max(r, 10) for r in obs['scans'][0]]
        self.total_time = 0
        self.total_dist = 0
        return preprocessed_obs

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        # Use run_on_gym for visualization.
        pass
