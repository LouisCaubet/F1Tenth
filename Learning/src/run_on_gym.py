import time
from stable_baselines3 import A2C
from stable_baselines3.ppo.ppo import PPO
import yaml
import gym
import numpy as np
from argparse import Namespace
import os
from gym_env import F1TenthGymEnv
import math

from config_loader import *
from network import Network
from lidar_preprocessing import convert_lidar_to_image, preprocess_image, restrict_lidar_fov

import torch

LOOKAHEAD_DISTANCE = 0.9
ANGLE_MIN = -math.pi


class Planner:

    def __init__(self, model1, model2=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.step = 0
        self.pred = np.array([[0, 0]])
        self.is_a2c = isinstance(model, A2C) or isinstance(model, PPO)

        self.pull_away_from_wall_steps = 0
        self.min_left = False

        self.use_second_model = model2 != None
        self.model2 = model2

    # Source: wall_follow.py
    def getRange(self, ranges, angle):
        # data: single message from topic /scan
        # angle IN RADIANS: between -45 to 225 degrees, where 0 degrees is directly to the right
        # Outputs length in meters to object with angle in lidar scan field of view
        # make sure to take care of nans etc.

        angle_increment = len(ranges) / (2*math.pi)

        def findFirstAcceptableMeasure(start_angle):
            index = math.floor((start_angle - ANGLE_MIN) / angle_increment)
            start_angle = ANGLE_MIN + angle_increment*index
            dist = ranges[index]

            return dist, start_angle

        dist_a, angle_a = findFirstAcceptableMeasure(angle)
        dist_b, angle_b = findFirstAcceptableMeasure(math.pi/2)

        theta = angle_b - angle_a

        alpha = math.atan((dist_a*math.cos(theta) - dist_b)/(dist_a * math.sin(theta)))
        D = dist_b * math.cos(alpha)
        Dnext = D + LOOKAHEAD_DISTANCE * math.sin(alpha)
        return Dnext

    def plan(self, input):
        self.step += 1

        if self.step % 5 != 0:
            return self.pred

        ranges = input['scans'][0]

        range_in_front = ranges[len(ranges) // 2]
        range_right = ranges[len(ranges) // 4]
        range_left = ranges[3*len(ranges) // 4]
        range_right_ahead = self.getRange(ranges, -math.pi/4)
        range_left_ahead = self.getRange(ranges, math.pi/4)
        arg_min = np.argmin(ranges)

        input = restrict_lidar_fov(ranges)
        input = convert_lidar_to_image(input)
        input = preprocess_image(input)
        input = input.to(self.device)

        if self.is_a2c:
            output, _ = self.model.predict(input, deterministic=True)
            steering_angle = output[0]
            velocity = 7*(1-2*abs(steering_angle))
        else:
            output = self.model(input[None, ...])
            steering_angle = output[:, 0].detach().item()
            velocity = output[:, 1].detach().item()

            if range_in_front < 2:
                print("Emergency avoidance enabled")
                self.min_left = range_left < range_right
                if self.min_left:
                    steering_angle = -0.4
                    print("Turning right")
                else:
                    steering_angle = 0.4
                    print("Turning left")
                velocity = 1

            right_min = min(range_right, range_right_ahead)
            left_min = min(range_left, range_left_ahead)

            # First wall avoidance protection: do not allow steering angle getting us nearer
            if right_min < 0.6 and steering_angle < 0:
                steering_angle = 0
            if left_min < 0.6 and steering_angle > 0:
                steering_angle = 0

            # Second protection: actively avoid wall
            if right_min < 0.4 and steering_angle < 0.05:
                steering_angle = 0.1
            if left_min < 0.4 and steering_angle > -0.05:
                steering_angle = -0.1

        self.pred = np.array([[steering_angle, velocity]])

        return self.pred


def render_callback(env_renderer):
    """
    Source: waypoint_follow.py (f1tenth_gym)
    If used, the camera will keep following the car.
    """
    # custom extra drawing function
    e = env_renderer

    # update camera to follow car
    x = e.cars[0].vertices[::2]
    y = e.cars[0].vertices[1::2]
    top, bottom, left, right = max(y), min(y), min(x), max(x)
    e.score_label.x = left
    e.score_label.y = top - 700
    e.left = left - 800
    e.right = right + 800
    e.top = top + 800
    e.bottom = bottom - 800


def run_on_gym(map_name, model, model2=None, render=False):
    map_conf_path = os.path.join(os.path.dirname(__file__), "gym_maps", map_name, "config_" + map_name + ".yaml")

    # Load the map
    with open(map_conf_path) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    map_path = os.path.join(os.path.dirname(__file__), "gym_maps", map_name, conf.map_path)

    start = time.time()

    # instantiating the environment
    racecar_env = gym.make('f110_gym:f110-v0', map=map_path, map_ext=conf.map_ext, num_agents=1)

    # TODO add some random to the starting pose
    obs, step_reward, done, info = racecar_env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))

    if render:
        # racecar_env.add_render_callback(render_callback)
        racecar_env.render()

    # instantiating your policy
    planner = Planner(model, model2)

    # simulation loop
    lap_time = 0.
    start = time.time()

    # loops when env not done
    while not done:
        # get action based on the observation
        actions = planner.plan(obs)
        # stepping through the environment
        obs, step_reward, done, info = racecar_env.step(actions)

        lap_time += step_reward

        if render:
            racecar_env.render(mode="human")

    return lap_time, time.time()-start, info["is_crashed"]


if __name__ == "__main__":
    if GYM_USE_RL_MODEL:
        env = F1TenthGymEnv()
        model = A2C.load(SB3_MODEL_TO_USE, env=env, print_system_info=True, force_reset=True)
        print(run_on_gym(GYM_MAP_NAME, model, None, RENDER))
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = Network()
        model = model.to(device)
        model.load_state_dict(torch.load(os.path.join(
            os.path.dirname(__file__), PTH_PP_MODEL), map_location=device))

        model.eval()

        print(run_on_gym(GYM_MAP_NAME, model, None, RENDER))
