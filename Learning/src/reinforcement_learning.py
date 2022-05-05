import os
import json
import time

import torch
from torch import nn

from stable_baselines3.a2c.a2c import A2C
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from config_loader import *
from gym_env import F1TenthGymEnv
from network import Network
from run_on_gym import run_on_gym

env = F1TenthGymEnv()


class LidarFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, obs_space, feature_dim, network: Network):
        super().__init__(obs_space, feature_dim)
        self.feature_extraction = network.feature_extractor
        param: nn.parameter.Parameter
        for param in self.feature_extraction.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.feature_extraction(x)


def get_pretrained_feature_extractor_kwargs():
    cfg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "config.json"))
    CONFIG = json.load(open(cfg_path, "r", encoding="utf8"))
    DEFAULT_PTH_FILE = CONFIG["PTH_MODEL_TO_USE"]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pretrained = Network()
    pretrained = pretrained.to(device)
    pretrained.load_state_dict(torch.load(os.path.join(
        os.path.dirname(__file__), DEFAULT_PTH_FILE), map_location=device))

    return {"network": pretrained, "feature_dim": 576}


policy_kwargs = {
    # "features_extractor_class": LidarFeatureExtractor,
    # "features_extractor_kwargs": get_pretrained_feature_extractor_kwargs(),
    # "normalize_images": False,
    # "net_arch": [{"vf": [64, 64], "pi": [64, 64, 32]}],
    "activation_fn": nn.Tanh
}


def load_model(name):
    model = A2C.load(name, env=env, print_system_info=True, force_reset=True)
    return model


def create_model():
    model = A2C(
        policy="MlpPolicy",
        env=env,
        learning_rate=0.0005,
        # policy_kwargs=policy_kwargs,
        verbose=1,
        gamma=0.99
    )
    return model


TOTAL_TIMESTEPS = 1000
current_step = 0


def training_callback(arg1, arg2):
    global current_step
    current_step += 1
    if current_step % 100 == 0:
        print("    Step", current_step, "/", TOTAL_TIMESTEPS)
        run_on_gym(GYM_MAP_NAME, model, True)


if __name__ == "__main__":
    if RL_RESUME_TRAINING:
        model = load_model(SB3_MODEL_TO_USE)
    else:
        model = create_model()

    start = time.time()
    model = model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=training_callback)
    print("Total training time:", time.time()-start)

    print("Training complete. Saving...")
    model.save("f1tenth_rl_trained_1k_resumed")
