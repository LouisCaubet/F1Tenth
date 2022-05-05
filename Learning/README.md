# F1Tenth: End-to-End Self-Driving using MobileNet Vision

Here we use MobileNet's feature extraction, pretrained on the ImageNet dataset, to transform raw LIDAR inputs into steering angle and velocity commands.

Here's some detail on what each script (in the src/ subfolder) is doing:

-   network.py : Defines the neural network, as well as the training and testing code.

-   lidar_preprocessing.py : Contains code to transform a LIDAR input into a PIL image ready to be given to MobileNet.

-   generate_dataset.py : Applies the preprocessing to each entry of a dataset (in .pickle format, generated using Lab3 code) and saves the result on disk

-   lidar_dataset.py : Loads the pre-generated datasets into a PyTorch Dataset.

-   supervised_training.py : Entry point to call training code.

-   node.py : ROS node to execute the neural network on ROS. Please set PYTORCH_WEIGHTS_PATH to the pretrained model path before running.

-   run_on_gym.py : enables running the neural network on Gym, to be able to test it in a faster environment than the ROS simulator and execute reinforcement learning on it.

-   gym_env.py : custom gym environment, based on f1tenth_gym, to train our racecar using RL.

-   reinforcement_learning.py : implements RL training using stable-baselines3

## Requirements

PyTorch and Torchvision.

If you want to execute the supervised training, you'll need to download the datasets from http://short.binets.fr/f1tenth-datasets, or generate them using the wall following algorithm (Lab 3).

To execute the driving in OpenAI Gym, you need to install our custom version of `f1tenth-gym`:

```
git clone https://gitlab.binets.fr/louis.caubet/f1tenth-gym.git
cd f1tenth-gym
pip3 install --user -e gym/
```

To verify that the installation was successful, you may run `waypoint_follow.py` in the `f1tenth_gym/examples` folder.

To execute the reinforcement learning algorithms, you need to install `stable-baselines3`.

## Configuration

The configuration file is `config.json`:

-   DATASET_FOLDER must be set to where you downloaded the [datasets](http://short.binets.fr/f1tenth-datasets).

-   DEBUG_GENERATE_ONLY_ONE_DATASET: If set to true, the supervised training code will only use the first dataset from the list, which shortens execution time.

-   DATASETS: List containing the names of the datasets to use (without the .pickle extension)

-   PTH_MODEL_TO_USE : Name of the \*.pth weight file used by scripts `node.py` and `run_on_gym.py`

-   SB3_MODEL_TO_USE : Name of the `stable-baselines3` network used in RL.

-   GYM_MAP : The map to use in the OpenAI Gym simulator

-   GYM_ENABLE_RENDER: If disabled, Gym will run with no GUI and using accelerated physics, and return the time needed for a lap. If enables, physics will run real-time and the simulator will be rendered.

-   GYM_USE_RL_MODEL: Whether `run_on_gym.py` should execute the `.pth` model or the `stable-baselines3` model.

-   RL_RESUME_TRAINING: Whether `reinforcement_learning.py` should resume training on `SB3_MODEL_TO_USE` or start from scratch.

## Legacy

This folder contains another approach to self-driving: instead of predicting velocity and steering angle using a regression network, we use different classes and a classification network.

This approach makes for a much faster training and achieves reliable results, but we are no longer pursuing it as we've been able to execute the bigger MobileNet network in real-time on the car.

## Sources

-   Mingyu Park, Hyeonseok Kim and Seongkeun Park, _A Convolutional Neural Network-Based End-to-End Self-Driving Using LiDAR and Camera Fusion: Analysis - Perspectives in a Real-World Environment_, MDPI, 2021

-   Michael Bosello, F1Tenth-RL, GitHub: https://github.com/MichaelBosello/f1tenth-RL

-   O’Kelly, Matthew and Zheng, Hongrui and Karthik, Dhruv and Mangharam, Rahul, _F1TENTH: An Open-source Evaluation Environment for Continuous Control and Reinforcement Learning_, NeurIPS 2019 Competition and Demonstration Track, 2020.
