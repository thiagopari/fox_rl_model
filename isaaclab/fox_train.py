"""Train the Fox walking policy with Isaac Lab's stock rsl_rl script (PPO), headless.

    cd ~/Documents/fox_rl_model && ~/isaacenv/bin/python isaaclab/fox_train.py --task Fox-Velocity-Flat --headless
Checkpoints: logs/rsl_rl/fox_flat/<date>/model_<iter>.pt (relative to the current directory). Any train.py option works
(--num_envs, --max_iterations, --resume ...). Watch progress: tensorboard --logdir logs/rsl_rl/fox_flat
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.expanduser("~/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py")
sys.path[:0] = [HERE, os.path.dirname(TRAIN)]   # fox_cfg / fox_tasks, and train.py's own cli_args helper
import fox_tasks  # noqa: E402,F401  (registers Fox-Velocity-Flat and Fox-Velocity-Flat-Play)

sys.argv[0] = TRAIN
runpy.run_path(TRAIN, run_name="__main__")
