import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
import minigrid
from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper
from rl_dqn import make_report_logger_dqn

environment_id = "MiniGrid-Empty-16x16-v0"
SEED = 1

# ---- Env (DQN expects VecEnv; use VecTransposeImage for CnnPolicy) ----
def make_env():
    env = gym.make(environment_id, render_mode="rgb_array")
    env = RGBImgObsWrapper(env, tile_size=8)
    env = ImgObsWrapper(env)
    env = Monitor(env)
    env.reset(seed=SEED)
    return env

env = DummyVecEnv([make_env])
env = VecTransposeImage(env)  # (H,W,C) -> (C,H,W) for CNN

# ---- Model ----
model = DQN(
    policy="CnnPolicy",
    env=env,
    buffer_size=200_000,
    learning_starts=5_000,
    batch_size=64,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    target_update_interval=1_000,
    exploration_fraction=0.1,
    exploration_final_eps=0.05,
    learning_rate=1e-4,
    device="cuda",
    verbose=1,
    seed=SEED,
)

callback = make_report_logger_dqn(
    env_id=environment_id,
    out_dir="runs/dqn/MiniGrid-Empty-16x16-v0",
    run_tag=environment_id,
    log_every=2048,  # match PPO n_steps for fair comparison
)

# ---- Train ----
model.learn(total_timesteps=200_000, callback=callback)

# ---- Eval ----
mean_return, std_return = evaluate_policy(model, env, n_eval_episodes=20, deterministic=True)
print("eval return:", mean_return, "+/-", std_return)

model.save("dqn_minigrid_baseline")
env.close()
