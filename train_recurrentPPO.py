import gymnasium as gym
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import minigrid
from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper
from report_logger import make_report_logger

environment_id = "MiniGrid-Empty-16x16-v0"
SEED = 1

env = gym.make(environment_id, render_mode="rgb_array")
env = RGBImgObsWrapper(env, tile_size=8)
env = ImgObsWrapper(env)
env = Monitor(env)

obs, info = env.reset(seed=SEED)

model = RecurrentPPO(
    policy="CnnLstmPolicy",
    env=env,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    learning_rate=3e-4,
    clip_range=0.2,
    ent_coef=0.0,
    device="cuda",
    verbose=1,
    seed=SEED,
)

call_back = make_report_logger(
    env_id=environment_id,
    out_dir="runs/rppo/MiniGrid-Empty-16x16-v0",
    run_tag=environment_id,
)

model.learn(total_timesteps=200_000, callback=call_back)

mean_return, std_return = evaluate_policy(model, env, n_eval_episodes=20, deterministic=True)
print("eval return:", mean_return, "+/-", std_return)

model.save("rppo_minigrid_baseline")
env.close()
