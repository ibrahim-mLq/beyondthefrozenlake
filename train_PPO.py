import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import minigrid
from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper
from report_logger import make_report_logger



environment_id = "MiniGrid-Empty-8x8-v0"
SEED = 0

env = gym.make(environment_id, render_mode="rgb_array")
env = RGBImgObsWrapper(env, tile_size=8)
env = ImgObsWrapper(env)

obs, info = env.reset(seed=SEED)

model = PPO(
    policy="CnnPolicy",
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

call_back = make_report_logger(env_id=environment_id, out_dir="runs/ppo", run_tag=environment_id) #custom report logger

model.learn(total_timesteps=102_400, callback=call_back) #starts the actual training, and logs following call_back config

mean_return, std_return = evaluate_policy(model, env, n_eval_episodes=20, deterministic=True)
print("eval return: ", mean_return, "+/-", std_return)
model.save("ppo_minigrid_baseline")
env.close()