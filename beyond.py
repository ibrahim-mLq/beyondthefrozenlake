import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy


environment_id = "MiniGrid-empty-8x8-v1"
SEED = 0

env = gym.make(environment_id)
env.reset(seed=SEED)

model = PPO(
    policy="MlpPolicy",
    env=env,
    n_steps=2000,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    learning_rate=3e-4,
    clip_range=0.2,
    ent_coef=0.0,
    verbose=1,
    seed=SEED,
)
model.learn(total_timesteps=200_000)

mean_return, std_return = evaluate_policy(model, env, n_eval_episodes=20, deterministic=True)
print("eval return: ", mean_return, "+/-", std_return)
model.save("ppo_minigrid_baseline")
env.close()