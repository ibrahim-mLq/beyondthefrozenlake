# report_logger_dqn.py
import os
import json
import time
from datetime import datetime

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class ReportLoggerCallbackDQN(BaseCallback):
    """
    DQN-specific report logger (step-based, since DQN has no PPO-style rollouts).

    Logs every `log_every` env steps:
      - time/total_timesteps, time/fps
      - rollout/ep_rew_mean, rollout/ep_len_mean (from ep_info_buffer; requires Monitor/VecMonitor)
      - train/loss (if available in logger)
      - train/exploration_rate (if available in logger)

    Saves:
      - <out_dir>/<run_id>_summary.json
      - <out_dir>/<run_id>_steps.jsonl   (one line per log interval)
    """

    def __init__(
        self,
        env_id: str,
        out_dir: str = "runs",
        run_tag: str | None = None,
        log_every: int = 2048,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.env_id = env_id
        self.out_dir = out_dir
        self.run_tag = run_tag
        self.log_every = int(log_every)

        self.run_id = None
        self.summary_path = None
        self.steps_path = None

        self.meta = {}
        self.t0 = None
        self.log_count = 0

    def _safe_get(self, obj, name, default=None):
        return getattr(obj, name, default)

    def _get_logger_dict(self) -> dict:
        """
        Best-effort: SB3 stores last dumped values in model.logger.
        """
        logger = self.model.logger
        d = getattr(logger, "name_to_value", {}) or {}
        out = {}
        for k, v in d.items():
            try:
                out[k] = float(v)
            except Exception:
                out[k] = v
        return out

    def _ep_means_from_buffer(self):
        """
        Reliable episode stats: requires Monitor/VecMonitor wrapper.
        Returns (ep_rew_mean, ep_len_mean) or (None, None) if no episodes finished yet.
        """
        buf = getattr(self.model, "ep_info_buffer", None)
        if not buf or len(buf) == 0:
            return None, None

        rews = [ep.get("r") for ep in buf if isinstance(ep, dict) and ("r" in ep)]
        lens = [ep.get("l") for ep in buf if isinstance(ep, dict) and ("l" in ep)]

        ep_rew_mean = float(np.mean(rews)) if len(rews) else None
        ep_len_mean = float(np.mean(lens)) if len(lens) else None
        return ep_rew_mean, ep_len_mean

    def _write_json(self, path, data):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _append_jsonl(self, path, data):
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data) + "\n")

    def _on_training_start(self) -> None:
        os.makedirs(self.out_dir, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{ts}_{self.run_tag}" if self.run_tag else ts

        self.summary_path = os.path.join(self.out_dir, f"{self.run_id}_summary.json")
        self.steps_path = os.path.join(self.out_dir, f"{self.run_id}_steps.jsonl")
        self.t0 = time.time()
        self.log_count = 0

        alg = self.model.__class__.__name__
        policy = self.model.policy.__class__.__name__
        device = str(self.model.device)

        # DQN-relevant hyperparams
        hp = {
        "buffer_size": self._safe_get(self.model, "buffer_size"),
        "learning_starts": self._safe_get(self.model, "learning_starts"),
        "batch_size": self._safe_get(self.model, "batch_size"),
        "gamma": self._safe_get(self.model, "gamma"),

        # FIX: stringify train_freq (it’s a TrainFreq object / enum inside)
        "train_freq": str(self._safe_get(self.model, "train_freq")),

        "gradient_steps": self._safe_get(self.model, "gradient_steps"),
        "target_update_interval": self._safe_get(self.model, "target_update_interval"),
        "exploration_fraction": self._safe_get(self.model, "exploration_fraction"),
        "exploration_initial_eps": self._safe_get(self.model, "exploration_initial_eps"),
        "exploration_final_eps": self._safe_get(self.model, "exploration_final_eps"),
        "learning_rate": float(self.model.learning_rate) if hasattr(self.model, "learning_rate") else None,
        "seed": self._safe_get(self.model, "seed"),
        "log_every": self.log_every,}


        self.meta = {
            "run_id": self.run_id,
            "env_id": self.env_id,
            "algorithm": alg,
            "policy": policy,
            "device": device,
            "observation_space": str(self.training_env.observation_space),
            "action_space": str(self.training_env.action_space),
            "hyperparams": hp,
        }

        # Write initial summary + clear steps file
        self._write_json(self.summary_path, {**self.meta, "final": None})
        open(self.steps_path, "w", encoding="utf-8").close()

        print(f"[RUN] {self.meta['run_id']}")
        print(f" env: {self.meta['env_id']}")
        print(f" alg: {self.meta['algorithm']} | policy: {self.meta['policy']} | device: {self.meta['device']}")
        print(f" hp : buffer={hp['buffer_size']} batch={hp['batch_size']} gamma={hp['gamma']} "
              f"train_freq={hp['train_freq']} target_upd={hp['target_update_interval']} "
              f"eps=[{hp['exploration_initial_eps']}→{hp['exploration_final_eps']}] "
              f"log_every={hp['log_every']}")

    def _log_step_row(self):
        d = self._get_logger_dict()
        total_timesteps = int(self.num_timesteps)

        fps = d.get("time/fps", None)
        if fps is None and self.t0:
            elapsed = time.time() - self.t0
            fps = (total_timesteps / elapsed) if elapsed > 0 else None

        ep_rew_mean, ep_len_mean = self._ep_means_from_buffer()

        self.log_count += 1
        row = {
            "time/total_timesteps": total_timesteps,
            "time/fps": fps,
            "rollout/ep_rew_mean": ep_rew_mean,
            "rollout/ep_len_mean": ep_len_mean,
            "train/loss": d.get("train/loss", None),
            "train/exploration_rate": d.get("train/exploration_rate", None),
            "log_idx": int(self.log_count),
        }
        self._append_jsonl(self.steps_path, row)

        rew = row["rollout/ep_rew_mean"]
        ln = row["rollout/ep_len_mean"]
        loss = row["train/loss"]
        eps = row["train/exploration_rate"]

        def fmt(x, nd=3):
            return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "NA"

        print(
            f"[{row['log_idx']:03d}] steps={total_timesteps} "
            f"rew={fmt(rew)} len={fmt(ln, nd=1)} loss={fmt(loss)} eps={fmt(eps)}"
        )

    def _on_step(self) -> bool:
        # Log every N environment steps
        if self.log_every > 0 and (self.num_timesteps % self.log_every == 0):
            self._log_step_row()
        return True

    def _on_training_end(self) -> None:
        d = self._get_logger_dict()
        total_timesteps = int(self.num_timesteps)
        ep_rew_mean, ep_len_mean = self._ep_means_from_buffer()

        final = {
            "time/total_timesteps": total_timesteps,
            "rollout/ep_rew_mean": ep_rew_mean,
            "rollout/ep_len_mean": ep_len_mean,
            "train/loss": d.get("train/loss", None),
            "train/exploration_rate": d.get("train/exploration_rate", None),
        }
        self._write_json(self.summary_path, {**self.meta, "final": final})


def make_report_logger_dqn(env_id: str, out_dir: str = "runs", run_tag: str | None = None, log_every: int = 2048):
    """Create the DQN callback. Pass into model.learn(callback=...)."""
    return ReportLoggerCallbackDQN(env_id=env_id, out_dir=out_dir, run_tag=run_tag, log_every=log_every, verbose=0)
