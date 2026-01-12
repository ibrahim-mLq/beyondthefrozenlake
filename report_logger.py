# report_logger.py
import os
import json
import time
from datetime import datetime

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class ReportLoggerCallback(BaseCallback):
    """
    Logs only report-useful info:
    - Run meta (algorithm, policy, env, seed, device, key hyperparams)
    - Per-rollout summary:
        * episode means from ep_info_buffer (reliable): ep_rew_mean, ep_len_mean
        * PPO train metrics if present in the logger dict
    Saves:
      - <out_dir>/<run_id>_summary.json
      - <out_dir>/<run_id>_rollouts.jsonl   (one line per rollout)
    """

    def __init__(self, env_id: str, out_dir: str = "runs", run_tag: str = None, verbose: int = 0):
        super().__init__(verbose)
        self.env_id = env_id
        self.out_dir = out_dir
        self.run_tag = run_tag

        self.run_id = None
        self.summary_path = None
        self.rollouts_path = None

        self.meta = {}
        self.rollout_count = 0
        self.t0 = None

    def _on_step(self) -> bool:
        return True

    def _safe_get(self, obj, name, default=None):
        return getattr(obj, name, default)

    def _get_logger_dict(self):
        """
        Best-effort: SB3 stores last dumped values in model.logger.
        NOTE: rollout/ep_* are often NOT available at _on_rollout_end timing.
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
        self.rollouts_path = os.path.join(self.out_dir, f"{self.run_id}_rollouts.jsonl")
        self.t0 = time.time()
        self.rollout_count = 0

        alg = self.model.__class__.__name__
        policy = self.model.policy.__class__.__name__
        device = str(self.model.device)

        # Hyperparams (report-useful)
        hp = {
            "n_steps": self._safe_get(self.model, "n_steps"),
            "batch_size": self._safe_get(self.model, "batch_size"),
            "n_epochs": self._safe_get(self.model, "n_epochs"),
            "gamma": self._safe_get(self.model, "gamma"),
            "gae_lambda": self._safe_get(self.model, "gae_lambda"),
            "learning_rate": float(self.model.learning_rate) if hasattr(self.model, "learning_rate") else None,
            "clip_range": float(self.model.clip_range(1.0))
            if callable(getattr(self.model, "clip_range", None))
            else self._safe_get(self.model, "clip_range"),
            "ent_coef": self._safe_get(self.model, "ent_coef"),
            "vf_coef": self._safe_get(self.model, "vf_coef"),
            "max_grad_norm": self._safe_get(self.model, "max_grad_norm"),
            "seed": self._safe_get(self.model, "seed"),
        }

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

        # Write initial summary + clear rollouts file
        self._write_json(self.summary_path, {**self.meta, "final": None})
        open(self.rollouts_path, "w", encoding="utf-8").close()

        # Console header
        print(f"[RUN] {self.meta['run_id']}")
        print(f" env: {self.meta['env_id']}")
        print(f" alg: {self.meta['algorithm']} | policy: {self.meta['policy']} | device: {self.meta['device']}")
        print(
            f" hp : n_steps={hp['n_steps']} batch={hp['batch_size']} epochs={hp['n_epochs']} "
            f"lr={hp['learning_rate']} gamma={hp['gamma']} gae={hp['gae_lambda']} clip={hp['clip_range']}"
        )

    def _on_rollout_end(self) -> None:
        d = self._get_logger_dict()
        total_timesteps = int(self.num_timesteps)

        # fps: read if available, else compute
        fps = d.get("time/fps", None)
        if fps is None and self.t0:
            elapsed = time.time() - self.t0
            fps = (total_timesteps / elapsed) if elapsed > 0 else None

        # Reliable episode stats (requires Monitor/VecMonitor)
        ep_rew_mean, ep_len_mean = self._ep_means_from_buffer()

        # 1-based rollout index (nicer for reports)
        self.rollout_count += 1

        row = {
            "time/total_timesteps": total_timesteps,
            "time/fps": fps,
            "rollout/ep_rew_mean": ep_rew_mean,
            "rollout/ep_len_mean": ep_len_mean,
            # Train metrics (may be None depending on SB3 dump timing)
            "train/approx_kl": d.get("train/approx_kl", None),
            "train/clip_fraction": d.get("train/clip_fraction", None),
            "train/entropy_loss": d.get("train/entropy_loss", None),
            "train/explained_variance": d.get("train/explained_variance", None),
            "train/policy_gradient_loss": d.get("train/policy_gradient_loss", None),
            "train/value_loss": d.get("train/value_loss", None),
            "train/loss": d.get("train/loss", None),
            "rollout_idx": int(self.rollout_count),
        }

        self._append_jsonl(self.rollouts_path, row)

        # Console output
        rew = row["rollout/ep_rew_mean"]
        ln = row["rollout/ep_len_mean"]
        kl = row["train/approx_kl"]
        ev = row["train/explained_variance"]

        rew_str = f"{rew:.3f}" if isinstance(rew, (int, float)) else "NA"
        len_str = f"{ln:.1f}" if isinstance(ln, (int, float)) else "NA"
        kl_str = f"{kl:.4f}" if isinstance(kl, (int, float)) else "NA"
        ev_str = f"{ev:.3f}" if isinstance(ev, (int, float)) else "NA"

        print(
            f"[{row['rollout_idx']:03d}] steps={total_timesteps} "
            f"rew={rew_str} len={len_str} kl={kl_str} ev={ev_str}"
        )

    def _on_training_end(self) -> None:
        """
        Write a final summary snapshot (best-effort).
        """
        d = self._get_logger_dict()
        total_timesteps = int(self.num_timesteps)
        ep_rew_mean, ep_len_mean = self._ep_means_from_buffer()

        final = {
            "time/total_timesteps": total_timesteps,
            "rollout/ep_rew_mean": ep_rew_mean,
            "rollout/ep_len_mean": ep_len_mean,
            "train/approx_kl": d.get("train/approx_kl", None),
            "train/clip_fraction": d.get("train/clip_fraction", None),
            "train/entropy_loss": d.get("train/entropy_loss", None),
            "train/explained_variance": d.get("train/explained_variance", None),
            "train/value_loss": d.get("train/value_loss", None),
            "train/loss": d.get("train/loss", None),
        }
        self._write_json(self.summary_path, {**self.meta, "final": final})


def make_report_logger(env_id: str, out_dir: str = "runs", run_tag: str = None):
    """Create the callback. Pass it into model.learn(callback=...)."""
    return ReportLoggerCallback(env_id=env_id, out_dir=out_dir, run_tag=run_tag, verbose=0)
