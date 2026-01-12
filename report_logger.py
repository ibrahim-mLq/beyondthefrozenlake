import os, json, time
from datetime import datetime
from stable_baselines3.common.callbacks import BaseCallback


class ReportLoggerCallback(BaseCallback):
    """
    Logs only report-useful info:
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
        logger = self.model.logger
        d = getattr(logger, "name_to_value", {}) or {}
        out = {}
        for k, v in d.items():
            try:
                out[k] = float(v)
            except Exception:
                out[k] = v
        return out

    def _write_json(self, path, data):
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _append_jsonl(self, path, data):
        with open(path, "a") as f:
            f.write(json.dumps(data) + "\n")

    def _on_training_start(self) -> None:
        os.makedirs(self.out_dir, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{ts}_{self.run_tag}" if self.run_tag else ts
        self.summary_path = os.path.join(self.out_dir, f"{self.run_id}_summary.json")
        self.rollouts_path = os.path.join(self.out_dir, f"{self.run_id}_rollouts.jsonl")
        self.t0 = time.time()

        alg = self.model.__class__.__name__
        policy = self.model.policy.__class__.__name__
        device = str(self.model.device)

        hp = {
            "n_steps": self._safe_get(self.model, "n_steps"),
            "batch_size": self._safe_get(self.model, "batch_size"),
            "n_epochs": self._safe_get(self.model, "n_epochs"),
            "gamma": self._safe_get(self.model, "gamma"),
            "gae_lambda": self._safe_get(self.model, "gae_lambda"),
            "learning_rate": float(self.model.learning_rate) if hasattr(self.model, "learning_rate") else None,
            "clip_range": float(self.model.clip_range(1.0)) if callable(getattr(self.model, "clip_range", None)) else self._safe_get(self.model, "clip_range"),
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

        self._write_json(self.summary_path, {**self.meta, "final": None})
        open(self.rollouts_path, "w").close()

        print(f"[RUN] {self.meta['run_id']}")
        print(f" env: {self.meta['env_id']}")
        print(f" alg: {self.meta['algorithm']} | policy: {self.meta['policy']} | device: {self.meta['device']}")
        print(f" hp : n_steps={hp['n_steps']} batch={hp['batch_size']} epochs={hp['n_epochs']} lr={hp['learning_rate']} gamma={hp['gamma']} gae={hp['gae_lambda']} clip={hp['clip_range']}")

    def _on_rollout_end(self) -> None:
        # --- core rollout step info ---
        steps = int(getattr(self.model, "num_timesteps", 0))

        # --- episode stats (requires Monitor/VecMonitor) ---
        rew_mean = None
        len_mean = None
        try:
            buf = getattr(self.model, "ep_info_buffer", None)
            if buf and len(buf) > 0:
                ep_rews = [ep.get("r") for ep in buf if ep is not None]
                ep_lens = [ep.get("l") for ep in buf if ep is not None]
                ep_rews = [r for r in ep_rews if r is not None]
                ep_lens = [l for l in ep_lens if l is not None]
                if ep_rews:
                    rew_mean = sum(ep_rews) / len(ep_rews)
                if ep_lens:
                    len_mean = sum(ep_lens) / len(ep_lens)
        except Exception:
            rew_mean, len_mean = None, None

        # --- training stats (may be empty depending on timing) ---
        try:
            d = self._get_logger_dict()
        except Exception:
            d = {}

        row = {
            "rollout_idx": self.rollout_count + 1,
            "time/total_timesteps": steps,
            "rollout/ep_rew_mean": rew_mean,
            "rollout/ep_len_mean": len_mean,
            "train/approx_kl": d.get("train/approx_kl", None),
            "train/clip_fraction": d.get("train/clip_fraction", None),
            "train/entropy_loss": d.get("train/entropy_loss", None),
            "train/explained_variance": d.get("train/explained_variance", None),
            "train/policy_gradient_loss": d.get("train/policy_gradient_loss", None),
            "train/value_loss": d.get("train/value_loss", None),
            "train/loss": d.get("train/loss", None),
            "time/fps": d.get("time/fps", None),
        }

        self.rollout_count += 1
        self._append_jsonl(self.rollouts_path, row)

        def fmt(x, nd=3):
            return "NA" if x is None else f"{x:.{nd}f}"

        print(
            f"[{row['rollout_idx']:03d}] "
            f"steps={row['time/total_timesteps']} "
            f"rew={fmt(row['rollout/ep_rew_mean'], 3)} "
            f"len={fmt(row['rollout/ep_len_mean'], 1)} "
            f"kl={fmt(row['train/approx_kl'], 4)} "
            f"ev={fmt(row['train/explained_variance'], 3)}"
        )


def make_report_logger(env_id: str, out_dir: str = "runs", run_tag: str = None):
    return ReportLoggerCallback(env_id=env_id, out_dir=out_dir, run_tag=run_tag, verbose=0)
