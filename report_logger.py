import os, json, time
from datetime import datetime
from stable_baselines3.common.callbacks import BaseCallback


# THIS IS AN AI GENERATED CUSTOM LOGGER!!!!!!!!!



class ReportLoggerCallback(BaseCallback):
    """
    Logs only report-useful info:
    - Run meta (algorithm, policy, env, seed, device, key hyperparams)
    - Per-rollout summary (ep_rew_mean, ep_len_mean + core PPO train metrics if present)
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
        self.rollout_idx = 0
        self.rollouts_path = None
        self.meta = {}
        self.rollout_count = 0
        self.t0 = None


    def _on_step(self) -> bool:
        return True


    def _safe_get(self, obj, name, default=None):
        return getattr(obj, name, default)

    def _get_logger_dict(self):
        # SB3 stores the last dumped values here (names vary by version)
        logger = self.model.logger
        d = getattr(logger, "name_to_value", {}) or {}
        # Convert numpy scalars to python
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

        # hyperparams (only the ones that matter in a report)
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

        # write initial summary + clear rollouts file
        self._write_json(self.summary_path, {**self.meta, "final": None})
        open(self.rollouts_path, "w").close()

        # clean console header
        print(f"[RUN] {self.meta['run_id']}")
        print(f" env: {self.meta['env_id']}")
        print(f" alg: {self.meta['algorithm']} | policy: {self.meta['policy']} | device: {self.meta['device']}")
        print(f" hp : n_steps={hp['n_steps']} batch={hp['batch_size']} epochs={hp['n_epochs']} lr={hp['learning_rate']} gamma={hp['gamma']} gae={hp['gae_lambda']} clip={hp['clip_range']}")

def _on_rollout_end(self) -> None:
    d = self._get_logger_dict()

    keep = [
        "time/total_timesteps",
        "time/fps",
        "rollout/ep_rew_mean",
        "rollout/ep_len_mean",
        "train/approx_kl",
        "train/clip_fraction",
        "train/entropy_loss",
        "train/explained_variance",
        "train/policy_gradient_loss",
        "train/value_loss",
        "train/loss",
    ]

    # 1) build row FIRST
    row = {k: d.get(k, None) for k in keep}

    # 2) set rollout index ONCE (use a single counter)
    row["rollout_idx"] = int(self.rollout_count)

    # 3) increment
    self.rollout_count += 1

    self._append_jsonl(self.rollouts_path, row)

    # ---- clean console line (safe if None) ----
    steps = int(row.get("time/total_timesteps") or 0)

    rew = row.get("rollout/ep_rew_mean")
    rew_str = f"{float(rew):.3f}" if rew is not None else "NA"

    ep_len = row.get("rollout/ep_len_mean")
    len_str = f"{float(ep_len):.1f}" if ep_len is not None else "NA"

    kl = row.get("train/approx_kl")
    kl_str = f"{float(kl):.4f}" if kl is not None else "NA"

    ev = row.get("train/explained_variance")
    ev_str = f"{float(ev):.3f}" if ev is not None else "NA"

    print(f"[{row['rollout_idx']:03d}] steps={steps} rew={rew_str} len={len_str} kl={kl_str} ev={ev_str}")



def make_report_logger(env_id: str, out_dir: str = "runs", run_tag: str = None):
    """Create the callback. Pass it into model.learn(callback=...)."""
    return ReportLoggerCallback(env_id=env_id, out_dir=out_dir, run_tag=run_tag, verbose=0)
