"""
DD-DQN agent for predictive AMM RL (minimal, production‑ready)
--------------------------------------------------------------
This module implements a compact Dueling Double DQN agent that works with our
`ml/rl_env.py` environment. It supports:

• Dueling Q‑network (value + advantage streams)
• Double‑DQN target selection with a target network
• Epsilon‑greedy exploration with linear or exponential decay
• Prioritized replay (optional; defaults to uniform replay)
• Gradient clipping, soft target updates, checkpointing

Quick start
----------
from ml.rl_env import RLEnv, RLEnvConfig
from ml.dqn import DDQNConfig, DDQNAgent, train

agent = DDQNAgent(state_dim=5, n_actions=2, cfg=DDQNConfig())
history = train(env, agent, steps=100_000)

Notes
-----
• The environment exposes discrete actions {0,1} by construction.
• Observations are 1‑D state vectors. If you pass stacked frames, set state_dim accordingly.
• The loss is standard TD‑error; the *task* reward shaping lives in env (Section‑based design).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional, List
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------- Replay Buffer ---------------------------------------

class ReplayBuffer:
    def __init__(self, capacity: int, prioritized: bool = False, alpha: float = 0.6, eps: float = 1e-6):
        self.capacity = int(capacity)
        self.pos = 0
        self.full = False
        self.s = np.zeros((capacity, 0), dtype=np.float32)  # lazy shape set on first push
        self.a = np.zeros((capacity,), dtype=np.int64)
        self.r = np.zeros((capacity,), dtype=np.float32)
        self.sp = np.zeros((capacity, 0), dtype=np.float32)
        self.d = np.zeros((capacity,), dtype=np.bool_)
        self.prioritized = prioritized
        self.alpha = alpha
        self.eps = eps
        self.prior = np.zeros((capacity,), dtype=np.float32)

    def __len__(self):
        return self.capacity if self.full else self.pos

    def _ensure_shape(self, s: np.ndarray):
        if self.s.shape[1] == 0:
            dim = s.shape[-1]
            self.s = np.zeros((self.capacity, dim), dtype=np.float32)
            self.sp = np.zeros((self.capacity, dim), dtype=np.float32)

    def push(self, s: np.ndarray, a: int, r: float, sp: np.ndarray, done: bool, td_error: Optional[float] = None):
        s = np.asarray(s, dtype=np.float32)
        sp = np.asarray(sp, dtype=np.float32)
        self._ensure_shape(s)
        i = self.pos
        self.s[i] = s
        self.a[i] = a
        self.r[i] = r
        self.sp[i] = sp
        self.d[i] = done
        if self.prioritized:
            p = (abs(td_error) + self.eps) if td_error is not None else 1.0
            self.prior[i] = p ** self.alpha
        self.pos = (self.pos + 1) % self.capacity
        self.full = self.full or self.pos == 0

    def sample(self, batch_size: int, beta: float = 0.4):
        n = len(self)
        idx = None
        weights = None
        if not self.prioritized:
            idx = np.random.randint(0, n, size=batch_size)
            weights = np.ones_like(idx, dtype=np.float32)
        else:
            p = self.prior[:n]
            p = p / (p.sum() + 1e-12)
            idx = np.random.choice(n, size=batch_size, p=p)
            # importance sampling weights
            w = (n * p[idx]) ** (-beta)
            w = w / (w.max() + 1e-12)
            weights = w.astype(np.float32)
        batch = (
            torch.as_tensor(self.s[idx], dtype=torch.float32),
            torch.as_tensor(self.a[idx], dtype=torch.int64),
            torch.as_tensor(self.r[idx], dtype=torch.float32),
            torch.as_tensor(self.sp[idx], dtype=torch.float32),
            torch.as_tensor(self.d[idx], dtype=torch.bool),
            torch.as_tensor(weights, dtype=torch.float32),
            torch.as_tensor(idx, dtype=torch.int64),
        )
        return batch

    def update_priorities(self, idx: torch.Tensor, td_err: torch.Tensor):
        if not self.prioritized:
            return
        td = td_err.detach().abs().cpu().numpy()
        idx_np = idx.cpu().numpy()
        self.prior[idx_np] = (td + self.eps) ** self.alpha

# -------------------------- Q‑Network -------------------------------------------

class DuelingQNet(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.val = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.adv = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_actions))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        v = self.val(h)
        a = self.adv(h)
        # Combine streams: Q = V + (A - mean(A))
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q

# -------------------------- Config ----------------------------------------------

@dataclass
class DDQNConfig:
    gamma: float = 0.98
    lr: float = 1e-3
    weight_decay: float = 0.0
    hidden: int = 128
    batch_size: int = 256
    buffer_size: int = 200_000
    min_buffer: int = 10_000
    train_freq: int = 1
    target_sync: int = 1000          # hard update interval (steps) if tau==0
    tau: float = 0.0                 # if >0, soft update factor
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 100_000   # linear decay length
    per: bool = False                # prioritized replay
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_beta_steps: int = 200_000
    grad_clip: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0

# -------------------------- Agent -----------------------------------------------

class DDQNAgent:
    def __init__(self, state_dim: int, n_actions: int, cfg: DDQNConfig = DDQNConfig()):
        self.cfg = cfg
        self.device = cfg.device
        self.n_actions = n_actions
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)

        self.q = DuelingQNet(state_dim, n_actions, hidden=cfg.hidden).to(self.device)
        self.targ = DuelingQNet(state_dim, n_actions, hidden=cfg.hidden).to(self.device)
        self.targ.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        self.buf = ReplayBuffer(cfg.buffer_size, prioritized=cfg.per)
        self.step_count = 0

    # ----- exploration schedule -----
    def epsilon(self) -> float:
        t = self.step_count
        eps = self.cfg.eps_end + max(0, self.cfg.eps_decay_steps - t) * (self.cfg.eps_start - self.cfg.eps_end) / max(1, self.cfg.eps_decay_steps)
        return float(eps)

    def beta(self) -> float:
        if not self.cfg.per:
            return 1.0
        t = self.step_count
        b = self.cfg.per_beta_start + min(1.0, t / max(1, self.cfg.per_beta_steps)) * (self.cfg.per_beta_end - self.cfg.per_beta_start)
        return float(b)

    # ----- action selection -----
    def act(self, s: np.ndarray) -> int:
        self.step_count += 1
        if np.random.rand() < self.epsilon():
            return int(np.random.randint(self.n_actions))
        with torch.no_grad():
            q = self.q(torch.as_tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0))
            return int(torch.argmax(q, dim=1).item())

    # ----- learning -----
    def _soft_update(self, tau: float):
        if tau <= 0:
            self.targ.load_state_dict(self.q.state_dict())
            return
        with torch.no_grad():
            for p, tp in zip(self.q.parameters(), self.targ.parameters()):
                tp.data.mul_(1 - tau).add_(tau * p.data)

    def learn(self):
        if len(self.buf) < self.cfg.min_buffer:
            return None
        if self.step_count % self.cfg.train_freq != 0:
            return None
        s, a, r, sp, d, w, idx = self.buf.sample(self.cfg.batch_size, beta=self.beta())
        s = s.to(self.device); a = a.to(self.device); r = r.to(self.device)
        sp = sp.to(self.device); d = d.to(self.device); w = w.to(self.device); idx = idx.to(self.device)

        # Q(s,a)
        q_all = self.q(s)
        q_sa = q_all.gather(1, a.view(-1, 1)).squeeze(1)

        # Double DQN target: a* from online, evaluate via target net
        with torch.no_grad():
            a_star = torch.argmax(self.q(sp), dim=1)
            q_sp = self.targ(sp).gather(1, a_star.view(-1, 1)).squeeze(1)
            y = r + self.cfg.gamma * (1.0 - d.float()) * q_sp

        td_err = y - q_sa
        loss = (w * td_err.pow(2)).mean()
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.grad_clip)
        self.opt.step()

        # update PER priorities
        self.buf.update_priorities(idx, td_err)

        # target net updates
        if self.cfg.tau > 0:
            self._soft_update(self.cfg.tau)
        elif self.step_count % self.cfg.target_sync == 0:
            self._soft_update(0.0)

        return float(loss.item()), float(td_err.abs().mean().item())

    # ----- checkpoint -----
    def save(self, path: str, meta: Optional[Dict] = None):
        torch.save({
            "q": self.q.state_dict(),
            "targ": self.targ.state_dict(),
            "opt": self.opt.state_dict(),
            "cfg": asdict(self.cfg),
            "meta": meta or {},
        }, path)

    @staticmethod
    def load(path: str, state_dim: int, n_actions: int) -> "DDQNAgent":
        obj = torch.load(path, map_location="cpu")
        cfg = DDQNConfig(**obj["cfg"])  # type: ignore
        agent = DDQNAgent(state_dim, n_actions, cfg)
        agent.q.load_state_dict(obj["q"])  # type: ignore
        agent.targ.load_state_dict(obj["targ"])  # type: ignore
        agent.opt.load_state_dict(obj["opt"])  # type: ignore
        return agent

# -------------------------- Training Loop ---------------------------------------

def train(env, agent: DDQNAgent, steps: int = 200_000, warm_start_random: int = 5_000) -> Dict[str, List[float]]:
    """Generic step‑based trainer. Returns history dict.

    • Performs environment interaction, fills replay, and optimizes online.
    • Uses epsilon‑greedy through agent.epsilon schedule.
    """
    s = env.reset()
    hist = {"loss": [], "td": [], "reward": []}

    for t in range(1, steps + 1):
        # explore heavily during warm start
        if t < warm_start_random:
            a = np.random.randint(agent.n_actions)
        else:
            a = agent.act(s)
        sp, r, done, info = env.step(a)

        # approximate TD error for PER bootstrap (0 if unknown)
        td_boot = None
        if agent.buf.prioritized and len(agent.buf) > agent.cfg.min_buffer:
            with torch.no_grad():
                qs = agent.q(torch.as_tensor(s, dtype=torch.float32, device=agent.device).unsqueeze(0))[0]
                qsp = agent.targ(torch.as_tensor(sp, dtype=torch.float32, device=agent.device).unsqueeze(0))[0]
                a_star = torch.argmax(agent.q(torch.as_tensor(sp, dtype=torch.float32, device=agent.device).unsqueeze(0)), dim=1)[0]
                y = r + agent.cfg.gamma * (0.0 if done else qsp[a_star].item())
                td_boot = float(y - qs[a].item())

        agent.buf.push(s, a, r, sp, done, td_error=td_boot)
        learn_out = agent.learn()
        if learn_out is not None:
            loss, td = learn_out
            hist["loss"].append(loss)
            hist["td"].append(td)
        hist["reward"].append(r)

        s = sp
        if done:
            s = env.reset()

    return hist

# -------------------------- Demo -------------------------------------------------
