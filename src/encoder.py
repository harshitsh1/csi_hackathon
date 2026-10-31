"""1-D CNN encoder over time (subcarriers = input channels) + SimCLR-style contrastive pretraining.
No activity or room labels are used anywhere in this file."""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CSIEncoder(nn.Module):
    def __init__(self, n_sub=256, feat_dim=512, proj_dim=128):
        super().__init__()
        c = feat_dim // 2
        self.conv = nn.Sequential(
            nn.Conv1d(n_sub, 192, 5, stride=2, padding=2), nn.BatchNorm1d(192), nn.GELU(),
            nn.Conv1d(192, c, 5, stride=2, padding=2), nn.BatchNorm1d(c), nn.GELU(),
            nn.Conv1d(c, c, 3, padding=1), nn.BatchNorm1d(c), nn.GELU())
        self.proj = nn.Sequential(nn.Linear(feat_dim, 256), nn.ReLU(), nn.Linear(256, proj_dim))

    def features(self, x):                      # x: (B, T, S)
        h = self.conv(x.transpose(1, 2))
        return torch.cat([h.mean(2), h.std(2)], 1)   # (B, feat_dim)

    def forward(self, x):
        f = self.features(x)
        return f, F.normalize(self.proj(f), dim=1)


def augment(x, c):
    """x: (B,T,S) normalized windows. Label-free stochastic views."""
    B, T, S = x.shape
    L = np.random.randint(int(c["crop_min_frac"] * T), T + 1)          # random temporal crop...
    st = torch.randint(0, T - L + 1, (B,), device=x.device)
    idx = st[:, None] + torch.arange(L, device=x.device)[None]
    x = x[torch.arange(B, device=x.device)[:, None], idx]
    x = F.interpolate(x.transpose(1, 2), size=T, mode="linear", align_corners=False).transpose(1, 2)  # ...resized
    x = x + c["gain_std"] * torch.randn(B, 1, 1, device=x.device) \
          + c["subcarrier_shift_std"] * torch.randn(B, 1, S, device=x.device) \
          + c["noise_std"] * torch.randn_like(x)
    x = x * (torch.rand(B, 1, S, device=x.device) > c["subcarrier_dropout"]).float()
    tm = np.random.randint(0, c["max_time_mask"] + 1)
    if tm:
        s0 = np.random.randint(0, T - tm + 1)
        x = x.clone(); x[:, s0:s0 + tm] = 0
    return x


def nt_xent(z1, z2, t):
    n = len(z1)
    z = torch.cat([z1, z2])
    sim = (z @ z.T) / t
    sim.fill_diagonal_(float("-inf"))
    tgt = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, tgt)


def train_ssl(Xn, cfg, device="cpu", seed=42, log=print):
    c = cfg["ssl"]
    torch.manual_seed(seed); np.random.seed(seed)
    model = CSIEncoder(Xn.shape[-1], c["feat_dim"], c["proj_dim"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
    n, bs = len(Xn), min(c["batch_size"], len(Xn) // 2)
    steps = c["epochs"] * max(1, n // bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / steps)))
    hist = []
    for ep in range(c["epochs"]):
        model.train(); perm = np.random.permutation(n); tot = 0.0; nb = 0
        for i in range(0, n - bs + 1, bs):
            xb = torch.from_numpy(Xn[np.sort(perm[i:i + bs])]).to(device)
            _, z1 = model(augment(xb, c)); _, z2 = model(augment(xb, c))
            loss = nt_xent(z1, z2, c["temperature"])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item(); nb += 1
        hist.append(tot / max(nb, 1))
        log(f"[ssl] epoch {ep+1}/{c['epochs']} loss {hist[-1]:.4f}")
    return model, hist


def random_encoder(cfg, n_sub=256, seed=0):
    torch.manual_seed(seed)
    return CSIEncoder(n_sub, cfg["ssl"]["feat_dim"], cfg["ssl"]["proj_dim"])


@torch.no_grad()
def embed(model, Xn, device="cpu", bs=256):
    model.eval().to(device)
    out = [model.features(torch.from_numpy(Xn[i:i + bs]).to(device)).cpu().numpy()
           for i in range(0, len(Xn), bs)]
    return np.concatenate(out).astype(np.float32)
