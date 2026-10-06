"""train_exp03.py - EXPERIMENT 3: masked encoder + semi-hard, impostor-aware mining.

What changes vs baseline (Exp02 kept architecture/loss fixed; this changes
HOW negatives are chosen and how sequences are pooled):
  * model_v2: masked mean pooling (no padding dilution), LayerNorm fusion,
    modality gate. Longer windows (key 96, mouse 256).
  * batch semi-hard mining on TRAIN users only: embed a batch of P users x
    G genuine trials (+ their targeted false_data trials), then per anchor
    pick the hardest same-user positive and the hardest SEMI-HARD negative:
      d(a,p) < d(a,n) < d(a,p) + margin, preferring same-user false_data
      (targeted impostor) candidates first, cross-user genuine second.
    Fallbacks (hardest available negative) keep every anchor informative.
    Rationale: random sampling wastes most updates on zero-loss easy
    triplets; semi-hard keeps gradients informative without the collapse
    risk of pure hardest-mining.
  * impostor-aware quota: at least --target_frac of mined negatives must be
    same-user false_data (targeted attack model); the rest are cross-user
    (zero-effort model). Both threat models stay represented.
  * best-validation-AUC checkpointing, cosine LR, grad clip (as in Exp02).
  * separate cache (kmt_cache_v3.pkl) with config guard: v2 preprocessing
    differs (longer windows), so it must NEVER reuse the v2/v1 cache
    silently. Separate output: kmt_multimodal_siamese_exp03.pth.

Usage (Colab T4):
  !python train_exp03.py --data /content/raw_kmt_dataset \\
      --epochs 35 --steps 32 --P 16 --G 4 --K 5
"""
import argparse
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dataset_v2 import load_kmt_dataset_v2, KEY_MAX, MOUSE_MAX
from model_v2 import MultimodalBiometricEncoderV2
from train import compute_eer

CACHE_V3 = Path("kmt_cache_v3.pkl")
OUT_PTH = Path("kmt_multimodal_siamese_exp03.pth")
BEST_PTH = Path("kmt_multimodal_siamese_exp03_best.pth")


def build_index_v2(samples):
    gen, imp = {}, {}
    for s in samples:
        gen.setdefault(s.user_idx, [])
        imp.setdefault(s.user_idx, [])
        (gen if s.is_genuine else imp)[s.user_idx].append(s)
    users = [u for u in gen if len(gen[u]) >= 2]
    return gen, imp, users


def embed_pool(model, keys, mouse, key_len, mouse_len):
    return model(keys, mouse, key_len, mouse_len)


@torch.no_grad()
def evaluate_few_shot_v2(model, samples, K=5, device="cpu"):
    model.eval()
    users = {}
    for s in samples:
        users.setdefault(s.user_idx, {"gen": [], "imp": []})[
            "gen" if s.is_genuine else "imp"].append(s)
    y, sc = [], []

    def emb(L):
        return model(torch.stack([s.key_matrix for s in L]).to(device),
                     torch.stack([s.mouse_matrix for s in L]).to(device),
                     torch.tensor([s.key_len for s in L],
                                  dtype=torch.long, device=device),
                     torch.tensor([s.mouse_len for s in L],
                                  dtype=torch.long, device=device))

    for d in users.values():
        if len(d["gen"]) < K + 1 or not d["imp"]:
            continue
        c = F.normalize(emb(d["gen"][:K]).mean(0, keepdim=True), dim=-1)
        sg = (emb(d["gen"][K:]) @ c.t()).squeeze(-1).cpu().numpy()
        si = (emb(d["imp"]) @ c.t()).squeeze(-1).cpu().numpy()
        sc += list(sg) + list(si)
        y += [1] * len(sg) + [0] * len(si)
    return np.array(y), np.array(sc)


def mine_batch(model, index, P: int, G: int, margin: float,
               target_frac: float, device):
    """Build one mined triplet batch from TRAIN users only.

    Returns stacked (ak, am, akl, aml, pk, ..., nk, ...) tensors + lengths.
    Mining is done under no_grad; the returned triplets get a fresh
    forward pass with gradients.
    """
    gen, imp, users = index
    chosen = random.sample(users, min(P, len(users)))
    # Gather: G genuine per user + up to G targeted impostor trials
    items = []  # (sample, user, role) role: 0 genuine, 1 targeted impostor
    for u in chosen:
        gs = random.sample(gen[u], min(G, len(gen[u])))
        items += [(s, u, 0) for s in gs]
        if imp.get(u):
            fs = random.sample(imp[u], min(G, len(imp[u])))
            items += [(s, u, 1) for s in fs]
    K = torch.stack([s.key_matrix for s, _, _ in items]).to(device)
    M = torch.stack([s.mouse_matrix for s, _, _ in items]).to(device)
    KL = torch.tensor([s.key_len for s, _, _ in items],
                      dtype=torch.long, device=device)
    ML = torch.tensor([s.mouse_len for s, _, _ in items],
                      dtype=torch.long, device=device)
    UU = torch.tensor([u for _, u, _ in items], device=device)
    RR = torch.tensor([r for _, _, r in items], device=device)
    with torch.no_grad():
        model.eval()
        Z = model(K, M, KL, ML)  # L2-normalized
        D = torch.cdist(Z, Z, p=2)
    model.train()
    A, P_, N = [], [], []
    n_target = 0
    order = torch.randperm(len(items)).tolist()
    for i in order:
        if RR[i].item() != 0:
            continue  # anchors are genuine trials only
        u = UU[i].item()
        pos = ((UU == u) & (RR == 0)).nonzero().flatten().tolist()
        pos = [j for j in pos if j != i]
        if not pos:
            continue
        # hardest same-user positive
        p = max(pos, key=lambda j: D[i, j].item())
        d_ap = D[i, p].item()
        # candidate negatives: targeted first, then cross-user
        targ = ((UU == u) & (RR == 1)).nonzero().flatten().tolist()
        cross = ((UU != u) & (RR == 0)).nonzero().flatten().tolist()
        want_target = (random.random() < target_frac) and targ
        cand = targ if want_target else cross
        if not cand:  # fallback to whichever pool is non-empty
            cand = cross if cross else targ
        if not cand:
            continue
        # semi-hard: d(a,p) < d(a,n) < d(a,p)+margin; else hardest available
        semi = [j for j in cand if d_ap < D[i, j].item() < d_ap + margin]
        n = min(semi, key=lambda j: D[i, j].item()) if semi else \
            min(cand, key=lambda j: D[i, j].item())
        if (UU[n].item() == u and RR[n].item() == 1):
            n_target += 1
        A.append(i)
        P_.append(p)
        N.append(n)
    if not A:
        return None
    get = lambda idx, T: T[idx]
    ai = torch.tensor(A, device=device)
    pi = torch.tensor(P_, device=device)
    ni = torch.tensor(N, device=device)
    return (get(ai, K), get(ai, M), get(ai, KL), get(ai, ML),
            get(pi, K), get(pi, M), get(pi, KL), get(pi, ML),
            get(ni, K), get(ni, M), get(ni, KL), get(ni, ML),
            n_target / len(A))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/content/raw_kmt_dataset")
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--P", type=int, default=16, help="users per mining batch")
    ap.add_argument("--G", type=int, default=4, help="genuine trials per user")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--target_frac", type=float, default=0.5)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Exp03 device: {device}")

    cfg = {"key_max": KEY_MAX, "mouse_max": MOUSE_MAX, "data": str(a.data)}
    t0 = time.time()
    if CACHE_V3.exists():
        with open(CACHE_V3, "rb") as f:
            blob = pickle.load(f)
        assert blob.get("config") == cfg, \
            f"Stale {CACHE_V3}: config changed {blob.get('config')} vs {cfg}; delete it."
        all_samples = blob["samples"]
        print(f"[*] Loaded {len(all_samples)} v2 trials from cache ({time.time()-t0:.1f}s)")
    else:
        raw = Path(a.data)
        assert raw.exists(), f"Dataset folder not found: {raw}"
        all_samples, _ = load_kmt_dataset_v2(raw)
        with open(CACHE_V3, "wb") as f:
            pickle.dump({"config": cfg, "samples": all_samples}, f)
        print(f"[*] Parsed {len(all_samples)} v2 trials from JSON ({time.time()-t0:.1f}s), cached.")

    # IDENTICAL split arithmetic to baseline (same seed)
    all_users = sorted({s.user_idx for s in all_samples})
    random.shuffle(all_users)
    n = len(all_users)
    n_tr, n_va = int(n * 52 / 88), int(n * 16 / 88)
    tr = set(all_users[:n_tr])
    va = set(all_users[n_tr:n_tr + n_va])
    te = set(all_users[n_tr + n_va:])
    pick = lambda S: [s for s in all_samples if s.user_idx in S]
    train_s, val_s, test_s = pick(tr), pick(va), pick(te)
    print(f"[*] Split: {len(tr)} train | {len(va)} val | {len(te)} test users (subject-disjoint)")

    index = build_index_v2(train_s)
    model = MultimodalBiometricEncoderV2(128).to(device)
    opt = optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    loss_fn = nn.TripletMarginLoss(margin=a.margin, p=2.0)

    print("\n" + "=" * 66 + "\n  EXP03: MASKED ENCODER + IMPOSTOR-AWARE MINING\n" + "=" * 66)
    best_auc, best_ep = -1.0, -1
    for ep in range(1, a.epochs + 1):
        te0 = time.time()
        model.train()
        losses, tfrac = [], []
        for _ in range(a.steps):
            out = mine_batch(model, index, a.P, a.G, a.margin, a.target_frac, device)
            if out is None:
                continue
            (ak, am, akl, aml, pk, pm, pkl, pml,
             nk, nm, nkl, nml, tf) = out
            B = ak.size(0)
            z = embed_pool(model, torch.cat([ak, pk, nk]), torch.cat([am, pm, nm]),
                           torch.cat([akl, pkl, nkl]), torch.cat([aml, pml, nml]))
            loss = loss_fn(z[:B], z[B:2 * B], z[2 * B:])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
            tfrac.append(tf)
        sched.step()
        msg = (f"Epoch [{ep:02d}/{a.epochs:02d}] Loss: {float(np.mean(losses)):.4f} "
               f"| tgtNeg: {float(np.mean(tfrac)):.2f}")
        if ep % a.eval_every == 0 or ep == a.epochs:
            yv, sv = evaluate_few_shot_v2(model, val_s, a.K, device)
            eer, _, auc = compute_eer(yv, sv)
            msg += f" | Val EER: {eer*100:5.2f}% | Val AUC: {auc:.3f}"
            if auc > best_auc:
                best_auc, best_ep = auc, ep
                torch.save(model.state_dict(), BEST_PTH)
        print(msg + f" | lr {opt.param_groups[0]['lr']:.2e} | {time.time()-te0:.1f}s",
              flush=True)

    model.load_state_dict(torch.load(BEST_PTH, map_location=device))
    print(f"[*] Restored best-val checkpoint: epoch {best_ep} (val AUC {best_auc:.4f})")
    yv, sv = evaluate_few_shot_v2(model, val_s, a.K, device)
    v_eer, thr_val, v_auc = compute_eer(yv, sv)
    yt, st = evaluate_few_shot_v2(model, test_s, a.K, device)
    t_eer, _, t_auc = compute_eer(yt, st)
    far = float(np.mean(st[yt == 0] >= thr_val))
    frr = float(np.mean(st[yt == 1] < thr_val))

    print("\n" + "=" * 66 +
          f"\n  EXP03 UNSEEN TEST USERS ({len(te)}), K={a.K} few-shot (best-val ep {best_ep})\n" +
          "=" * 66)
    print(f"Val:   EER {v_eer*100:.2f}% | AUC {v_auc:.4f}")
    print(f"Test:  EER {t_eer*100:.2f}% | AUC {t_auc:.4f}")
    print(f"At val-tuned threshold {thr_val:.3f}: FAR {far*100:.2f}% | FRR {frr*100:.2f}")
    torch.save(model.state_dict(), OUT_PTH)
    print(f"[*] Saved {BEST_PTH} and {OUT_PTH} (baseline pth untouched)")

    print(f"\n  FROZEN-MODEL K-SWEEP (no retraining):")
    print(f"{'K':>3} | {'Test EER':>8} | {'AUC':>7} | {'FAR':>7} | {'FRR':>7} | {'thr(V)':>7}")
    for K in (1, 3, 5, 7, 9):
        yv2, sv2 = evaluate_few_shot_v2(model, val_s, K, device)
        _, thr, _ = compute_eer(yv2, sv2)
        yt2, st2 = evaluate_few_shot_v2(model, test_s, K, device)
        eer, _, auc = compute_eer(yt2, st2)
        far2 = float(np.mean(st2[yt2 == 0] >= thr))
        frr2 = float(np.mean(st2[yt2 == 1] < thr))
        print(f"{K:>3} | {eer*100:7.2f}% | {auc:.4f} | "
              f"{far2*100:6.2f}% | {frr2*100:6.2f}% | {thr:.3f}")


if __name__ == "__main__":
    main()
