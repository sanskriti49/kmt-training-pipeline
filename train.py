"""train.py - demo-ready triplet training + few-shot evaluation"""
import argparse, pickle, random, time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_curve, roc_auc_score

from dataset import load_kmt_dataset, KMTSample
from model import MultimodalBiometricEncoder

CACHE = Path("kmt_cache_v2.pkl")


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float, float]:
    fpr, tpr, thr = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2), float(thr[i]), float(roc_auc_score(y_true, y_scores))


def build_index(samples: List[KMTSample]):
    gen, imp = {}, {}
    for s in samples:
        gen.setdefault(s.user_idx, [])
        imp.setdefault(s.user_idx, [])
        (gen if s.is_genuine else imp)[s.user_idx].append(s)
    users = [u for u in gen if len(gen[u]) >= 2]
    return gen, imp, users


def sample_triplets(index, num_triplets: int):
    gen, imp, users = index
    A, P, N = [], [], []
    while len(A) < num_triplets:
        u = random.choice(users)
        a, p = random.sample(gen[u], 2)
        if random.random() < 0.5 and imp[u]:
            n = random.choice(imp[u])  # targeted impostor
        else:
            n = random.choice(gen[random.choice([x for x in users if x != u])])
        A.append(a); P.append(p); N.append(n)
    k = lambda L: torch.stack([s.key_matrix for s in L])
    m = lambda L: torch.stack([s.mouse_matrix for s in L])
    return k(A), m(A), k(P), m(P), k(N), m(N)


@torch.no_grad()
def evaluate_few_shot(model, samples, K=5, device="cpu"):
    model.eval()
    users = {}
    for s in samples:
        users.setdefault(s.user_idx, {"gen": [], "imp": []})["gen" if s.is_genuine else "imp"].append(s)

    y, sc = [], []
    emb = lambda L: model(torch.stack([s.key_matrix for s in L]).to(device),
                          torch.stack([s.mouse_matrix for s in L]).to(device))
    for d in users.values():
        if len(d["gen"]) < K + 1 or not d["imp"]:
            continue
        c = F.normalize(emb(d["gen"][:K]).mean(0, keepdim=True), dim=-1)
        sg = (emb(d["gen"][K:]) @ c.t()).squeeze(-1).cpu().numpy()
        si = (emb(d["imp"]) @ c.t()).squeeze(-1).cpu().numpy()
        sc += list(sg) + list(si)
        y += [1] * len(sg) + [0] * len(si)
    return np.array(y), np.array(sc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/content/raw_kmt_dataset")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--triplets", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--eval_every", type=int, default=3)
    a = ap.parse_args()

    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Training Device: {device}")

    t0 = time.time()
    if CACHE.exists():
        with open(CACHE, "rb") as f:
            all_samples = pickle.load(f)
        print(f"[*] Loaded {len(all_samples)} trials from cache ({time.time()-t0:.1f}s)")
    else:
        raw = Path(a.data)
        assert raw.exists(), f"Dataset folder not found: {raw}"
        all_samples, _ = load_kmt_dataset(raw)
        with open(CACHE, "wb") as f:
            pickle.dump(all_samples, f)
        print(f"[*] Parsed {len(all_samples)} trials from JSON ({time.time()-t0:.1f}s), cached.")

    all_users = sorted({s.user_idx for s in all_samples})
    random.shuffle(all_users)
    n = len(all_users)
    n_tr, n_va = int(n * 52 / 88), int(n * 16 / 88)
    tr, va, te = set(all_users[:n_tr]), set(all_users[n_tr:n_tr+n_va]), set(all_users[n_tr+n_va:])
    pick = lambda S: [s for s in all_samples if s.user_idx in S]
    train_s, val_s, test_s = pick(tr), pick(va), pick(te)
    print(f"[*] Split: {len(tr)} train | {len(va)} val | {len(te)} test users (subject-disjoint)")

    index = build_index(train_s)
    model = MultimodalBiometricEncoder(128).to(device)
    opt = optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    loss_fn = nn.TripletMarginLoss(margin=a.margin, p=2.0)

    print("\n" + "=" * 66 + "\n  SIAMESE MULTIMODAL METRIC TRAINING (triplet loss)\n" + "=" * 66)
    history = []
    for ep in range(1, a.epochs + 1):
        te0 = time.time()
        model.train()
        losses = []
        for _ in range(a.steps):
            ak, am, pk, pm, nk, nm = [t.to(device) for t in sample_triplets(index, a.triplets)]
            B = ak.size(0)
            # Concatenating into single forward pass guarantees unified BatchNorm statistics
            z = model(torch.cat([ak, pk, nk]), torch.cat([am, pm, nm]))
            loss = loss_fn(z[:B], z[B:2*B], z[2*B:])
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        mean_loss = float(np.mean(losses)); history.append(mean_loss)

        msg = f"Epoch [{ep:02d}/{a.epochs:02d}] Loss: {mean_loss:.4f}"
        if ep == 1 or ep % a.eval_every == 0 or ep == a.epochs:
            yv, sv = evaluate_few_shot(model, val_s, a.K, device)
            eer, _, auc = compute_eer(yv, sv)
            msg += f" | Val EER: {eer*100:5.2f}% | Val AUC: {auc:.3f}"
        print(msg + f" | {time.time()-te0:.1f}s", flush=True)

    # Threshold tuned on Validation, tested on strictly Unseen Test identities
    yv, sv = evaluate_few_shot(model, val_s, a.K, device)
    _, thr_val, _ = compute_eer(yv, sv)
    yt, st = evaluate_few_shot(model, test_s, a.K, device)
    t_eer, _, t_auc = compute_eer(yt, st)
    far = float(np.mean(st[yt == 0] >= thr_val)); frr = float(np.mean(st[yt == 1] < thr_val))

    print("\n" + "=" * 66 + f"\n  UNSEEN TEST USERS ({len(te)}), K={a.K} few-shot\n" + "=" * 66)
    print(f"Test EER: {t_eer*100:.2f}% | AUC: {t_auc:.4f}")
    print(f"At val-tuned threshold {thr_val:.3f}: FAR {far*100:.2f}% | FRR {frr*100:.2f}%")

    torch.save(model.state_dict(), "kmt_multimodal_siamese.pth")
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 3.5))
        plt.plot(range(1, len(history)+1), history, marker="o", color="royalblue")
        plt.xlabel("Epoch"); plt.ylabel("Triplet loss"); plt.title("Multimodal Training Loss"); plt.grid(alpha=.3)
        plt.tight_layout(); plt.savefig("loss_curve.png", dpi=150)
        print("[*] Saved loss_curve.png and kmt_multimodal_siamese.pth")
    except ImportError:
        pass


if __name__ == "__main__":
    main()