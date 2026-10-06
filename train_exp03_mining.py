"""
train_exp03_mining.py - Experiment 3: In-Batch Semi-Hard & Targeted-Aware Negative Mining
Hypothesis:
Random negative sampling starves the model of gradient signals as training progresses.
In-batch mining dynamically identifies semi-hard negatives (D(a,p) < D(a,n) < D(a,p) + alpha)
focusing specifically on both:
1. Targeted impostor trials of the same user (same card details)
2. Cross-user impostor trials (different card details)
Preserves:
- Subject-disjoint split (seed 42)
- Architecture (1D-CNN + BiGRU, 128-D)
- Validation model selection (best Val AUC)
- Evaluates on unseen test cohort only once.
"""
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
CHECKPOINT_PATH = Path("kmt_multimodal_siamese_exp03.pth")


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float, float]:
    fpr, tpr, thr = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2), float(thr[i]), float(roc_auc_score(y_true, y_scores))


def build_user_pools(samples: List[KMTSample]):
    gen_pool = {}
    imp_pool = {}
    for s in samples:
        gen_pool.setdefault(s.user_idx, [])
        imp_pool.setdefault(s.user_idx, [])
        (gen_pool if s.is_genuine else imp_pool)[s.user_idx].append(s)
    valid_users = [u for u in gen_pool if len(gen_pool[u]) >= 2 and len(imp_pool[u]) >= 1]
    return gen_pool, imp_pool, valid_users


def sample_structured_batch(gen_pool, imp_pool, users: List[int], num_users: int = 8, gen_per_user: int = 4, imp_per_user: int = 2):
    """
    Constructs a structured batch containing genuine and targeted impostor samples per user.
    """
    chosen_users = random.sample(users, min(num_users, len(users)))
    batch_samples = []
    user_ids = []
    is_gen_flags = []

    for u in chosen_users:
        # Sample genuine trials
        g_samples = random.sample(gen_pool[u], min(gen_per_user, len(gen_pool[u])))
        for s in g_samples:
            batch_samples.append(s)
            user_ids.append(u)
            is_gen_flags.append(True)
        # Sample targeted impostor trials
        i_samples = random.sample(imp_pool[u], min(imp_per_user, len(imp_pool[u])))
        for s in i_samples:
            batch_samples.append(s)
            user_ids.append(u)
            is_gen_flags.append(False)

    keys = torch.stack([s.key_matrix for s in batch_samples])
    mouse = torch.stack([s.mouse_matrix for s in batch_samples])
    return keys, mouse, np.array(user_ids), np.array(is_gen_flags)


def mine_semi_hard_triplets(embeddings: torch.Tensor, user_ids: np.ndarray, is_gen: np.ndarray, margin: float = 0.5):
    """
    Given batch embeddings (N, D), mines semi-hard triplets:
    D(a, p) < D(a, n) < D(a, p) + margin
    """
    # Pairwise Euclidean distance
    dist_matrix = torch.cdist(embeddings, embeddings, p=2.0)
    N = embeddings.size(0)

    anchors, positives, negatives = [], [], []

    for i in range(N):
        if not is_gen[i]:
            continue  # Anchors are genuine trials

        u_i = user_ids[i]

        # Positives: other genuine trials of same user
        pos_indices = [j for j in range(N) if j != i and user_ids[j] == u_i and is_gen[j]]
        if not pos_indices:
            continue

        for p in pos_indices:
            d_ap = dist_matrix[i, p].item()

            # Negatives: either targeted impostors of user u_i OR any trial of different user
            neg_candidates = []
            for k in range(N):
                if k == i or k == p:
                    continue
                # Valid negative conditions:
                is_neg = (user_ids[k] != u_i) or (user_ids[k] == u_i and not is_gen[k])
                if is_neg:
                    d_an = dist_matrix[i, k].item()
                    # Semi-hard condition: d_ap < d_an < d_ap + margin
                    if d_ap < d_an < (d_ap + margin):
                        neg_candidates.append((k, d_an))

            if neg_candidates:
                # Pick the hardest among semi-hard
                neg_candidates.sort(key=lambda x: x[1])
                n_best = neg_candidates[0][0]
                anchors.append(i)
                positives.append(p)
                negatives.append(n_best)
            else:
                # Fallback: hardest negative that produces non-zero loss (d_an < d_ap + margin)
                loss_producing = []
                for k in range(N):
                    if (user_ids[k] != u_i) or (user_ids[k] == u_i and not is_gen[k]):
                        d_an = dist_matrix[i, k].item()
                        if d_an < (d_ap + margin):
                            loss_producing.append((k, d_an))
                if loss_producing:
                    loss_producing.sort(key=lambda x: x[1])
                    anchors.append(i)
                    positives.append(p)
                    negatives.append(loss_producing[0][0])

    if not anchors:
        return None, None, None

    return torch.tensor(anchors), torch.tensor(positives), torch.tensor(negatives)


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
    ap.add_argument("--data", default="C:/Users/sansk/Downloads/behaviour_biometrics_dataset/raw_kmt_dataset")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--eval_every", type=int, default=2)
    a = ap.parse_args()

    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Training Device: {device}")

    assert CACHE.exists(), f"Cache not found: {CACHE}"
    with open(CACHE, "rb") as f:
        all_samples = pickle.load(f)
    print(f"[*] Loaded {len(all_samples)} trials from cache.")

    all_users = sorted({s.user_idx for s in all_samples})
    random.shuffle(all_users)
    n = len(all_users)
    n_tr, n_va = int(n * 52 / 88), int(n * 16 / 88)
    tr, va, te = set(all_users[:n_tr]), set(all_users[n_tr:n_tr+n_va]), set(all_users[n_tr+n_va:])
    pick = lambda S: [s for s in all_samples if s.user_idx in S]
    train_s, val_s, test_s = pick(tr), pick(va), pick(te)
    print(f"[*] Split: {len(tr)} train | {len(va)} val | {len(te)} test users (subject-disjoint)")

    gen_pool, imp_pool, valid_users = build_user_pools(train_s)
    model = MultimodalBiometricEncoder(128).to(device)
    opt = optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    loss_fn = nn.TripletMarginLoss(margin=a.margin, p=2.0)

    best_val_auc = -1.0
    best_val_eer = 1.0
    best_epoch = -1
    best_val_thr = 0.5

    print("\n" + "=" * 70)
    print("  EXPERIMENT 3: TARGETED-AWARE SEMI-HARD TRIPLET MINING")
    print("=" * 70)

    for ep in range(1, a.epochs + 1):
        t_start = time.time()
        model.train()
        losses = []
        triplets_count = 0

        for _ in range(a.steps):
            k_b, m_b, u_ids, is_g = sample_structured_batch(gen_pool, imp_pool, valid_users, num_users=8, gen_per_user=4, imp_per_user=2)
            k_b, m_b = k_b.to(device), m_b.to(device)

            z_b = model(k_b, m_b)
            a_idx, p_idx, n_idx = mine_semi_hard_triplets(z_b.detach(), u_ids, is_g, margin=a.margin)

            if a_idx is not None and len(a_idx) > 0:
                loss = loss_fn(z_b[a_idx], z_b[p_idx], z_b[n_idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())
                triplets_count += len(a_idx)

        mean_loss = float(np.mean(losses)) if losses else 0.0
        msg = f"Epoch [{ep:02d}/{a.epochs:02d}] Loss: {mean_loss:.4f} (Mined: {triplets_count} triplets)"

        if ep % a.eval_every == 0 or ep == 1:
            yv, sv = evaluate_few_shot(model, val_s, a.K, device)
            eer_v, thr_v, auc_v = compute_eer(yv, sv)
            if auc_v > best_val_auc:
                best_val_auc = auc_v
                best_val_eer = eer_v
                best_val_thr = thr_v
                best_epoch = ep
                torch.save(model.state_dict(), CHECKPOINT_PATH)
                msg += f" | Val EER: {eer_v*100:5.2f}% | Val AUC: {auc_v:.4f} --> [BEST SAVED]"
            else:
                msg += f" | Val EER: {eer_v*100:5.2f}% | Val AUC: {auc_v:.4f}"
        print(msg + f" | {time.time()-t_start:.1f}s", flush=True)

    print("\n" + "=" * 70)
    print(f"[*] Best Checkpoint: Epoch {best_epoch:02d} with Val AUC={best_val_auc:.4f}, Val EER={best_val_eer*100:.2f}%")
    print(f"[*] Calibrated Validation Threshold: {best_val_thr:.4f}")
    print("=" * 70)

    # Evaluate on strictly unseen test identities
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    model.eval()

    print("\nEVALUATION ACROSS ENROLLMENT SIZES (K in {1, 3, 5, 7, 9}):")
    print("-" * 70)
    for test_k in [1, 3, 5, 7, 9]:
        yt, st = evaluate_few_shot(model, test_s, K=test_k, device=device)
        t_eer, _, t_auc = compute_eer(yt, st)
        far = float(np.mean(st[yt == 0] >= best_val_thr))
        frr = float(np.mean(st[yt == 1] < best_val_thr))
        primary_tag = " <-- [PRIMARY REPORTING]" if test_k == 5 else ""
        print(f"K={test_k:1d} | Test EER: {t_eer*100:5.2f}% | AUC: {t_auc:.4f} | FAR: {far*100:5.2f}% | FRR: {frr*100:5.2f}%{primary_tag}")
    print("-" * 70)
    print(f"[*] Checkpoint saved: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
