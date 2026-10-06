"""dataset_v2.py - Length-aware KMT samples (Exp03+).

Reuses the validated process_key_events / process_mouse_events from
dataset.py (same features, same scales), but additionally returns true
sequence lengths so model_v2.py can do masked pooling instead of
averaging over zero padding.

Length definition (must match what the GRU actually sees):
  key_len   = number of press rows written (row_idx), clipped to max_len
  mouse_len = number of movement rows written, clipped to max_len
"""
import json
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch

from dataset import process_key_events, process_mouse_events

KEY_MAX = 96    # raised from 64: presses/trial median ~45, p90 ~62
MOUSE_MAX = 256  # raised from 128: median length 316; still truncates the
                 # extreme tail (p90 ~680) but keeps 2x more trajectory


class KMTSampleV2:
    def __init__(self, key_matrix: np.ndarray, mouse_matrix: np.ndarray,
                 key_len: int, mouse_len: int, user_idx: int, is_genuine: bool):
        self.key_matrix = torch.tensor(key_matrix, dtype=torch.float32)
        self.mouse_matrix = torch.tensor(mouse_matrix, dtype=torch.float32)
        self.key_len = int(key_len)
        self.mouse_len = int(mouse_len)
        self.user_idx = user_idx
        self.is_genuine = is_genuine


def _counted_key_matrix(events, max_len: int):
    """Same as process_key_events, but also returns press-row count."""
    mat = process_key_events(events, max_len=max_len)
    n_press = 0
    for e in events:
        if isinstance(e, dict) and str(e.get("Event", "")).lower() == "pressed":
            n_press += 1
    return mat, min(n_press, max_len)


def _counted_mouse_matrix(events, max_len: int):
    """Same as process_mouse_events, but also returns movement-row count."""
    mat = process_mouse_events(events, max_len=max_len)
    valid = [e for e in events
             if isinstance(e, dict) and "Coordinates" in e and "Epoch" in e]
    return mat, min(max(len(valid) - 1, 0), max_len)


def load_kmt_dataset_v2(raw_dir: Path,
                        key_max: int = KEY_MAX,
                        mouse_max: int = MOUSE_MAX) -> Tuple[List[KMTSampleV2], List[int]]:
    files = sorted(list(raw_dir.glob("raw_kmt_user_*.json")))
    all_samples, user_labels = [], []
    for user_idx, fpath in enumerate(files):
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        for _, trial in data.get("true_data", {}).items():
            k_mat, k_len = _counted_key_matrix(trial.get("key_events", []), key_max)
            m_mat, m_len = _counted_mouse_matrix(trial.get("mouse_events", []), mouse_max)
            all_samples.append(KMTSampleV2(k_mat, m_mat, k_len, m_len, user_idx, True))
            user_labels.append(user_idx)
        for _, trial in data.get("false_data", {}).items():
            k_mat, k_len = _counted_key_matrix(trial.get("key_events", []), key_max)
            m_mat, m_len = _counted_mouse_matrix(trial.get("mouse_events", []), mouse_max)
            all_samples.append(KMTSampleV2(k_mat, m_mat, k_len, m_len, user_idx, False))
            user_labels.append(user_idx)
    return all_samples, user_labels
