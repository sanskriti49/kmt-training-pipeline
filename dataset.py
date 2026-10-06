"""
dataset.py - Robust KMT Data Parser with Dwell Overlap Fix
"""
import json
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import torch


class KMTSample:
    def __init__(self, key_matrix: np.ndarray, mouse_matrix: np.ndarray, user_idx: int, is_genuine: bool):
        self.key_matrix = torch.tensor(key_matrix, dtype=torch.float32)
        self.mouse_matrix = torch.tensor(mouse_matrix, dtype=torch.float32)
        self.user_idx = user_idx
        self.is_genuine = is_genuine


def process_key_events(events: List[dict], max_len: int = 64) -> np.ndarray:
    matrix = np.zeros((max_len, 4), dtype=np.float32)
    if not events:
        return matrix

    # Fix: Store (press_time, row_index) to correctly assign dwell times on overlapping keys
    press_dict: Dict[str, Tuple[float, int]] = {}
    last_press_time = -1.0
    last_release_time = -1.0
    row_idx = 0

    for e in events:
        if not isinstance(e, dict) or "Epoch" not in e or "Key" not in e:
            continue

        epoch = float(e["Epoch"])
        key = str(e["Key"]).lower()
        event_type = str(e.get("Event", "")).lower()

        if event_type == "pressed":
            if row_idx >= max_len:
                continue  # keep scanning so late releases don't break
            press_dict[key] = (epoch, row_idx)
            flight_pp = (epoch - last_press_time) if last_press_time > 0 else 0.0
            flight_rp = (epoch - last_release_time) if last_release_time > 0 else 0.0
            matrix[row_idx, 1] = np.clip(flight_rp, 0.0, 2.0)
            matrix[row_idx, 2] = np.clip(flight_pp, 0.0, 2.0)
            matrix[row_idx, 3] = 1.0 if key in ("backspace", "delete") else 0.0
            last_press_time = epoch
            row_idx += 1

        elif event_type == "released":
            if key in press_dict:
                t0, r = press_dict.pop(key)
                matrix[r, 0] = np.clip(epoch - t0, 0.01, 1.5)
            last_release_time = epoch

    return matrix


def process_mouse_events(events: List[dict], max_len: int = 128) -> np.ndarray:
    matrix = np.zeros((max_len, 5), dtype=np.float32)
    # Filter out non-event objects such as {'false_enters': 0}
    valid = [e for e in events if isinstance(e, dict) and "Coordinates" in e and "Epoch" in e]
    if len(valid) < 2:
        return matrix

    row_idx = 0
    prev_v = 0.0

    for i in range(1, min(len(valid), max_len + 1)):
        p1 = valid[i - 1]
        p2 = valid[i]

        t1, t2 = float(p1["Epoch"]), float(p2["Epoch"])
        dt = t2 - t1
        dx = float(p2["Coordinates"][0]) - float(p1["Coordinates"][0])
        dy = float(p2["Coordinates"][1]) - float(p1["Coordinates"][1])
        dist = np.hypot(dx, dy)

        is_click = 1.0 if "press" in str(p2.get("Event", "")).lower() else 0.0

        if dt > 0.001:
            v = dist / dt
            a = (v - prev_v) / dt
            prev_v = v
        else:
            v = 0.0
            a = 0.0

        matrix[row_idx, 0] = np.clip(dx / 100.0, -10.0, 10.0)
        matrix[row_idx, 1] = np.clip(dy / 100.0, -10.0, 10.0)
        matrix[row_idx, 2] = np.clip(v / 1000.0, 0.0, 15.0)
        matrix[row_idx, 3] = np.clip(a / 10000.0, -10.0, 10.0)
        matrix[row_idx, 4] = is_click

        row_idx += 1
        if row_idx >= max_len:
            break

    return matrix


def load_kmt_dataset(raw_dir: Path) -> Tuple[List[KMTSample], List[int]]:
    files = sorted(list(raw_dir.glob("raw_kmt_user_*.json")))
    all_samples = []
    user_labels = []

    for user_idx, fpath in enumerate(files):
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        for _, test_val in data.get("true_data", {}).items():
            k_mat = process_key_events(test_val.get("key_events", []), max_len=64)
            m_mat = process_mouse_events(test_val.get("mouse_events", []), max_len=128)
            all_samples.append(KMTSample(k_mat, m_mat, user_idx, is_genuine=True))
            user_labels.append(user_idx)

        for _, test_val in data.get("false_data", {}).items():
            k_mat = process_key_events(test_val.get("key_events", []), max_len=64)
            m_mat = process_mouse_events(test_val.get("mouse_events", []), max_len=128)
            all_samples.append(KMTSample(k_mat, m_mat, user_idx, is_genuine=False))
            user_labels.append(user_idx)

    return all_samples, user_labels