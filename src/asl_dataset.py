"""
ASLMed Dataset Manager: Landmark Extraction, Anti-Flicker Filtering,
Bilateral Augmentation, and Dataset Loaders.

Supports:
- Signs 1..40 (Isolated medical signs)
- Signs 41..50 (Extended clinical phrases & vitals)
- Custom user-recorded signs (Sign Studio)
"""

import re
import threading
import time
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .asl_model import (D_IN, N_LM, flip_bilateral, resample, to_features)

MEANINGS = {
    # 1 - 40 Core Medical Signs
    1: "Body", 2: "Shoulder", 3: "Nurse", 4: "Fingers", 5: "Feet", 6: "Skin",
    7: "Neck", 8: "Doctor", 9: "Lips", 10: "Teeth", 11: "Nose", 12: "Tongue",
    13: "Eyes", 14: "Mouth", 15: "Weak", 16: "Bones", 17: "X-ray", 18: "Rib Cage",
    19: "Blood", 20: "Heart", 21: "Pain", 22: "Kidney", 23: "Asthma", 24: "Cough",
    25: "Swollen", 26: "Itchy", 27: "Vomit", 28: "Flu", 29: "Fever", 30: "Headache",
    31: "Stomach ache", 32: "Waist pain", 33: "Sore throat", 34: "Surgery",
    35: "Emergency", 36: "Cream", 37: "Pill", 38: "Pregnant", 39: "Tired", 40: "Stomach",
    
    # 41 - 50 Extended Clinical Phrases / Vitals from dataset_2
    41: "High blood pressure", 42: "Heart attack", 43: "Chest pain", 44: "Can't breathe",
    45: "Allergic reaction", 46: "Dizziness", 47: "Bleeding", 48: "Severe infection",
    49: "Need wheelchair", 50: "Call ambulance",
}

NAME_RE = re.compile(r"(.+)_(\d{1,2})_([abc])", re.I)
PROJECT_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_DIR / "cache_landmarks"


# ============================ Landmark Extraction ============================
def make_holistic(complexity=1):
    import mediapipe as mp
    return mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=complexity,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )


def landmarks_from_bgr(bgr, holistic, return_face=False):
    """Process single BGR frame -> (67, 3) landmark array; NaN where missing.
    If return_face=True, returns (lm_67, face_pts_468).
    """
    res = holistic.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    out = np.full((N_LM, 3), np.nan, np.float32)

    def fill(lms, a, b):
        if lms:
            pts = [[l.x, l.y, l.z] for l in lms.landmark][: b - a]
            out[a:a + len(pts)] = pts

    fill(res.pose_landmarks, 0, 25)
    fill(res.left_hand_landmarks, 25, 46)
    fill(res.right_hand_landmarks, 46, 67)

    if return_face:
        face_pts = None
        if res.face_landmarks:
            face_pts = np.array([[l.x, l.y, l.z] for l in res.face_landmarks.landmark], dtype=np.float32)
        return out, face_pts
    return out


class LandmarkSmoother:
    """Temporal Exponential Moving Average (EMA) smoother to eliminate camera flicker."""
    def __init__(self, alpha=0.65):
        self.alpha = alpha
        self.prev = None

    def update(self, lm):
        if self.prev is None:
            self.prev = lm.copy()
            return lm
        smoothed = lm.copy()
        valid = ~np.isnan(lm) & ~np.isnan(self.prev)
        smoothed[valid] = self.alpha * lm[valid] + (1.0 - self.alpha) * self.prev[valid]
        self.prev = smoothed.copy()
        return smoothed

    def reset(self):
        self.prev = None


# ============================ Preprocessing & Normalization ==================
def clean_hands(a, need_shoulders=False):
    """Filter out implausible hand detections and detached hands."""
    a = a.copy()
    for t in range(len(a)):
        sh = a[t, 11:13, :2]
        if np.isnan(sh).any():
            if need_shoulders:
                a[t, 25:] = np.nan
            continue
        sw = float(np.linalg.norm(sh[0] - sh[1]))
        for s in (25, 46):
            h = a[t, s:s + 21]
            if np.isnan(h[0, 0]):
                continue
            span = np.linalg.norm(h[12, :2] - h[0, :2]) / max(sw, 1e-6)
            if sw < 0.04 or not (0.08 < span < 1.8):
                a[t, s:s + 21] = np.nan
    return a


def normalize(raw, need_shoulders=False):
    """Center landmarks on shoulders and scale by shoulder width."""
    raw = clean_hands(raw, need_shoulders)
    sh = raw[:, [11, 12], :2]
    valid = ~np.isnan(sh).any(axis=(1, 2))
    if valid.sum() == 0:
        c, s = np.array([0.5, 0.5]), 0.25
    else:
        sh = sh[valid]
        c = np.median(sh.mean(1), 0)
        s = max(float(np.median(np.linalg.norm(sh[:, 0] - sh[:, 1], axis=1))), 1e-3)
    center = np.array([c[0], c[1], 0.0], np.float32)
    return ((raw - center) / s).astype(np.float32)


def trim(seq, pad=4):
    """Trim leading and trailing idle frames where neither hand is detected."""
    has = ~np.isnan(seq[:, 25, 0]) | ~np.isnan(seq[:, 46, 0])
    idx = np.where(has)[0]
    if len(idx) < 3:
        return seq
    return seq[max(idx[0] - pad, 0): idx[-1] + pad + 1]


def augment(seq, allow_bilateral=True):
    """Apply spatial jitter, rotation, scale, hand dropout, and optional bilateral flip."""
    seq = seq.copy()
    
    # 50% probability to horizontally mirror (bilateral flip) for ambidextrous robustness
    if allow_bilateral and np.random.rand() > 0.5:
        seq = flip_bilateral(seq)
        
    # Random slight 2D rotation (-10 to 10 deg)
    a = np.deg2rad(np.random.uniform(-10, 10))
    c, s = np.cos(a), np.sin(a)
    x, y = seq[..., 0].copy(), seq[..., 1].copy()
    seq[..., 0], seq[..., 1] = c * x - s * y, s * x + c * y
    
    # Random scaling
    seq *= np.random.uniform(0.9, 1.1)
    
    # Random translation
    seq[..., :2] += np.random.uniform(-0.15, 0.15, 2)
    
    # Gaussian jitter
    seq += np.random.normal(0, 0.01, seq.shape).astype(np.float32)
    
    # Simulate occasional hand tracking dropouts (5% frame dropout)
    for a_, b_ in ((25, 46), (46, 67)):
        drop = np.random.rand(len(seq)) < 0.05
        seq[drop, a_:b_] = np.nan
        
    return seq


def prepare_sequence(x_norm, train=False, crop=(0.0, 0.0), seq_len=48, allow_bilateral=True):
    """Normalize, augment (if train), resample to fixed seq_len, and compute features."""
    x = trim(x_norm)
    if train:
        x = augment(x, allow_bilateral=allow_bilateral)
    resampled = resample(x, train=train, crop=crop, seq_len=seq_len)
    return to_features(resampled)


def parse_filename(path, max_cls=50):
    m = NAME_RE.fullmatch(Path(path).stem)
    if not m:
        return None
    cls_id = int(m.group(2))
    if not (1 <= cls_id <= max_cls):
        return None
    return m.group(1), cls_id, m.group(3).lower()


# ============================ Dataset Class ==================================
class ASLDataset(Dataset):
    def __init__(self, items, train=True, seq_len=48, allow_bilateral=True):
        self.items = items
        self.train = train
        self.seq_len = seq_len
        self.allow_bilateral = allow_bilateral

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        x, y, _ = self.items[i]
        feats = prepare_sequence(
            x,
            train=self.train,
            seq_len=self.seq_len,
            allow_bilateral=self.allow_bilateral
        )
        return torch.from_numpy(feats), y


def load_landmarks_cache(cache_dir=CACHE_DIR, max_cls=40):
    """Load cached .npy landmarks into memory with labels and signer info."""
    data = []
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return data
    for f in sorted(cache_path.glob("*.npy")):
        info = parse_filename(f, max_cls=max_cls)
        if info:
            signer, cls_id, take = info
            raw = np.load(f)
            data.append((normalize(raw), cls_id - 1, signer))
    return data


# ============================ Camera & Motion Tracking ========================
class LatestFrame(threading.Thread if 'threading' in globals() else object):
    """Buffered-frame eliminator to keep camera latency at 0ms."""
    def __init__(self, cam):
        import threading
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(cam)
        self.is_opened = self.cap.isOpened()
        # Fallback to alternate camera indices if primary requested index is not available
        if not self.is_opened and isinstance(cam, int):
            for alt_idx in [0, 1, 2]:
                if alt_idx != cam:
                    alt_cap = cv2.VideoCapture(alt_idx)
                    if alt_cap.isOpened():
                        self.cap = alt_cap
                        self.is_opened = True
                        break
                    alt_cap.release()
        if self.is_opened:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame, self.n, self.alive = None, 0, True
        self.lock = threading.Lock()
        if self.is_opened:
            self.start()

    def run(self):
        import time
        while self.alive:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self.lock:
                self.frame, self.n = f, self.n + 1

    def get(self):
        with self.lock:
            return self.frame, self.n

    def stop(self):
        self.alive = False
        self.join(timeout=1)
        self.cap.release()


def hand_speed(prev, cur, dt):
    """Compute instantaneous hand speed normalized to shoulder width."""
    sh = cur[11:13, :2]
    if np.isnan(sh).any():
        return 0.0
    sw = max(float(np.linalg.norm(sh[0] - sh[1])), 1e-3)
    sp = []
    for s in (25, 46):
        c = cur[s:s + 21, :2]
        if np.isnan(c[0, 0]):
            continue
        if prev is None or np.isnan(prev[s, 0]):
            sp.append(2.0)
            continue
        sp.append(float(np.mean(np.linalg.norm(c - prev[s:s + 21, :2], axis=1))) / sw / max(dt, 0.02))
    return max(sp) if sp else 0.0
