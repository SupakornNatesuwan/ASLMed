"""
ASL MAIN APP (v3 Expert Interpreter Edition) -- Patient <-> Nurse/Doctor Translator
Run directly from VS Code (F5 or Run Python File) or terminal:
    python asl_main.py

Combines THREE Recognition Engines in Real Time:
  1. MEDICAL MODEL (50 signs) -> Trained BiGRU+Attention / TorchScript with Bilateral TTA
        Body, Shoulder, Nurse, Fingers, Feet, Skin, Neck, Doctor, Lips, Teeth, Nose,
        Tongue, Eyes, Mouth, Weak, Bones, X-ray, Rib Cage, Blood, Heart, Pain, Kidney,
        Asthma, Cough, Swollen, Itchy, Vomit, Flu, Fever, Headache, Stomach ache,
        Waist pain, Sore throat, Surgery, Emergency, Cream, Pill, Pregnant, Tired,
        Stomach, High blood pressure, Heart attack, Chest pain, Can't breathe,
        Allergic reaction, Dizziness, Bleeding, Severe infection, Need wheelchair,
        Call ambulance
  2. DAILY ASL KINEMATIC MODEL (18 daily signs, 100% offline, 0ms latency) ->
        hello, I, my, you, want, feel, need, please, thank-you, sorry, yes, no,
        help, good, water, bathroom, where, more
  3. ROBOFLOW DAILY ASL CLOUD MODEL (asl-avoix/1) ->
        apple, can, good, get, have, help, how, I, like, love, my, no, sorry,
        thank-you, want, yes, you, your

And uses the Expert AI Sentence Interpreter (asl_sentence.py) to formulate natural,
smooth English sentences (e.g. ["I", "want", "pill", "headache"] ->
"I want a pill for my headache." or "I have a headache, so I want a pill.").

Keys:
    ENTER      Finish + speak sentence aloud     BACKSPACE  Delete last word
    TAB        Switch sentence phrasing style    A          Accept "not sure" guess
    C          Clear current words               R          Repeat last spoken sentence
    S          Toggle hand skeleton overlay      V          Voice on/off
    M          Mirror image sent to Roboflow     Q / ESC    Quit
    Quick sign keys: H hello | F feel | N need | W water | B bathroom | P please | I "I" | Y yes | O no
"""
import argparse
import os
import queue
import subprocess
import sys
import threading
import time
import warnings
from collections import deque
from pathlib import Path

from dotenv import load_dotenv

# Load secret API keys from .env if present
load_dotenv()

import cv2
import numpy as np
import torch

warnings.filterwarnings("ignore", category=FutureWarning)

from asl_sentence import canon, realise, realise_variants
from asl_sign_train import (
    D_IN, MEANINGS, MODEL_PATH, TTA_CROPS, LatestFrame, hand_speed,
    landmarks_from_bgr, clean_hands, make_holistic, model_from_state,
    normalize, resample, to_features, trim
)
from asl_dataset import LandmarkSmoother
from asl_model import TorchScriptPredictor, flip_bilateral

FONT = cv2.FONT_HERSHEY_SIMPLEX
PACE = {
    "slow":   dict(still_gap=0.90, end_gap=0.50, cooldown=0.90, min_dur=0.50, max_seg=5.0),
    "normal": dict(still_gap=0.65, end_gap=0.38, cooldown=0.55, min_dur=0.42, max_seg=4.5),
    "fast":   dict(still_gap=0.48, end_gap=0.28, cooldown=0.30, min_dur=0.38, max_seg=4.0),
}
QUICK_WORDS = {
    ord("h"): "hello", ord("f"): "feel", ord("n"): "need",
    ord("w"): "water", ord("b"): "bathroom", ord("p"): "please",
    ord("i"): "i", ord("y"): "yes", ord("o"): "no",
    ord("t"): "thank-you", ord("d"): "doctor",
}

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]
POSE_CONNECTIONS = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]


# ============================ 1. Medical Model (50 Signs) ====================
class MedicalModel:
    def __init__(self, path):
        ts_path = Path(path).with_suffix(".torchscript")
        self.predictor = None
        self.models = []
        if Path(path).exists():
            ck = torch.load(path, map_location="cpu")
            if ck.get("d_in") != D_IN or "states" not in ck:
                raise SystemExit(f"{path} is from an older version. Run: python asl_sign_train.py train --final")
            self.models = [model_from_state(s) for s in ck["states"]]
        elif ts_path.exists():
            self.predictor = TorchScriptPredictor(ts_path)
        else:
            raise SystemExit(f"Model not found at {path}. Run: python asl_sign_train.py train --final")

    def predict(self, body):
        x = trim(normalize(np.stack(body), need_shoulders=True))
        crops = [to_features(resample(x, False, c)) for c in TTA_CROPS]
        # Bilateral TTA: evaluate both normal and horizontally mirrored hands
        x_flip = flip_bilateral(x)
        crops += [to_features(resample(x_flip, False, c)) for c in TTA_CROPS]
        feats = np.stack(crops)

        if self.models:
            with torch.no_grad():
                t_feats = torch.from_numpy(feats)
                p = torch.stack([torch.softmax(m(t_feats), 1) for m in self.models]).mean((0, 1)).numpy()
        else:
            p = self.predictor.predict_probs(feats).mean(axis=0)

        order = p.argsort()[::-1][:3]
        return [(canon(MEANINGS.get(i + 1, f"sign {i+1}")), float(p[i])) for i in order]


# ============================ 2. Daily ASL Kinematic Model ===================
class DailyASLModel:
    """
    Recognizes everyday ASL signs ('hello', 'I', 'my', 'you', 'want', 'feel',
    'need', 'please', 'thank-you', 'sorry', 'water', 'bathroom', 'yes', 'no',
    'help', 'good', 'where', 'more') directly from 3D MediaPipe Holistic landmarks.
    Works 100% offline with 0ms latency and complements Roboflow + MedicalModel.
    """

    @staticmethod
    def _finger_ext(hand):
        """Return extension ratios for (index, middle, ring, pinky) on a (T, 21, 3) hand."""
        valid = ~np.isnan(hand[:, 0, 0])
        if valid.sum() == 0:
            return np.zeros(4, dtype=np.float32)
        h = hand[valid]
        w = h[:, 0, :2]
        ratios = []
        for mcp_i, tip_i in ((5, 8), (9, 12), (13, 16), (17, 20)):
            d_mcp = np.linalg.norm(h[:, mcp_i, :2] - w, axis=-1)
            d_tip = np.linalg.norm(h[:, tip_i, :2] - w, axis=-1)
            r = np.median(d_tip / np.maximum(d_mcp, 1e-4))
            ratios.append(float(r))
        return np.array(ratios, dtype=np.float32)

    def predict(self, body):
        seq = trim(normalize(np.stack(body), need_shoulders=True))
        if len(seq) < 5:
            return []

        lh = seq[:, 25:46]
        rh = seq[:, 46:67]
        l_valid = ~np.isnan(lh[:, 0, 0])
        r_valid = ~np.isnan(rh[:, 0, 0])
        l_ratio = float(l_valid.mean())
        r_ratio = float(r_valid.mean())

        if l_ratio < 0.25 and r_ratio < 0.25:
            return []

        both_hands = (l_ratio >= 0.45 and r_ratio >= 0.45)
        dom = rh if r_ratio >= l_ratio else lh
        dom_valid = r_valid if r_ratio >= l_ratio else l_valid
        dom_pts = dom[dom_valid]

        # Key body anchors in shoulder-normalized coordinates
        nose = np.nanmedian(seq[:, 0, :2], axis=0)
        if np.isnan(nose).any():
            nose = np.array([0.0, -0.70], dtype=np.float32)
        mouth = np.nanmedian(seq[:, [9, 10], :2].mean(axis=1), axis=0)
        if np.isnan(mouth).any():
            mouth = np.array([0.0, -0.45], dtype=np.float32)
        chest = np.array([0.0, 0.25], dtype=np.float32)

        # Dominant hand finger states
        ext = self._finger_ext(dom)  # [index, middle, ring, pinky]
        n_ext = int((ext > 1.52).sum())
        n_curled = int((ext < 1.35).sum())
        is_open = n_ext >= 3
        is_fist = n_ext == 0 and n_curled >= 3
        is_index_only = (ext[0] > 1.55 and ext[1] < 1.40 and ext[2] < 1.38 and ext[3] < 1.38)
        is_pinky_only = (ext[3] > 1.50 and ext[0] < 1.40 and ext[1] < 1.40)
        is_w_shape = (ext[0] > 1.50 and ext[1] > 1.50 and ext[2] > 1.45 and ext[3] < 1.38)
        is_two_fingers = (ext[0] > 1.50 and ext[1] > 1.50 and ext[2] < 1.38 and ext[3] < 1.38)

        # Dominant hand trajectory (wrist & index tip)
        wrist = dom_pts[:, 0, :2]
        tip_center = dom_pts[:, [8, 12], :2].mean(axis=1)
        pos_start = tip_center[:max(2, len(tip_center) // 4)].mean(axis=0)
        pos_end = tip_center[-max(2, len(tip_center) // 4):].mean(axis=0)
        pos_med = np.median(tip_center, axis=0)
        wrist_med = np.median(wrist, axis=0)

        dx = float(pos_end[0] - pos_start[0])
        dy = float(pos_end[1] - pos_start[1])  # +dy is downward, -dy is upward
        outward_dx = float(abs(pos_end[0]) - abs(pos_start[0]))
        diffs = np.diff(wrist, axis=0)
        path_len = float(np.linalg.norm(diffs, axis=-1).sum())
        x_osc = float(np.abs(diffs[:, 0]).sum())
        y_osc = float(np.abs(diffs[:, 1]).sum())
        net_disp = float(np.linalg.norm(wrist[-1] - wrist[0]))

        dist_nose = float(np.linalg.norm(pos_med - nose))
        dist_mouth_start = float(np.linalg.norm(pos_start - mouth))
        dist_mouth_med = float(np.linalg.norm(pos_med - mouth))
        dist_chest = float(np.linalg.norm(wrist_med - chest))
        dist_tip_chest = float(np.linalg.norm(pos_med - chest))

        scores = {}

        # ---- Two-Handed Daily Signs: want, help, more, good ----
        if both_hands:
            both_mask = l_valid & r_valid
            if both_mask.sum() >= 3:
                lw = lh[both_mask, 0, :2]
                rw = rh[both_mask, 0, :2]
                ltip = lh[both_mask, 8, :2]
                rtip = rh[both_mask, 8, :2]
                inter_wrist = np.linalg.norm(lw - rw, axis=-1)
                inter_tip = np.linalg.norm(ltip - rtip, axis=-1)
                min_tip_d = float(inter_tip.min())
                start_tip_d = float(inter_tip[:2].mean())
                end_tip_d = float(inter_tip[-2:].mean())
                both_dy = float((lw[-1, 1] - lw[0, 1] + rw[-1, 1] - rw[0, 1]) / 2.0)

                # HELP: Two hands together in front of torso lifting UPWARD
                if float(inter_wrist.min()) < 0.48 and both_dy < -0.08 and pos_med[1] > -0.15:
                    scores["help"] = min(0.88, 0.62 + abs(both_dy) * 0.8)

                # MORE: Both hands bringing fingertips together at midline
                if min_tip_d < 0.28 and start_tip_d > end_tip_d + 0.10 and pos_med[1] > -0.10:
                    scores["more"] = 0.82

                # WANT: Both hands open/clawed in neutral space pulling inward/down
                l_ext = self._finger_ext(lh)
                r_ext = self._finger_ext(rh)
                if (l_ext > 1.38).sum() >= 3 and (r_ext > 1.38).sum() >= 3:
                    if float(inter_wrist.mean()) > 0.35 and 0.0 <= pos_med[1] <= 0.75:
                        scores["want"] = 0.78

                # GOOD: One hand starts at mouth and moves down to the other hand
                if dist_mouth_start < 0.42 and dy > 0.12 and min_tip_d < 0.42:
                    scores["good"] = 0.82

        # ---- Single-Handed Daily Signs ----
        if not both_hands:
            # HELLO: Open hand near forehead/temple/eye level moving outward or waving
            if is_open and (pos_start[1] < nose[1] + 0.15 or dist_nose < 0.45):
                if outward_dx > 0.06 or x_osc > 0.22:
                    scores["hello"] = 0.86

            # THANK-YOU: Open hand starting near chin/mouth and moving forward/downward
            if is_open and dist_mouth_start < 0.40 and dy > 0.10 and pos_end[1] > mouth[1] + 0.08:
                scores["thank-you"] = 0.85

            # WATER: 'W' handshape (3 fingers up) tapping near chin/mouth
            if is_w_shape and dist_mouth_med < 0.42:
                scores["water"] = 0.86

            # FEEL: Hand on chest moving UPWARD (middle finger or open palm brushing up chest)
            if (dist_chest < 0.45 or dist_tip_chest < 0.40) and dy < -0.09 and pos_med[1] > -0.15:
                scores["feel"] = 0.84

            # PLEASE: Open flat hand rubbing chest in a circle
            if is_open and (dist_chest < 0.42 or dist_tip_chest < 0.38) and path_len > 0.42 and abs(dy) < 0.18:
                scores["please"] = 0.82

            # SORRY: Closed fist rubbing chest in a circle
            if is_fist and dist_chest < 0.42 and path_len > 0.38:
                scores["sorry"] = 0.82

            # I: Pinky extended OR index pointing inward at chest center with low motion
            if dist_tip_chest < 0.36 and path_len < 0.40:
                if is_pinky_only or is_index_only or n_ext == 1:
                    scores["i"] = 0.84
                elif is_open:
                    scores["my"] = 0.80

            # NEUTRAL SPACE SIGNS (away from face and chest center)
            in_neutral = (dist_nose > 0.42 and dist_mouth_med > 0.38 and dist_tip_chest > 0.32)
            if in_neutral:
                # YES: Fist nodding up and down in neutral space
                if is_fist and y_osc > 0.22 and y_osc >= x_osc * 0.9:
                    scores["yes"] = 0.80
                # BATHROOM: Fist shaking side-to-side in neutral space
                elif is_fist and x_osc > 0.28 and x_osc > y_osc * 1.2:
                    scores["bathroom"] = 0.78
                # NO: Two fingers (index + middle) snapping toward thumb in neutral space
                elif is_two_fingers and path_len < 0.55:
                    scores["no"] = 0.78
                # NEED: Index/hooked finger moving firmly DOWNWARD in neutral space
                elif (is_index_only or n_ext <= 1) and dy > 0.14 and net_disp > 0.12:
                    scores["need"] = 0.78
                # WHERE: Index finger wagging side-to-side in neutral space
                elif is_index_only and x_osc > 0.30 and x_osc > y_osc * 1.2:
                    scores["where"] = 0.80
                # YOU: Index finger pointing forward in neutral space with low motion
                elif is_index_only and path_len < 0.38:
                    scores["you"] = 0.78

        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:3]
        return [(canon(w), float(s)) for w, s in ranked]


# ============================ 3. Roboflow Cloud Worker =======================
class RoboflowWorker(threading.Thread):
    """Sends the newest frame to Roboflow in the background (never blocks the video)."""

    def __init__(self, api_key, api_url, model_id, conf, width=416):
        super().__init__(daemon=True)
        from inference_sdk import InferenceConfiguration, InferenceHTTPClient
        self.client = InferenceHTTPClient(api_url=api_url, api_key=api_key).configure(
            InferenceConfiguration(api_key_transport="header", confidence_threshold=conf))
        self.model_id, self.width = model_id, width
        self.lock, self.event = threading.Lock(), threading.Event()
        self.job = None
        self.results = deque(maxlen=120)
        self.last_submit_t = self.last_done_t = 0.0
        self.latency, self.error, self.alive = 0.0, "", True

    def submit(self, frame, t, mirror):
        with self.lock:
            self.job = (frame, t, mirror)
            self.last_submit_t = t
        self.event.set()

    def run(self):
        while self.alive:
            self.event.wait(0.2)
            with self.lock:
                job, self.job = self.job, None
                self.event.clear()
            if job is None:
                continue
            frame, t, mirror = job
            try:
                t0 = time.time()
                img = frame
                if img.shape[1] > self.width:
                    s = self.width / img.shape[1]
                    img = cv2.resize(img, None, fx=s, fy=s)
                if mirror:
                    img = cv2.flip(img, 1)
                preds = self.client.infer(img, model_id=self.model_id).get("predictions", [])
                dets = [(canon(p["class"]), float(p["confidence"])) for p in preds]
                with self.lock:
                    self.results.append((t, dets))
                    self.last_done_t = max(self.last_done_t, t)
                    self.latency = time.time() - t0
                    self.error = ""
            except Exception as e:
                with self.lock:
                    self.error = (str(e)[:50] or "error")
                time.sleep(0.5)

    def between(self, t0, t1):
        with self.lock:
            return [d for t, d in self.results if t0 <= t <= t1]


def rf_votes(frames):
    """Per-frame best detections inside one sign -> [(word, score)], steadier than one frame."""
    n = len(frames)
    per = {}
    for dets in frames:
        if dets:
            w, s = max(dets, key=lambda d: d[1])
            per.setdefault(w, []).append(s)
    out = [(w, float(np.mean(ss)) * (0.4 + 0.6 * len(ss) / n)) for w, ss in per.items()]
    return sorted(out, key=lambda x: -x[1])[:3]


def decide(med, rf, daily, a):
    """
    Intelligent Multi-Model Fusion (Medical 50-sign + Roboflow Daily + Local Daily Kinematic).
    - If Roboflow and Local Daily agree on a daily sign (e.g. 'I', 'want', 'yes'), boosts confidence!
    - Allows daily signs ('hello', 'feel', 'I', 'want', 'need', 'water', 'please', 'thank-you')
      to win cleanly when the Medical Model has low confidence.
    """
    cands = {w: (p * a.med_weight, "med") for w, p in med}

    for w, s in rf:
        sc = s * a.rf_weight
        if w not in cands or sc > cands[w][0]:
            cands[w] = (sc, "rf")

    med_top_score = med[0][1] if med else 0.0
    for w, s in daily:
        sc = s * a.daily_weight
        # If Roboflow also detected the same daily word, boost both!
        if w in cands and cands[w][1] == "rf":
            sc = min(0.98, max(sc, cands[w][0]) + 0.12)
            cands[w] = (sc, "rf+daily")
        # Otherwise, if Medical model is not strongly confident (< 0.72), allow local daily sign
        elif med_top_score < 0.72:
            if w not in cands or sc > cands[w][0]:
                cands[w] = (sc, "daily")

    ranked = sorted(cands.items(), key=lambda kv: -kv[1][0])
    top = [(w, s, src) for w, (s, src) in ranked[:3]]
    if not top:
        return False, []
    s2 = top[1][1] if len(top) > 1 else 0.0
    return (top[0][1] >= a.threshold and top[0][1] - s2 >= a.margin), top


# ============================ Sign Segmentation ==============================
class Segmenter:
    """Finds the start and end of each sign from hand motion (not from hands vanishing)."""

    def __init__(self, pace, start_speed, still_speed):
        self.p, self.start_speed, self.still_speed = pace, start_speed, still_speed
        self.active, self.seg, self.seg_t = False, [], []
        self.preroll = deque(maxlen=6)
        self.prev_lm = self.prev_t = None
        self.speed = self.last_hand = self.last_move = self.block_until = 0.0

    def still_progress(self, now):
        return min(max((now - self.last_move) / self.p["still_gap"], 0.0), 1.0) if self.active else 0.0

    def update(self, lm, now):
        hands = not (np.isnan(lm[25, 0]) and np.isnan(lm[46, 0]))
        raw = hand_speed(self.prev_lm, lm, now - self.prev_t) if self.prev_t else 0.0
        self.speed = 0.5 * self.speed + 0.5 * raw
        self.prev_lm, self.prev_t = lm, now
        if hands:
            self.last_hand = now
        if not self.active:
            self.preroll.append((now, lm))
            if hands and self.speed > self.start_speed and now >= self.block_until:
                self.active = True
                self.seg = [x for _, x in self.preroll]
                self.seg_t = [t for t, _ in self.preroll]
                self.last_move = now
                return "start", None
            return None, None
        self.seg.append(lm)
        self.seg_t.append(now)
        if hands and self.speed > self.still_speed:
            self.last_move = now
        p = self.p
        finished = (now - self.last_hand > p["end_gap"] or now - self.last_move > p["still_gap"]
                    or now - self.seg_t[0] > p["max_seg"])
        if not finished:
            return None, None
        ref = max(self.last_move, self.last_hand if hands else 0.0) + 0.15
        keep = [i for i, t in enumerate(self.seg_t) if t <= ref]
        n = (max(keep) + 1) if keep else len(self.seg)
        body, body_t = self.seg[:n], self.seg_t[:n]
        self.active, self.seg, self.seg_t = False, [], []
        self.preroll.clear()
        self.block_until = now + p["cooldown"]
        dur = body_t[-1] - body_t[0]
        shown = np.mean([(not np.isnan(f[25, 0])) or (not np.isnan(f[46, 0])) for f in body])
        if len(body) >= 8 and dur >= p["min_dur"] and shown >= 0.5:
            return "end", (body, body_t[0], body_t[-1])
        return "cancel", None


# ============================ Voice Synthesis ================================
PYTTSX3_CODE = (
    "import sys, pyttsx3\n"
    "e = pyttsx3.init()\n"
    "e.setProperty('rate', int(sys.argv[2]))\n"
    "e.say(sys.argv[1])\n"
    "e.runAndWait()\n"
)


class Speaker(threading.Thread):
    """Speaks every sentence reliably in a background process."""

    def __init__(self, rate, enabled):
        super().__init__(daemon=True)
        self.q, self.rate, self.enabled, self.ok = queue.Queue(), rate, enabled, True
        self.start()

    def say(self, text):
        if self.enabled and text:
            self.q.put(text)

    @staticmethod
    def _run(cmd, env=None):
        try:
            return subprocess.run(cmd, env=env, timeout=40, stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL).returncode == 0
        except Exception:
            return False

    def _windows(self, text):
        rate = max(-10, min(10, round((self.rate - 150) / 15)))
        ps = ("Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
              "$s.Rate = [int]$env:ASL_RATE; $s.Speak($env:ASL_TEXT)")
        return self._run(["powershell", "-NoProfile", "-Command", ps],
                         {**os.environ, "ASL_TEXT": text, "ASL_RATE": str(rate)})

    def _mac(self, text):
        return self._run(["say", "-r", str(self.rate + 30), text])

    def _espeak(self, text):
        return (self._run(["espeak", "-s", str(self.rate + 30), text])
                or self._run(["spd-say", "-w", text]))

    def _pyttsx3(self, text):
        return self._run([sys.executable, "-c", PYTTSX3_CODE, text, str(self.rate)])

    def _chain(self):
        if sys.platform.startswith("win"):
            return [self._windows, self._pyttsx3]
        if sys.platform == "darwin":
            return [self._mac, self._pyttsx3]
        return [self._espeak, self._pyttsx3]

    def run(self):
        while True:
            text = self.q.get()
            self.ok = any(speak(text) for speak in self._chain())


# ============================ Drawing & HUD ==================================
def draw_skeleton(view, lm):
    """Draw smooth anti-aliased pose and hand skeleton onto the flipped camera view."""
    h, w = view.shape[:2]

    def pt(idx):
        x, y = lm[idx, 0], lm[idx, 1]
        if np.isnan(x) or np.isnan(y):
            return None
        return int((1.0 - x) * w), int(y * h)

    for i, j in POSE_CONNECTIONS:
        p1, p2 = pt(i), pt(j)
        if p1 and p2:
            cv2.line(view, p1, p2, (255, 180, 0), 2, cv2.LINE_AA)

    for base, col in ((25, (80, 255, 120)), (46, (56, 189, 248))):
        if np.isnan(lm[base, 0]):
            continue
        for a, b in HAND_CONNECTIONS:
            p1, p2 = pt(base + a), pt(base + b)
            if p1 and p2:
                cv2.line(view, p1, p2, col, 2, cv2.LINE_AA)
        for k in range(21):
            p = pt(base + k)
            if p:
                cv2.circle(view, p, 3, (255, 255, 255), -1, cv2.LINE_AA)


def wrap(text, scale, thick, max_w):
    lines, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if cur and cv2.getTextSize(t, FONT, scale, thick)[0][0] > max_w:
            lines.append(cur)
            cur = w
        else:
            cur = t
    return lines + ([cur] if cur else [])


def render(frame, lm, show_skel, a, now, seg, status, status_col, cand_txt,
           words, sentence, alt_sentence, log, rf, speaker, fps):
    view = cv2.flip(frame, 1)
    if show_skel and lm is not None:
        draw_skeleton(view, lm)

    h, w = view.shape[:2]
    if rf is None:
        tag, col = "Models: Medical (50) + Daily ASL (Local)", (0, 220, 255)
    elif rf.error:
        tag, col = f"Models: Medical (50) + Daily Local (Roboflow err: {rf.error[:22]})", (0, 165, 255)
    else:
        tag, col = f"Models: Medical (50) + Daily Local + Roboflow ({rf.latency * 1000:.0f} ms)", (0, 220, 100)
    cv2.putText(view, tag, (10, 22), FONT, 0.52, col, 2, cv2.LINE_AA)

    if not speaker.enabled:
        cv2.putText(view, "Voice OFF (press V)", (10, 44), FONT, 0.52, (0, 165, 255), 2, cv2.LINE_AA)
    elif not speaker.ok:
        cv2.putText(view, "Voice ERROR: check speakers", (10, 44), FONT, 0.52, (0, 0, 255), 2, cv2.LINE_AA)
    if a.debug:
        cv2.putText(view, f"fps {fps:.0f} | speed {seg.speed:.2f} | active {int(seg.active)}",
                    (10, 66), FONT, 0.52, (0, 255, 255), 2, cv2.LINE_AA)

    panel = np.full((295, w, 3), 24, np.uint8)
    prog = seg.still_progress(now)
    cv2.rectangle(panel, (10, 6), (w - 10, 14), (60, 60, 60), -1)
    if prog > 0:
        cv2.rectangle(panel, (10, 6), (10 + int((w - 20) * prog), 14), (0, 255, 255), -1)

    cv2.putText(panel, status, (12, 44), FONT, 0.90, status_col, 2, cv2.LINE_AA)
    if cand_txt:
        cv2.putText(panel, cand_txt, (12, 68), FONT, 0.48, (175, 175, 175), 1, cv2.LINE_AA)

    gloss = "  ·  ".join(x["w"].upper() for x in words) or "(sign medical or daily ASL words: I, want, feel, hello...)"
    cv2.putText(panel, gloss, (12, 102), FONT, 0.68, (255, 255, 255), 2, cv2.LINE_AA)

    y = 136
    for ln in wrap(sentence, 0.88, 2, w - 24)[:2]:
        cv2.putText(panel, ln, (12, y), FONT, 0.88, (80, 255, 160), 2, cv2.LINE_AA)
        y += 30

    if alt_sentence and alt_sentence != sentence:
        alt_disp = f"Alt (TAB): {alt_sentence}"
        cv2.putText(panel, alt_disp[:78], (12, y - 4), FONT, 0.50, (120, 200, 255), 1, cv2.LINE_AA)

    y = 222
    for s in log[-2:]:
        cv2.putText(panel, f"Spoken: {s[:72]}", (12, y), FONT, 0.52, (145, 145, 145), 1, cv2.LINE_AA)
        y += 20

    cv2.putText(
        panel,
        "ENTER speak | TAB alt style | BKSP undo | C clear | A accept | S skeleton | Q quit",
        (12, 266), FONT, 0.43, (125, 125, 125), 1, cv2.LINE_AA
    )
    cv2.putText(
        panel,
        "Quick keys: H hello | F feel | N need | W water | B bathroom | P please | I 'I' | Y yes | O no",
        (12, 286), FONT, 0.42, (95, 150, 190), 1, cv2.LINE_AA
    )
    canvas = np.vstack([view, panel])
    if a.scale != 1.0:
        canvas = cv2.resize(canvas, None, fx=a.scale, fy=a.scale)
    return canvas


# ============================ Main Loop ======================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--width", type=int, default=480, help="processing width for pose/hands")
    ap.add_argument("--complexity", type=int, default=1, choices=[0, 1, 2])
    ap.add_argument("--pace", choices=list(PACE), default="normal", help="how patient the app is")
    ap.add_argument("--threshold", type=float, default=0.48)
    ap.add_argument("--margin", type=float, default=0.10)
    ap.add_argument("--start_speed", type=float, default=0.48)
    ap.add_argument("--still_speed", type=float, default=0.25)
    ap.add_argument("--med_weight", type=float, default=1.0, help="raise to trust medical model more")
    ap.add_argument("--rf_weight", type=float, default=1.05, help="raise to trust Roboflow more")
    ap.add_argument("--daily_weight", type=float, default=1.0, help="raise to trust local daily ASL model more")
    ap.add_argument("--rf_conf", type=float, default=0.38)
    ap.add_argument("--rf_grace", type=float, default=0.75, help="max wait for Roboflow after a sign")
    ap.add_argument("--model_id", default="asl-avoix/1")
    ap.add_argument("--no_roboflow", action="store_true")
    ap.add_argument("--mirror_rf", action="store_true", help="send mirrored image to Roboflow (toggle: M)")
    ap.add_argument("--sentence_gap", type=float, default=3.2, help="seconds of silence that end a sentence")
    ap.add_argument("--mute", action="store_true")
    ap.add_argument("--rate", type=int, default=150, help="voice speed")
    ap.add_argument("--scale", type=float, default=1.35, help="window size multiplier")
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()

    med = MedicalModel(MODEL_PATH)
    daily = DailyASLModel()
    rf = None
    if not a.no_roboflow:
        key = os.environ.get("ROBOFLOW_API_KEY", "")
        if key:
            rf = RoboflowWorker(
                key,
                os.environ.get("ROBOFLOW_API_URL", "https://serverless.roboflow.com"),
                a.model_id,
                a.rf_conf
            )
            rf.start()

    speaker = Speaker(a.rate, not a.mute)
    seg = Segmenter(PACE[a.pace], a.start_speed, a.still_speed)
    cam = LatestFrame(a.cam)
    hol = make_holistic(a.complexity)
    smoother = LandmarkSmoother(alpha=0.65)

    mirror_rf = a.mirror_rf
    show_skel = True
    style_idx = 0
    words, log = [], []
    pending, unsure = None, None
    status, status_col, cand_txt = "Ready - sign a word", (170, 170, 170), ""
    status_until, last_word_t, last_rf_sub = 0.0, 0.0, 0.0
    last_n, fps, prev_t = -1, 0.0, None
    sentence, alt_sentence = "", ""

    def update_sentences():
        nonlocal sentence, alt_sentence
        vars_ = realise_variants([x["w"] for x in words])
        if not vars_:
            sentence, alt_sentence = "", ""
        else:
            sentence = vars_[style_idx % len(vars_)]
            alt_sentence = vars_[(style_idx + 1) % len(vars_)] if len(vars_) > 1 else ""

    def push(word, src, score):
        nonlocal last_word_t
        words.append(dict(w=word, src=src, s=score))
        last_word_t = time.time()
        update_sentences()

    def finish(speak=True):
        nonlocal sentence, alt_sentence, style_idx
        if words:
            update_sentences()
            text = sentence
            log.append(text)
            if speak:
                speaker.say(text)
            words.clear()
            sentence, alt_sentence = "", ""
            style_idx = 0

    while True:
        frame, n = cam.get()
        if frame is None or n == last_n:
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
            time.sleep(0.003)
            continue
        last_n, now = n, time.time()
        if prev_t:
            fps = 0.9 * fps + 0.1 / max(now - prev_t, 1e-3)
        prev_t = now

        small = frame
        if frame.shape[1] > a.width:
            s = a.width / frame.shape[1]
            small = cv2.resize(frame, None, fx=s, fy=s)
        raw_lm = smoother.update(landmarks_from_bgr(small, hol))
        lm = clean_hands(raw_lm[None], need_shoulders=True)[0]
        ev, payload = seg.update(lm, now)

        if ev == "start":
            status, status_col, cand_txt = "Signing...", (0, 255, 255), ""
            status_until = 0.0
        if seg.active and rf is not None and now - last_rf_sub > 0.1:
            rf.submit(frame, now, mirror_rf)
            last_rf_sub = now
        if ev == "end":
            body, t0, t1 = payload
            pending = dict(
                t0=t0, t1=t1,
                med=med.predict(body),
                daily=daily.predict(body),
                created=now,
                last_sub=rf.last_submit_t if rf else 0.0
            )
            status, status_col = "Reading sign...", (0, 255, 255)
        elif ev == "cancel":
            status, status_col = "Ready - sign a word", (170, 170, 170)

        if pending:
            rf_ok = rf is not None and not rf.error
            if (not rf_ok) or rf.last_done_t >= pending["last_sub"] or now - pending["created"] > a.rf_grace:
                votes = rf_votes(rf.between(pending["t0"] - 0.05, pending["t1"] + 0.05)) if rf_ok else []
                ok, top = decide(pending["med"], votes, pending["daily"], a)
                cand_txt = "   ".join(f"{w} {s * 100:.0f}% ({src})" for w, s, src in top)
                if ok:
                    push(top[0][0], top[0][2], top[0][1])
                    status, status_col, unsure = f"{top[0][0].upper()} ({top[0][2]})", (0, 255, 0), None
                else:
                    unsure = top[0] if top else None
                    status = f"Not sure ({top[0][0]}?) - sign again, or press A" if top else "Not sure - sign again"
                    status_col = (0, 165, 255)
                status_until = now + 3.5
                pending = None

        if status_until and now > status_until and not seg.active and not pending:
            status, status_col, cand_txt, status_until = "Ready - sign a word", (170, 170, 170), "", 0.0
        elif not seg.active and not pending and now < seg.block_until and not status_until:
            status, status_col = "...", (120, 120, 120)

        # A pause after the last word ends and speaks the sentence
        if words and not seg.active and not pending and now - last_word_t > a.sentence_gap:
            finish(True)

        canvas = render(
            frame, raw_lm, show_skel, a, now, seg, status, status_col, cand_txt,
            words, sentence, alt_sentence, log, rf, speaker, fps
        )
        cv2.imshow("ASLMed - Expert Medical & Daily ASL Interpreter", canvas)
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord("q")):
            break
        elif k in (10, 13):
            if words:
                finish(True)
            elif log:
                speaker.say(log[-1])
        elif k == 9 and words:  # TAB key: switch sentence phrasing style
            style_idx += 1
            update_sentences()
            last_word_t = time.time()
        elif k in (8, 127) and words:
            words.pop()
            update_sentences()
        elif k == ord("c"):
            words.clear()
            update_sentences()
        elif k == ord("r") and log:
            speaker.say(log[-1])
        elif k == ord("a") and unsure:
            push(unsure[0], unsure[2], unsure[1])
            unsure = None
        elif k == ord("v"):
            speaker.enabled = not speaker.enabled
        elif k == ord("s"):
            show_skel = not show_skel
        elif k == ord("m"):
            mirror_rf = not mirror_rf
        elif k in QUICK_WORDS:
            push(QUICK_WORDS[k], "key", 1.0)

    cam.stop()
    if rf:
        rf.alive = False
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
