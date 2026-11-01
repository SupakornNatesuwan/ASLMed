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

from src.asl_sentence import canon, realise, realise_variants
from src.asl_dataset import (
    MEANINGS, LandmarkSmoother, LatestFrame, hand_speed,
    landmarks_from_bgr, clean_hands, make_holistic,
    normalize, resample, trim
)
from src.asl_model import (
    D_IN, MODEL_PATH, TTA_CROPS, TorchScriptPredictor,
    flip_bilateral, model_from_state, to_features
)
from src.asl_intent import SmartIntentEngine
from src.asl_ui import render_modern_dashboard, wrap_text as wrap
from src.asl_face_mesh import FaceMeshAnalyzer
from src.asl_fingerspelling import FingerspellingEngine
from src.asl_clinical_agents import (
    TriageAgent, TriageAssessment,
    DrugSafetyAgent, DrugSafetyAlert,
    ClinicalScribeAgent,
    TwoWayDoctorAgent,
    DoctorConsoleWorker,
    DoctorVoiceWorker,
    DiagnosticAgent, ClinicalDiagnosis,
)
from src.asl_session_exporter import ClinicalSession

FONT = cv2.FONT_HERSHEY_SIMPLEX
PACE = {
    "slow":   dict(still_gap=0.45, end_gap=0.26, cooldown=0.18, min_dur=0.28, max_seg=4.5),
    "normal": dict(still_gap=0.26, end_gap=0.16, cooldown=0.10, min_dur=0.18, max_seg=4.0),
    "fast":   dict(still_gap=0.16, end_gap=0.11, cooldown=0.06, min_dur=0.14, max_seg=3.5),
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
                if float(inter_wrist.min()) < 0.55 and both_dy < -0.06 and pos_med[1] > -0.20:
                    scores["help"] = min(0.90, 0.65 + abs(both_dy) * 0.8)

                # MORE: Both hands bringing fingertips together at midline
                if min_tip_d < 0.35 and start_tip_d > end_tip_d + 0.08 and pos_med[1] > -0.15:
                    scores["more"] = 0.84

                # WANT: Both hands open/clawed in neutral space pulling inward/down
                l_ext = self._finger_ext(lh)
                r_ext = self._finger_ext(rh)
                if (l_ext > 1.30).sum() >= 2 and (r_ext > 1.30).sum() >= 2:
                    if float(inter_wrist.mean()) > 0.30 and -0.10 <= pos_med[1] <= 0.80:
                        scores["want"] = 0.80

                # GOOD: One hand starts at mouth and moves down to the other hand
                if dist_mouth_start < 0.48 and dy > 0.08 and min_tip_d < 0.48:
                    scores["good"] = 0.82

        # ---- Single-Handed Daily Signs ----
        if not both_hands:
            # HELLO: Open hand near forehead/temple/eye level moving outward or waving
            if (is_open or n_ext >= 2) and (pos_start[1] < nose[1] + 0.22 or dist_nose < 0.52):
                if outward_dx > 0.04 or x_osc > 0.18:
                    scores["hello"] = 0.88

            # THANK-YOU: Open hand starting near chin/mouth and moving forward/downward
            if (is_open or n_ext >= 2) and dist_mouth_start < 0.48 and dy > 0.08 and pos_end[1] > mouth[1] + 0.05:
                scores["thank-you"] = 0.86

            # WATER: 'W' handshape (3 fingers up) or 2+ fingers tapping near chin/mouth
            if (is_w_shape or (ext[0] > 1.42 and ext[1] > 1.42)) and dist_mouth_med < 0.48:
                scores["water"] = 0.86

            # FEEL: Hand on chest moving UPWARD (middle finger or open palm brushing up chest)
            if (dist_chest < 0.52 or dist_tip_chest < 0.46) and dy < -0.06 and pos_med[1] > -0.20:
                scores["feel"] = 0.85

            # PLEASE: Open flat hand rubbing chest in a circle
            if is_open and (dist_chest < 0.48 or dist_tip_chest < 0.45) and path_len > 0.32:
                scores["please"] = 0.84

            # SORRY: Closed fist rubbing chest in a circle
            if is_fist and (dist_chest < 0.48 or dist_tip_chest < 0.45) and path_len > 0.30:
                scores["sorry"] = 0.84

            # I: Pinky extended OR index pointing inward at chest center with low motion
            if (dist_tip_chest < 0.42 or dist_chest < 0.45) and path_len < 0.48:
                if is_pinky_only or is_index_only or n_ext <= 2:
                    scores["i"] = 0.85
                elif is_open:
                    scores["my"] = 0.82

            # NEUTRAL SPACE SIGNS (away from face and chest center)
            in_neutral = (dist_nose > 0.38 and dist_mouth_med > 0.34 and dist_tip_chest > 0.28)
            if in_neutral:
                # YES: Fist nodding up and down in neutral space
                if is_fist and y_osc > 0.16 and y_osc >= x_osc * 0.8:
                    scores["yes"] = 0.82
                # BATHROOM: Fist shaking side-to-side in neutral space
                elif (is_fist or n_ext <= 1) and x_osc > 0.22 and x_osc > y_osc * 1.1:
                    scores["bathroom"] = 0.80
                # NO: Two fingers (index + middle) snapping toward thumb in neutral space
                elif (is_two_fingers or n_ext <= 2) and path_len < 0.65:
                    scores["no"] = 0.80
                # NEED: Index/hooked finger moving firmly DOWNWARD in neutral space
                elif (is_index_only or n_ext <= 2) and dy > 0.09:
                    scores["need"] = 0.80
                # WHERE: Index finger wagging side-to-side in neutral space
                elif (is_index_only or n_ext <= 2) and x_osc > 0.22 and x_osc > y_osc * 1.1:
                    scores["where"] = 0.82
                # YOU: Index finger pointing forward in neutral space with low motion
                elif (is_index_only or n_ext <= 2) and path_len < 0.45:
                    scores["you"] = 0.80

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


def decide(med, rf, daily, a, intent_engine=None, history=None):
    """
    Intelligent Multi-Model Fusion (Medical 50-sign + Roboflow Daily + Local Daily Kinematic)
    with optional Semantic Intent Boosting.
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
    raw_top = [(w, s, src) for w, (s, src) in ranked[:4]]
    if not raw_top:
        return (False, []) if intent_engine is None else (False, [], None)

    if intent_engine is not None and history is not None:
        return intent_engine.boost_candidates(
            raw_top,
            history,
            base_threshold=a.threshold,
            base_margin=a.margin
        )

    s2 = raw_top[1][1] if len(raw_top) > 1 else 0.0
    ok = (raw_top[0][1] >= a.threshold and raw_top[0][1] - s2 >= a.margin)
    return ok, raw_top[:3]


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
        if len(body) >= 6 and dur >= p["min_dur"] and shown >= 0.40:
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
    ap.add_argument("--threshold", type=float, default=0.42, help="base recognition threshold")
    ap.add_argument("--margin", type=float, default=0.06, help="confidence margin over 2nd candidate")
    ap.add_argument("--start_speed", type=float, default=0.45)
    ap.add_argument("--still_speed", type=float, default=0.22)
    ap.add_argument("--med_weight", type=float, default=1.0, help="raise to trust medical model more")
    ap.add_argument("--rf_weight", type=float, default=1.05, help="raise to trust Roboflow more")
    ap.add_argument("--daily_weight", type=float, default=1.0, help="raise to trust local daily ASL model more")
    ap.add_argument("--rf_conf", type=float, default=0.38)
    ap.add_argument("--rf_grace", type=float, default=0.20, help="max wait for Roboflow after a sign")
    ap.add_argument("--model_id", default="asl-avoix/1")
    ap.add_argument("--no_roboflow", action="store_true")
    ap.add_argument("--mirror_rf", action="store_true", help="send mirrored image to Roboflow (toggle: M)")
    ap.add_argument("--sentence_gap", type=float, default=1.4, help="seconds of silence that end a sentence")
    ap.add_argument("--context_weight", type=float, default=0.38, help="weight of semantic intent prediction")
    ap.add_argument("--mute", action="store_true")
    ap.add_argument("--rate", type=int, default=150, help="voice speed")
    ap.add_argument("--scale", type=float, default=1.35, help="window size multiplier")
    ap.add_argument("--mode", choices=["split", "doctor", "patient"], default="split", help="UI mode: split, doctor, or patient")
    ap.add_argument("--conditions", nargs="*", default=[], help="pre-existing patient conditions (e.g. asthma pregnancy)")
    ap.add_argument("--allergies", nargs="*", default=[], help="patient allergies (e.g. penicillin)")
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()

    med = MedicalModel(MODEL_PATH)
    daily = DailyASLModel()
    intent_engine = SmartIntentEngine(context_weight=a.context_weight)

    # Initialize Clinical AI Agents, Fingerspelling & Session Exporter
    face_analyzer = FaceMeshAnalyzer()
    triage_agent = TriageAgent()
    drug_agent = DrugSafetyAgent(patient_conditions=a.conditions, patient_allergies=a.allergies)
    diagnostic_agent = DiagnosticAgent()
    scribe_agent = ClinicalScribeAgent()
    doctor_agent = TwoWayDoctorAgent()
    fs_engine = FingerspellingEngine(commit_delay=1.5)
    session = ClinicalSession()

    ui_mode = a.mode
    doctor_menu_open = False
    toast_msg = None
    toast_until = 0.0
    last_pain_info = {"pain_score": 1, "facial_distress": False}
    last_triage = triage_agent.assess([], "", pain_score=1)
    last_diagnosis = diagnostic_agent.analyze([], "", pain_score=1)
    last_raw_top = []
    active_drug_alerts = []

    def on_doctor_card(card):
        nonlocal toast_msg, toast_until
        toast_msg = f"Sent to Patient: [{card.icon}] {card.headline}"
        toast_until = time.time() + 3.0

    console_worker = DoctorConsoleWorker(doctor_agent, session, on_message_callback=on_doctor_card)
    console_worker.start()

    voice_worker = DoctorVoiceWorker(doctor_agent, session, on_message_callback=on_doctor_card)
    voice_worker.start()
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
    if not cam.is_opened:
        print(f"\n[ERROR] Cannot open webcam at index {a.cam}!")
        if sys.platform == "darwin":
            print("  * On macOS: Enable Camera permission for Terminal / VS Code in:")
            print("    System Settings > Privacy & Security > Camera")
        print("  * If you have an external or alternate camera, try running:")
        print("    python asl_main.py --cam 1\n")
        return
    hol = make_holistic(a.complexity)
    smoother = LandmarkSmoother(alpha=0.65)

    mirror_rf = a.mirror_rf
    show_skel = True
    style_idx = 0
    words, log = [], []
    pending, unsure = None, None
    status, status_type, cand_txt = "Ready - sign a word", "ready", ""
    status_until, last_word_t, last_rf_sub = 0.0, 0.0, 0.0
    last_n, fps, prev_t = -1, 0.0, None
    sentence, alt_sentence = "", ""
    suggestions = intent_engine.predict_next_candidates([], top_k=3)

    def update_sentences():
        nonlocal sentence, alt_sentence, suggestions, last_triage, active_drug_alerts, last_diagnosis
        current_words = [x["w"] for x in words]
        vars_ = realise_variants(current_words)
        if not vars_:
            sentence, alt_sentence = "", ""
        else:
            sentence = vars_[style_idx % len(vars_)]
            alt_sentence = vars_[(style_idx + 1) % len(vars_)] if len(vars_) > 1 else ""
        suggestions = intent_engine.predict_next_candidates(current_words, top_k=3)

        # Update clinical triage
        last_triage = triage_agent.assess(
            recognized_signs=current_words,
            interpreted_text=sentence,
            pain_score=last_pain_info.get("pain_score", 1),
            facial_distress=last_pain_info.get("facial_distress", False),
        )
        session.record_triage(last_triage)

        # Update drug safety
        active_drug_alerts = drug_agent.evaluate(sentence + " " + " ".join(current_words))
        session.set_drug_alerts(active_drug_alerts)

        # Update AI Clinical Diagnostic Reasoning (DDx)
        last_diagnosis = diagnostic_agent.analyze(
            signs=current_words,
            interpreted_sentence=sentence,
            pain_score=last_pain_info.get("pain_score", 1),
            triage_category=last_triage.category,
        )
        session.set_diagnosis(last_diagnosis)

    def push(word, src, score):
        nonlocal last_word_t
        words.append(dict(w=word, src=src, s=score))
        session.record_sign(word, score)
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
            session.record_sentence(
                sentence=text,
                pain_score=last_pain_info.get("pain_score", 1),
                facial_distress=last_pain_info.get("facial_distress", False),
            )
            scribe_agent.record_utterance(
                raw_signs=[x["w"] for x in words],
                interpreted=text,
                pain_score=last_pain_info.get("pain_score", 1),
                distress=last_pain_info.get("facial_distress", False),
                question_marker=last_pain_info.get("question_marker"),
            )
            soap_note = scribe_agent.generate_soap_note(last_triage, active_drug_alerts)
            session.set_soap_note(soap_note)
            words.clear()
            sentence, alt_sentence = "", ""
            style_idx = 0
            update_sentences()

    WINDOW_TITLE = "ASLMed - Multi-Agent Clinical AI Co-Pilot & Interpreter"
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, 1280, 920)

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
        raw_lm, face_pts = landmarks_from_bgr(small, hol, return_face=True)
        raw_lm = smoother.update(raw_lm)
        lm = clean_hands(raw_lm[None], need_shoulders=True)[0]
        last_pain_info = face_analyzer.update(face_pts)

        # 3D Hand Fingerspelling Check (when not active in whole-body sign)
        if not seg.active and lm is not None:
            detected_letter = None
            for base in (46, 25):
                hand_pts = lm[base:base + 21]
                if not np.isnan(hand_pts[0, 0]):
                    detected_letter = fs_engine.classify_hand_shape(hand_pts)
                    if detected_letter:
                        break
            if detected_letter:
                if detected_letter == fs_engine.last_letter:
                    fs_engine.letter_hold_count += 1
                    if fs_engine.letter_hold_count == 4:
                        fs_engine.push_letter(detected_letter, now)
                        status, status_type = f"Spelling: {fs_engine.get_spelled_word()}", "signing"
                        status_until = now + 1.0
                else:
                    fs_engine.last_letter = detected_letter
                    fs_engine.letter_hold_count = 1
            else:
                fs_engine.last_letter = None
                fs_engine.letter_hold_count = 0

        # Auto-commit fingerspelled words into signed glosses
        spelled_word = fs_engine.check_auto_commit(now)
        if spelled_word:
            push(spelled_word, "fingerspelling", 0.99)
            status, status_type = f"Spelled: {spelled_word.upper()}", "recognized"
            status_until = now + 1.5

        ev, payload = seg.update(lm, now)

        if ev == "start":
            status, status_type, cand_txt = "Signing...", "signing", ""
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
            status, status_type = "Analyzing...", "signing"
        elif ev == "cancel":
            status, status_type = "Ready - sign a word", "ready"

        if pending:
            rf_ok = rf is not None and not rf.error
            if (not rf_ok) or rf.last_done_t >= pending["last_sub"] or now - pending["created"] > a.rf_grace:
                votes = rf_votes(rf.between(pending["t0"] - 0.05, pending["t1"] + 0.05)) if rf_ok else []
                history = [x["w"] for x in words]
                ok, top, hint = decide(pending["med"], votes, pending["daily"], a, intent_engine=intent_engine, history=history)
                cand_txt = "   ".join(f"{w} {s * 100:.0f}% ({src})" for w, s, src in top)
                last_raw_top = top
                if ok:
                    push(top[0][0], top[0][2], top[0][1])
                    is_inferred = "intent" in top[0][2]
                    status = f"Recognized: {top[0][0].upper()}"
                    status_type = "inferred" if is_inferred else "recognized"
                    unsure = None
                    status_until = now + 1.2
                elif hint:
                    unsure = top[0] if top else None
                    status = f"Intent likely: {hint.upper()} (Press A)"
                    status_type = "inferred"
                    status_until = now + 1.8
                else:
                    unsure = top[0] if top else None
                    status = f"Not sure ({top[0][0]}?) - Press A" if top else "Not sure - sign again"
                    status_type = "unsure"
                    status_until = now + 1.8
                pending = None

        if status_until and now > status_until and not seg.active and not pending:
            status, status_type, cand_txt, status_until = "Ready - sign a word", "ready", "", 0.0
        elif not seg.active and not pending and now < seg.block_until and not status_until:
            status, status_type = "...", "ready"

        # A pause after the last word ends and speaks the sentence
        if words and not seg.active and not pending and now - last_word_t > a.sentence_gap:
            finish(True)

        active_toast = toast_msg if (toast_until and now < toast_until) else None
        canvas = render_modern_dashboard(
            frame=frame,
            lm=lm if show_skel else None,
            show_skel=show_skel,
            now=now,
            status=status,
            status_type=status_type,
            cand_txt=cand_txt,
            words=words,
            sentence=sentence,
            alt_sentence=alt_sentence,
            suggestions=suggestions,
            log=log,
            speaker_active=speaker.enabled and speaker.ok,
            fps=fps,
            speed=seg.speed,
            progress=seg.still_progress(now),
            scale=a.scale,
            triage=last_triage,
            pain_info=last_pain_info,
            drug_alerts=active_drug_alerts,
            active_card=doctor_agent.active_card,
            ui_mode=ui_mode,
            doctor_reply_menu=doctor_menu_open,
            toast_message=active_toast,
            spelled_text=fs_engine.get_spelled_word(),
            diagnosis=last_diagnosis,
            neural_cands=last_raw_top,
        )
        cv2.imshow(WINDOW_TITLE, canvas)
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord("q")):
            break
        elif k == ord("d"):  # Cycle Split -> Doctor -> Patient -> Split
            if ui_mode == "split":
                ui_mode = "doctor"
            elif ui_mode == "doctor":
                ui_mode = "patient"
            else:
                ui_mode = "split"
            toast_msg = f"Mode: {ui_mode.upper()}"
            toast_until = now + 1.5
        elif k == ord("t"):  # Toggle Doctor Quick-Reply Menu
            doctor_menu_open = not doctor_menu_open
        elif k == ord("e"):  # Export Clinical Encounter Documentation (.md and .json)
            md_path = session.export_markdown()
            session.export_json()
            toast_msg = f"Encounter Saved: {md_path.name} & .json"
            toast_until = now + 3.5
        elif doctor_menu_open:
            if ord("1") <= k <= ord("9"):
                card_id = k - ord("0")
                card = doctor_agent.select_card(card_id)
                if card:
                    session.record_doctor_reply(card.headline, card.card_id)
                    toast_msg = f"Sent to Patient: [{card.icon}] {card.headline}"
                    toast_until = now + 3.0
                doctor_menu_open = False
            elif k == ord("0"):
                card = doctor_agent.select_card(10)
                if card:
                    session.record_doctor_reply(card.headline, card.card_id)
                    toast_msg = f"Sent to Patient: [{card.icon}] {card.headline}"
                    toast_until = now + 3.0
                doctor_menu_open = False
            elif k == ord("/"):
                doctor_menu_open = False
                toast_msg = "Doctor: Type message in terminal & press ENTER"
                toast_until = now + 3.5
            elif k in (27, ord("t")):
                doctor_menu_open = False
        elif k == ord("/"):
            toast_msg = "Doctor: Type message in terminal & press ENTER"
            toast_until = now + 3.5
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
            fs_engine.clear()
            update_sentences()
        elif k == ord("r") and log:
            speaker.say(log[-1])
        elif k == ord("a") and unsure:
            push(unsure[0], unsure[2], unsure[1])
            unsure = None
        elif k in (ord("1"), ord("2"), ord("3")) and suggestions:
            idx = k - ord("1")
            if idx < len(suggestions):
                push(suggestions[idx][0], "quick_pick", 0.95)
                status, status_type = f"Quick pick: {suggestions[idx][0].upper()}", "inferred"
                status_until = now + 1.2
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
    console_worker.alive = False
    voice_worker.alive = False
    hol.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
