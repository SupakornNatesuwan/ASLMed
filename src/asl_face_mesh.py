"""
ASLMed Facial Expression, Pain Scale & Non-Manual Marker Analyzer
Analyzes MediaPipe 468 face landmarks to extract:
1. Pain Scale (1-10 Grimace / Discomfort Index)
2. Non-Manual Grammatical Markers (Yes/No Questions, Wh- Questions, Intensity)
3. Blink & Eye Squint Dynamics
"""

from typing import Dict, Optional, Tuple
import numpy as np


class FaceMeshAnalyzer:
    """
    Real-time analyzer for non-manual facial grammatical markers and clinical pain severity.
    """

    def __init__(self, alpha: float = 0.70):
        self.alpha = alpha
        self.smoothed_pain: float = 1.0
        self.smoothed_brow_h: float = 0.28
        self.smoothed_brow_d: float = 0.26
        self.smoothed_eye_o: float = 0.08
        self.has_face: bool = False

    def update(self, face_pts: Optional[np.ndarray]) -> Dict:
        """
        Processes normalized 3D face landmarks (468, 3).
        Returns clinical pain score (1-10), question markers, and distress flags.
        """
        if face_pts is None or len(face_pts) < 468 or np.isnan(face_pts[1, 0]):
            self.has_face = False
            return {
                "has_face": False,
                "pain_score": int(np.round(self.smoothed_pain)),
                "pain_label": self._pain_label(int(np.round(self.smoothed_pain))),
                "brow_raised": False,
                "brow_furrowed": False,
                "question_marker": None,
                "facial_distress": False,
                "brow_tension": 0.0,
                "squint_ratio": 0.0,
                "mouth_grimace": 0.0,
            }

        self.has_face = True

        # Reference scale: Inter-canthal width between inner eye corners (33 and 263)
        ref_scale = max(float(np.linalg.norm(face_pts[33, :2] - face_pts[263, :2])), 0.04)

        # 1. Inner brow distance (66 and 296) -> narrows during pain, frowning, Wh-questions
        raw_brow_d = float(np.linalg.norm(face_pts[66, :2] - face_pts[296, :2])) / ref_scale

        # 2. Brow to eye height -> increases for Yes/No questions; decreases for frowning/pain
        eye_mid_l = (face_pts[159, :2] + face_pts[145, :2]) / 2.0
        eye_mid_r = (face_pts[386, :2] + face_pts[374, :2]) / 2.0
        raw_brow_h = float(
            (np.linalg.norm(face_pts[66, :2] - eye_mid_l) + np.linalg.norm(face_pts[296, :2] - eye_mid_r)) / 2.0
        ) / ref_scale

        # 3. Eye openness (eyelid separation) -> squints during acute pain
        open_l = float(np.linalg.norm(face_pts[159, :2] - face_pts[145, :2])) / ref_scale
        open_r = float(np.linalg.norm(face_pts[386, :2] - face_pts[374, :2])) / ref_scale
        raw_eye_o = (open_l + open_r) / 2.0

        # 4. Mouth stretch & corner retraction (61 and 291)
        mouth_w = float(np.linalg.norm(face_pts[61, :2] - face_pts[291, :2])) / ref_scale
        nasolabial_pull = float(np.linalg.norm(face_pts[13, :2] - face_pts[1, :2])) / ref_scale

        # Apply EMA smoothing to eliminate webcam landmark jitter
        a = self.alpha
        self.smoothed_brow_d = a * raw_brow_d + (1.0 - a) * self.smoothed_brow_d
        self.smoothed_brow_h = a * raw_brow_h + (1.0 - a) * self.smoothed_brow_h
        self.smoothed_eye_o = a * raw_eye_o + (1.0 - a) * self.smoothed_eye_o

        # Grammatical ASL Question Detection
        # - Eyebrow Raise (Frontalis) -> Yes/No questions or topicalization
        # - Eyebrow Furrow (Corrugator) -> Wh- questions (who, what, where) or doubt
        brow_raised = self.smoothed_brow_h > 0.33
        brow_furrowed = self.smoothed_brow_d < 0.22 or (self.smoothed_brow_h < 0.21 and self.smoothed_brow_d < 0.25)

        q_marker = None
        if brow_raised and not brow_furrowed:
            q_marker = "yes_no"
        elif brow_furrowed and not brow_raised:
            q_marker = "wh_question"

        # Pain Grimace Scale (1 to 10):
        # Combines inner brow furrow + eye tightening (squint) + mouth grimace
        furrow_pain = max(0.0, (0.27 - self.smoothed_brow_d) * 16.0)
        squint_pain = max(0.0, (0.075 - self.smoothed_eye_o) * 22.0)
        mouth_pain = max(0.0, (mouth_w - 0.46) * 7.0)

        raw_pain_score = 1.0 + (furrow_pain * 0.45 + squint_pain * 0.35 + mouth_pain * 0.20) * 8.5
        raw_pain_score = float(np.clip(raw_pain_score, 1.0, 10.0))

        self.smoothed_pain = 0.30 * raw_pain_score + 0.70 * self.smoothed_pain
        final_pain_int = int(np.clip(np.round(self.smoothed_pain), 1, 10))

        facial_distress = final_pain_int >= 7 or (brow_furrowed and self.smoothed_eye_o < 0.06)

        return {
            "has_face": True,
            "pain_score": final_pain_int,
            "pain_label": self._pain_label(final_pain_int),
            "brow_raised": brow_raised,
            "brow_furrowed": brow_furrowed,
            "question_marker": q_marker,
            "facial_distress": facial_distress,
            "brow_tension": round(float(np.clip(furrow_pain, 0.0, 1.0)), 2),
            "squint_ratio": round(float(np.clip(squint_pain, 0.0, 1.0)), 2),
            "mouth_grimace": round(float(np.clip(mouth_pain, 0.0, 1.0)), 2),
        }

    @staticmethod
    def _pain_label(score: int) -> str:
        if score <= 2:
            return f"Mild ({score}/10)"
        elif score <= 4:
            return f"Noticeable ({score}/10)"
        elif score <= 6:
            return f"Moderate ({score}/10)"
        elif score <= 8:
            return f"Severe ({score}/10)"
        return f"Critical ({score}/10)"
