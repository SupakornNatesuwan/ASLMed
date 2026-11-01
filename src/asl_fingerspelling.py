"""
ASLMed Fingerspelling & Manual Alphabet Engine
Enables deaf patients to spell out specific medication names, dosages, and proper nouns (A-Z)
using 3D hand landmark kinematics and letter buffer accumulation.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np


class FingerspellingEngine:
    """
    Recognizes ASL manual alphabet letters and accumulates them into clinical words.
    Includes known medication dictionary matching and auto-commit logic.
    """

    KNOWN_CLINICAL_WORDS = {
        "aspirin": "aspirin",
        "tylenol": "tylenol",
        "panadol": "panadol",
        "ibuprofen": "ibuprofen",
        "paracetamol": "paracetamol",
        "amoxicillin": "amoxicillin",
        "penicillin": "penicillin",
        "insulin": "insulin",
        "morphine": "morphine",
        "codeine": "codeine",
        "asthma": "asthma",
        "allergy": "allergy",
        "covid": "covid",
        "cancer": "cancer",
        "fever": "fever",
        "pain": "pain",
    }

    def __init__(self, commit_delay: float = 1.6):
        self.commit_delay = commit_delay
        self.buffer: List[str] = []
        self.last_letter_t: float = 0.0
        self.last_letter: Optional[str] = None
        self.letter_hold_count: int = 0

    def push_letter(self, letter: str, now: float = 0.0) -> None:
        c = letter.upper().strip()
        if not c or len(c) != 1 or not c.isalpha():
            return
        self.buffer.append(c)
        self.last_letter_t = now

    def pop_letter(self) -> Optional[str]:
        if self.buffer:
            return self.buffer.pop()
        return None

    def clear(self) -> None:
        self.buffer.clear()
        self.last_letter = None
        self.letter_hold_count = 0

    def get_spelled_word(self) -> str:
        return "".join(self.buffer)

    def check_auto_commit(self, now: float) -> Optional[str]:
        """If user paused after spelling letters, commits the word automatically."""
        if self.buffer and self.last_letter_t > 0 and (now - self.last_letter_t > self.commit_delay):
            return self.commit()
        return None

    def commit(self) -> Optional[str]:
        if not self.buffer:
            return None
        raw_word = "".join(self.buffer).lower()
        self.buffer.clear()
        return self.KNOWN_CLINICAL_WORDS.get(raw_word, raw_word)

    @staticmethod
    def classify_hand_shape(hand_pts: np.ndarray) -> Optional[str]:
        """
        Classifies 3D hand landmarks (21, 3) into key ASL manual alphabet letters.
        """
        if hand_pts is None or len(hand_pts) < 21 or np.isnan(hand_pts[0, 0]):
            return None

        # Wrist = 0
        wrist = hand_pts[0, :2]
        # MCP (bases): index=5, middle=9, ring=13, pinky=17
        # PIP: 6, 10, 14, 18
        # TIP: thumb=4, index=8, middle=12, ring=16, pinky=20
        tips = [4, 8, 12, 16, 20]
        mcps = [2, 5, 9, 13, 17]

        # Finger extension check: tip is farther from wrist than MCP
        dist_tips = np.linalg.norm(hand_pts[tips, :2] - wrist, axis=1)
        dist_mcps = np.linalg.norm(hand_pts[mcps, :2] - wrist, axis=1)
        is_extended = dist_tips > (dist_mcps * 1.15)

        thumb_ext, index_ext, mid_ext, ring_ext, pinky_ext = is_extended

        # Letter "B": 4 fingers extended upright together, thumb curled
        if not thumb_ext and index_ext and mid_ext and ring_ext and pinky_ext:
            return "B"

        # Letter "A": all 4 fingers closed into fist, thumb upright alongside
        if not index_ext and not mid_ext and not ring_ext and not pinky_ext and thumb_ext:
            return "A"

        # Letter "L": Index extended up, Thumb extended out, other 3 closed
        if thumb_ext and index_ext and not mid_ext and not ring_ext and not pinky_ext:
            # Check angle between thumb and index
            vec_idx = hand_pts[8, :2] - hand_pts[5, :2]
            vec_thb = hand_pts[4, :2] - hand_pts[2, :2]
            norm_idx = np.linalg.norm(vec_idx)
            norm_thb = np.linalg.norm(vec_thb)
            if norm_idx > 1e-3 and norm_thb > 1e-3:
                dot = np.dot(vec_idx, vec_thb) / (norm_idx * norm_thb)
                if abs(dot) < 0.65:  # Approximately perpendicular
                    return "L"

        # Letter "V": Index and middle extended, ring and pinky closed
        if index_ext and mid_ext and not ring_ext and not pinky_ext and not thumb_ext:
            return "V"

        # Letter "W": Index, middle, and ring extended, pinky closed
        if index_ext and mid_ext and ring_ext and not pinky_ext and not thumb_ext:
            return "W"

        # Letter "I": Pinky extended alone, other 3 fingers closed
        if pinky_ext and not index_ext and not mid_ext and not ring_ext and not thumb_ext:
            return "I"

        # Letter "Y": Thumb and pinky extended, middle 3 closed
        if thumb_ext and pinky_ext and not index_ext and not mid_ext and not ring_ext:
            return "Y"

        # Letter "F": Index and thumb tips touching (pinched), other 3 fingers extended
        if mid_ext and ring_ext and pinky_ext:
            pinch_dist = np.linalg.norm(hand_pts[4, :2] - hand_pts[8, :2])
            hand_scale = max(np.linalg.norm(hand_pts[0, :2] - hand_pts[9, :2]), 0.05)
            if pinch_dist / hand_scale < 0.28:
                return "F"

        # Letter "D": Index extended alone, thumb and other fingers touching
        if index_ext and not mid_ext and not ring_ext and not pinky_ext:
            return "D"

        return None
