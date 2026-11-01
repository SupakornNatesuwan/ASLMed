"""
ASLMed Smart Intent & Context Prediction Engine
Enables contextual prior inference, fuzzy/partial sign recognition, and next-sign prediction.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# Transition co-occurrence weights based on clinical & daily conversation patterns
CONTEXT_TRANSITIONS: Dict[str, Dict[str, float]] = {
    # Pronouns lead into verbs, body parts, or immediate needs
    "i": {
        "feel": 0.35, "want": 0.30, "need": 0.25, "have": 0.15,
        "headache": 0.15, "fever": 0.12, "cough": 0.10, "pain": 0.15,
        "dizziness": 0.12, "can't breathe": 0.10, "tired": 0.15, "weak": 0.12,
    },
    "my": {
        "skin": 0.25, "head": 0.25, "stomach": 0.25, "chest": 0.20,
        "throat": 0.20, "neck": 0.18, "shoulder": 0.18, "bones": 0.15,
        "heart": 0.15, "blood": 0.12, "blood pressure": 0.12,
    },
    "you": {
        "doctor": 0.35, "nurse": 0.30, "help": 0.25, "have": 0.20,
        "can": 0.20, "good": 0.15,
    },
    # Desires lead into treatments, objects, people, or vitals
    "want": {
        "pill": 0.35, "medicine": 0.30, "water": 0.25, "cream": 0.20,
        "doctor": 0.20, "nurse": 0.18, "x-ray": 0.15, "bathroom": 0.20,
        "wheelchair": 0.15, "ice": 0.12, "more": 0.15, "food": 0.12,
    },
    "need": {
        "pill": 0.30, "medicine": 0.30, "water": 0.25, "doctor": 0.28,
        "nurse": 0.22, "help": 0.30, "wheelchair": 0.25, "bathroom": 0.25,
        "x-ray": 0.18, "call ambulance": 0.20, "surgery": 0.15, "oxygen": 0.15,
    },
    "feel": {
        "dizziness": 0.30, "dizzy": 0.30, "tired": 0.30, "weak": 0.25,
        "sick": 0.25, "pain": 0.25, "itchy": 0.20, "swollen": 0.20,
        "hot": 0.18, "cold": 0.18, "vomit": 0.20, "headache": 0.20,
        "good": 0.22,
    },
    # Body parts lead into symptoms, sensations, or clinical requests
    "head": {
        "pain": 0.40, "headache": 0.35, "dizziness": 0.25, "x-ray": 0.15,
    },
    "stomach": {
        "pain": 0.40, "stomach ache": 0.40, "vomit": 0.25, "medicine": 0.20, "pill": 0.15,
    },
    "chest": {
        "pain": 0.45, "chest pain": 0.45, "heart attack": 0.30, "can't breathe": 0.25, "x-ray": 0.15,
    },
    "throat": {
        "pain": 0.40, "sore throat": 0.45, "cough": 0.30, "water": 0.25, "medicine": 0.20,
    },
    "neck": {
        "pain": 0.40, "shoulder": 0.30, "bones": 0.20, "x-ray": 0.20,
    },
    "shoulder": {
        "pain": 0.40, "neck": 0.30, "arm": 0.20, "x-ray": 0.20,
    },
    "skin": {
        "itchy": 0.45, "swollen": 0.35, "cream": 0.35, "allergic reaction": 0.25,
    },
    # Symptoms lead into needs/treatments/desires
    "headache": {
        "want": 0.35, "need": 0.30, "pill": 0.35, "medicine": 0.30, "water": 0.20,
        "tired": 0.20, "dizziness": 0.18, "doctor": 0.15,
    },
    "stomach ache": {
        "want": 0.30, "need": 0.30, "pill": 0.30, "medicine": 0.30, "vomit": 0.25,
        "bathroom": 0.25, "water": 0.18, "doctor": 0.18,
    },
    "chest pain": {
        "doctor": 0.35, "need": 0.35, "call ambulance": 0.30, "emergency": 0.30,
        "can't breathe": 0.25, "heart attack": 0.25,
    },
    "fever": {
        "cough": 0.35, "sore throat": 0.30, "flu": 0.30, "tired": 0.25,
        "want": 0.25, "pill": 0.25, "water": 0.25, "medicine": 0.25,
    },
    "cough": {
        "fever": 0.30, "sore throat": 0.35, "water": 0.25, "medicine": 0.25,
        "pill": 0.20, "can't breathe": 0.18,
    },
    "sore throat": {
        "cough": 0.35, "fever": 0.25, "water": 0.30, "medicine": 0.25, "pill": 0.20,
    },
    "dizziness": {
        "water": 0.30, "need": 0.30, "tired": 0.25, "weak": 0.25,
        "wheelchair": 0.20, "doctor": 0.20,
    },
    "can't breathe": {
        "help": 0.40, "emergency": 0.40, "call ambulance": 0.35, "oxygen": 0.30,
        "doctor": 0.30, "need": 0.30,
    },
    "bleeding": {
        "help": 0.40, "doctor": 0.35, "nurse": 0.30, "emergency": 0.30,
        "cream": 0.20, "need": 0.25,
    },
    # Greetings & Politeness
    "hello": {
        "doctor": 0.45, "nurse": 0.35, "i": 0.25, "help": 0.20, "how": 0.15,
    },
    "hi": {
        "doctor": 0.45, "nurse": 0.35, "i": 0.25, "help": 0.20,
    },
    "please": {
        "water": 0.35, "help": 0.35, "pill": 0.30, "doctor": 0.25,
        "bathroom": 0.25, "wheelchair": 0.20,
    },
}

DEFAULT_TOP_SIGNS = ["i", "feel", "want", "need", "hello", "doctor", "headache", "water", "help"]


class SmartIntentEngine:
    """
    Infers next intended signs and applies dynamic context boosting
    so imperfect, partial, or rushed signs are recognized accurately.
    """

    def __init__(self, context_weight: float = 0.38):
        self.context_weight = context_weight

    def predict_next_candidates(self, history: List[str], top_k: int = 3) -> List[Tuple[str, float]]:
        """Predict most likely next signs based on signed sentence history."""
        if not history:
            return [(s, 0.20) for s in DEFAULT_TOP_SIGNS[:top_k]]

        scores: Dict[str, float] = defaultdict(float)
        
        # Consider last 2 words with exponential decay
        weights = [0.4, 0.7] if len(history) >= 2 else [0.8]
        recent = history[-2:]

        for w, weight in zip(recent, weights):
            w_norm = str(w).strip().lower()
            if w_norm in CONTEXT_TRANSITIONS:
                for nxt, prior in CONTEXT_TRANSITIONS[w_norm].items():
                    # Avoid proposing exact same word immediately unless repetitively appropriate
                    if nxt != w_norm:
                        scores[nxt] += prior * weight

        if not scores:
            return [(s, 0.15) for s in DEFAULT_TOP_SIGNS[:top_k] if s not in history[-1:]]

        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        return ranked[:top_k]

    def boost_candidates(
        self,
        raw_candidates: List[Tuple[str, float, str]],
        history: List[str],
        base_threshold: float = 0.46,
        base_margin: float = 0.08
    ) -> Tuple[bool, List[Tuple[str, float, str]], Optional[str]]:
        """
        Fuse raw model scores with contextual priors.
        Allows imperfect / partial gestures to be accepted when contextually coherent.
        """
        if not raw_candidates:
            return False, [], None

        next_priors = dict(self.predict_next_candidates(history, top_k=8))
        fused = []

        for word, score, src in raw_candidates:
            prior = next_priors.get(word, 0.0)
            # Fused score combines raw model confidence with contextual intent prior
            if prior > 0:
                boost = prior * 0.45
                boosted_score = min(0.99, score + boost)
                fused_src = f"{src}+intent" if prior >= 0.18 else src
            else:
                boosted_score = score
                fused_src = src

            fused.append((word, float(boosted_score), fused_src))

        # Re-rank based on fused context-boosted score
        fused.sort(key=lambda x: -x[1])
        top = fused[:3]
        best_word, best_score, best_src = top[0]
        second_score = top[1][1] if len(top) > 1 else 0.0

        # Dynamic threshold softening:
        # If context strongly predicts this sign (prior >= 0.15), lower required threshold
        top_prior = next_priors.get(best_word, 0.0)
        effective_threshold = max(0.28, base_threshold - (top_prior * 0.35))
        effective_margin = max(0.04, base_margin - (top_prior * 0.10))

        is_accepted = (best_score >= effective_threshold and (best_score - second_score) >= effective_margin)
        intent_hint = best_word if (top_prior >= 0.18 and not is_accepted) else None

        return is_accepted, top, intent_hint
