import types
import numpy as np
import pytest

from asl_main import (
    DailyASLModel,
    PACE,
    Segmenter,
    decide,
    rf_votes,
    wrap,
)


class DummyArgs:
    threshold = 0.48
    margin = 0.10
    med_weight = 1.0
    rf_weight = 1.05
    daily_weight = 1.0


class TestASLMainLogic:
    def test_wrap_text(self):
        text = "Hello doctor I feel dizzy and I need water please"
        lines = wrap(text, scale=0.5, thick=1, max_w=200)
        assert len(lines) >= 2
        assert " ".join(lines) == text

    def test_decide_medical_confident(self):
        a = DummyArgs()
        med = [("headache", 0.85), ("fever", 0.10), ("flu", 0.05)]
        rf = [("want", 0.40)]
        daily = [("help", 0.30)]

        ok, top = decide(med, rf, daily, a)
        assert ok is True
        assert top[0][0] == "headache"
        assert top[0][2] == "med"

    def test_decide_daily_boosted_by_roboflow(self):
        a = DummyArgs()
        med = [("headache", 0.40), ("fever", 0.35)]
        rf = [("want", 0.70)]
        daily = [("want", 0.75)]

        ok, top = decide(med, rf, daily, a)
        assert ok is True
        assert top[0][0] == "want"
        assert top[0][2] == "rf+daily"
        assert top[0][1] > 0.80

    def test_decide_margin_and_threshold(self):
        a = DummyArgs()
        # Top score is below threshold
        med = [("headache", 0.30), ("fever", 0.28)]
        rf = []
        daily = []
        ok, top = decide(med, rf, daily, a)
        assert ok is False

        # Top score high, but margin too close
        med2 = [("headache", 0.70), ("fever", 0.68)]
        ok2, top2 = decide(med2, rf, daily, a)
        assert ok2 is False

    def test_rf_votes(self):
        # Frame 1: apple (0.8), Frame 2: apple (0.9), Frame 3: want (0.5)
        frames = [
            [("apple", 0.8)],
            [("apple", 0.9)],
            [("want", 0.5)],
        ]
        votes = rf_votes(frames)
        assert len(votes) >= 1
        assert votes[0][0] == "apple"
        assert votes[0][1] > 0.6

    def test_segmenter_transitions(self):
        seg = Segmenter(PACE["normal"], start_speed=0.48, still_speed=0.25)
        now = 100.0

        # No hands
        no_hand = np.full((67, 3), np.nan, dtype=np.float32)
        ev, payload = seg.update(no_hand, now)
        assert ev is None
        assert not seg.active

        # Hand enters -> hand appearance triggers initial speed burst -> "start"
        hand_lm1 = np.zeros((67, 3), dtype=np.float32)
        hand_lm1[11:13, :2] = [[-0.2, 0.0], [0.2, 0.0]]
        hand_lm1[25:46, :2] = 0.0

        ev_start, _ = seg.update(hand_lm1, now + 0.03)
        assert ev_start == "start"
        assert seg.active

        # Hand stays active while moving
        hand_lm2 = hand_lm1.copy()
        hand_lm2[25:46, :2] = 0.5
        ev_active, _ = seg.update(hand_lm2, now + 0.06)
        assert ev_active is None
        assert seg.active

    def test_daily_asl_finger_ext(self):
        # Test finger extension ratio computation on open hand
        hand = np.zeros((21, 3), dtype=np.float32)
        # Wrist at 0, MCP at 0.1, Tip at 0.3 (ratio = 3.0 > 1.5)
        for mcp, tip in [(5, 8), (9, 12), (13, 16), (17, 20)]:
            hand[mcp, :2] = [0.0, 0.1]
            hand[tip, :2] = [0.0, 0.3]

        ext = DailyASLModel._finger_ext(hand[None])
        assert len(ext) == 4
        assert np.all(ext > 1.5)

    def test_intent_engine_fuzzy_acceptance(self):
        from src.asl_intent import SmartIntentEngine
        intent_engine = SmartIntentEngine()
        a = DummyArgs()

        # Simulate imperfect sign for 'pill' (score only 0.34, normally rejected)
        # but following 'want' in conversation history
        raw_med = [("pill", 0.34), ("fever", 0.30)]
        ok, top, hint = decide(raw_med, [], [], a, intent_engine=intent_engine, history=["i", "want"])
        assert ok is True, "Imperfect sign for 'pill' should be accepted due to 'want' context"
        assert top[0][0] == "pill"
        assert "intent" in top[0][2]

    def test_intent_engine_suggestions(self):
        from src.asl_intent import SmartIntentEngine
        engine = SmartIntentEngine()
        sugs = engine.predict_next_candidates(["headache"], top_k=3)
        assert len(sugs) == 3
        words = [w for w, _ in sugs]
        assert any(w in ["want", "need", "pill", "water", "medicine"] for w in words)

    def test_modern_dashboard_render(self):
        from src.asl_ui import render_modern_dashboard
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        canvas = render_modern_dashboard(
            frame=dummy_frame,
            lm=None,
            show_skel=False,
            now=100.0,
            status="READY",
            status_type="ready",
            cand_txt="test",
            words=[{"w": "i", "src": "daily"}],
            sentence="I feel good.",
            alt_sentence="",
            suggestions=[("want", 0.3)],
            log=[],
            speaker_active=True,
            fps=30.0,
            speed=0.1,
            progress=0.0,
            scale=1.0
        )
        assert canvas.shape == (780, 640, 3)

        # Test Patient Mode & Doctor Reply Overlay
        from src.asl_clinical_agents import TriageAgent, TwoWayDoctorAgent, DrugSafetyAgent
        triage = TriageAgent().assess(["chest", "pain"], "I have chest pain", pain_score=8, facial_distress=True)
        doc = TwoWayDoctorAgent()
        card = doc.select_card(1)
        drug_agent = DrugSafetyAgent(["asthma"])
        alerts = drug_agent.evaluate("ibuprofen")

        canvas_patient = render_modern_dashboard(
            frame=dummy_frame,
            lm=None,
            show_skel=False,
            now=100.0,
            status="READY",
            status_type="ready",
            cand_txt="test",
            words=[{"w": "chest"}, {"w": "pain"}],
            sentence="I have severe chest pain",
            alt_sentence="",
            suggestions=[],
            log=[],
            speaker_active=True,
            fps=30.0,
            speed=0.0,
            progress=0.0,
            scale=1.0,
            triage=triage,
            pain_info={"pain_score": 8, "facial_distress": True},
            drug_alerts=alerts,
            active_card=card,
            ui_mode="patient",
            doctor_reply_menu=True,
            toast_message="Encounter saved successfully",
        )
        assert canvas_patient.shape == (780, 640, 3)

    def test_draw_neural_bar(self):
        from src.asl_ui import draw_neural_bar
        dummy = np.zeros((100, 300, 3), dtype=np.uint8)
        draw_neural_bar(dummy, "chest pain", 0.94, "BiGRU", 10, 20, 260)
        assert dummy.sum() > 0

