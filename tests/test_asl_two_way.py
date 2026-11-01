"""
Unit Tests for ASLMed Two-Way Doctor-Patient Communication Suite:
1. TwoWayDoctorAgent (Thai & English intent matching, 10 clinical cards, dynamic custom cards)
2. FingerspellingEngine (3D handshape classification, letter buffer, auto-commit, drug matching)
3. Split Clinical Dashboard Rendering (Dual-Pane: Patient on Left, Doctor on Right)
4. Worker Thread Life-cycle (DoctorConsoleWorker & DoctorVoiceWorker)
"""

import numpy as np
import pytest

from src.asl_clinical_agents import (
    TwoWayDoctorAgent,
    DoctorVisualCard,
    DoctorConsoleWorker,
    DoctorVoiceWorker,
    TriageAgent,
    DiagnosticAgent,
)
from src.asl_fingerspelling import FingerspellingEngine
from src.asl_session_exporter import ClinicalSession
from src.asl_ui import render_modern_dashboard


class TestTwoWayDoctorAgentAdvanced:
    def setup_method(self):
        self.doc = TwoWayDoctorAgent()

    def test_preset_10_cards_available(self):
        cards = self.doc.get_card_list()
        assert len(cards) >= 10
        # Card 9: Fasting status
        card9 = self.doc.select_card(9)
        assert card9.icon == "FST"
        assert "Fasting" in card9.headline

        # Card 10: Drug Allergy
        card10 = self.doc.select_card(10)
        assert card10.icon == "ALG"
        assert "Allergies" in card10.headline

    def test_clinical_intent_parsing_comprehensive(self):
        # Medication intent
        c_med = self.doc.parse_doctor_utterance("The nurse will bring painkiller medicine now")
        assert c_med.icon == "MED"
        assert "Medication" in c_med.headline

        # Blood test intent
        c_bld = self.doc.parse_doctor_utterance("We need to draw a blood sample for lab test")
        assert c_bld.icon == "BLD"
        assert "Blood" in c_bld.headline

        # X-Ray intent
        c_xry = self.doc.parse_doctor_utterance("An orderly will take you to imaging for x-ray")
        assert c_xry.icon == "XRY"

        # Wait intent
        c_wait = self.doc.parse_doctor_utterance("Please wait 10 minutes for your test results")
        assert c_wait.icon == "WAIT"

        # Fasting intent
        c_fst = self.doc.parse_doctor_utterance("Have you eaten or are you fasting?")
        assert c_fst.icon == "FST"

        # Allergy intent
        c_alg = self.doc.parse_doctor_utterance("Do you have any drug allergy or reaction?")
        assert c_alg.icon == "ALG"

    def test_english_intent_parsing(self):
        c_med = self.doc.parse_doctor_utterance("The nurse will bring painkiller medicine")
        assert c_med.icon == "MED"

        c_safe = self.doc.parse_doctor_utterance("You are safe with our team, please relax")
        assert c_safe.icon == "SAFE"

        c_loc = self.doc.parse_doctor_utterance("Where does it hurt the most? Please point")
        assert c_loc.icon == "LOC"

    def test_custom_doctor_message(self):
        custom_txt = "Please remove your shoes and lie down on the examination table"
        c_custom = self.doc.parse_doctor_utterance(custom_txt)
        assert c_custom.card_id == 99
        assert c_custom.icon == "DOC"
        assert c_custom.subtext == custom_txt
        assert self.doc.active_card == c_custom


class TestFingerspellingEngine:
    def setup_method(self):
        self.fs = FingerspellingEngine(commit_delay=1.0)

    def test_letter_push_and_commit(self):
        self.fs.push_letter("A", now=10.0)
        self.fs.push_letter("S", now=10.2)
        self.fs.push_letter("P", now=10.4)
        self.fs.push_letter("I", now=10.6)
        self.fs.push_letter("R", now=10.8)
        self.fs.push_letter("I", now=11.0)
        self.fs.push_letter("N", now=11.2)

        assert self.fs.get_spelled_word() == "ASPIRIN"

        # Not committed yet before delay
        assert self.fs.check_auto_commit(now=11.5) is None

        # Auto committed after delay
        word = self.fs.check_auto_commit(now=12.5)
        assert word == "aspirin"
        assert self.fs.get_spelled_word() == ""

    def test_backspace_and_clear(self):
        self.fs.push_letter("P")
        self.fs.push_letter("I")
        self.fs.push_letter("L")
        self.fs.push_letter("X")
        assert self.fs.get_spelled_word() == "PILX"

        popped = self.fs.pop_letter()
        assert popped == "X"
        assert self.fs.get_spelled_word() == "PIL"

        self.fs.clear()
        assert self.fs.get_spelled_word() == ""

    def test_hand_shape_classification(self):
        # Synthetic hand coordinates (21, 3)
        # Wrist at [0.5, 0.8]
        hand = np.zeros((21, 3), dtype=np.float32)
        hand[0] = [0.5, 0.8, 0.0]
        # MCPs at y=0.6
        for mcp in [2, 5, 9, 13, 17]:
            hand[mcp] = [0.5, 0.6, 0.0]

        # Case 1: Letter B (4 fingers up, thumb curled)
        # Tips at y=0.2 (much farther from wrist than MCPs)
        for tip in [8, 12, 16, 20]:
            hand[tip] = [0.5, 0.2, 0.0]
        # Thumb tip near palm y=0.65
        hand[4] = [0.5, 0.65, 0.0]

        res = FingerspellingEngine.classify_hand_shape(hand)
        assert res == "B"

        # Case 2: Letter I (Pinky up alone)
        for tip in [4, 8, 12, 16]:
            hand[tip] = [0.5, 0.65, 0.0]
        hand[20] = [0.5, 0.2, 0.0]  # Pinky extended
        res = FingerspellingEngine.classify_hand_shape(hand)
        assert res == "I"


class TestSplitClinicalDashboard:
    def test_render_split_mode(self):
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        triage = TriageAgent().assess(["chest", "pain"], "I have severe chest pain", pain_score=8, facial_distress=True)
        doc = TwoWayDoctorAgent()
        card = doc.parse_doctor_utterance("We will draw a blood test now")
        diag = DiagnosticAgent().analyze(["chest", "pain"], "I have chest pain", pain_score=8)

        canvas = render_modern_dashboard(
            frame=dummy_frame,
            lm=None,
            show_skel=False,
            now=100.0,
            status="READY",
            status_type="ready",
            cand_txt="test",
            words=[{"w": "chest"}, {"w": "pain"}],
            sentence="I have chest pain",
            alt_sentence="",
            suggestions=[("medicine", 0.85)],
            log=[],
            speaker_active=True,
            fps=30.0,
            speed=0.0,
            progress=0.0,
            scale=1.0,
            triage=triage,
            pain_info={"pain_score": 8, "facial_distress": True, "brow_tension": 0.82, "squint_ratio": 0.74, "mouth_grimace": 0.55},
            drug_alerts=[],
            active_card=card,
            ui_mode="split",
            doctor_reply_menu=False,
            toast_message="Test toast",
            spelled_text="ASPIRIN",
            diagnosis=diag,
            neural_cands=[("chest pain", 0.94, "BiGRU"), ("heart attack", 0.42, "Context")],
        )
        assert canvas.shape == (780, 640, 3)

    def test_worker_lifecycle(self):
        session = ClinicalSession()
        doc = TwoWayDoctorAgent()
        c_worker = DoctorConsoleWorker(doc, session)
        v_worker = DoctorVoiceWorker(doc, session)
        assert c_worker.daemon is True
        assert v_worker.daemon is True
        c_worker.alive = False
        v_worker.alive = False
