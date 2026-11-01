"""
Comprehensive Unit Tests for ASLMed Clinical AI Multi-Agent Architecture:
1. TriageAgent (ESI Levels 1-5, critical triggers, vital protocols)
2. DrugSafetyAgent (Contraindications: Asthma, Pregnancy, Peptic Ulcer, Penicillin)
3. ClinicalScribeAgent (Standardized SOAP documentation)
4. TwoWayDoctorAgent (Communication cards with EN/TH pictograms)
5. FaceMeshAnalyzer (Grimace pain scale, Yes/No question markers, Wh- markers)
6. ClinicalSession & Session Exporter (Markdown & JSON report validation)
"""

import json
from pathlib import Path
import tempfile
import numpy as np
import pytest

from src.asl_clinical_agents import (
    TriageAgent,
    DrugSafetyAgent,
    ClinicalScribeAgent,
    TwoWayDoctorAgent,
    DiagnosticAgent,
)
from src.asl_face_mesh import FaceMeshAnalyzer
from src.asl_session_exporter import ClinicalSession


class TestTriageAgent:
    def setup_method(self):
        self.triage = TriageAgent()

    def test_esi_level_1_resuscitation(self):
        # Triggered by life-threatening keywords
        res = self.triage.assess(
            recognized_signs=["cannot", "breathe", "chest"],
            interpreted_text="I cannot breathe and feel chest pain",
            pain_score=9,
            facial_distress=True,
        )
        assert res.esi_level == 1
        assert "ESI-1" in res.category
        assert res.is_critical is True
        assert any("Code" in a or "Resuscitation" in a or "12-Lead" in a for a in res.recommended_actions)

    def test_esi_level_2_emergent(self):
        # High acuity symptoms or severe pain scale >= 7
        res = self.triage.assess(
            recognized_signs=["severe", "headache", "dizzy"],
            interpreted_text="I have severe headache",
            pain_score=8,
            facial_distress=True,
        )
        assert res.esi_level == 2
        assert "ESI-2" in res.category
        assert res.is_critical is True
        assert "< 10" in res.response_target

    def test_esi_level_3_urgent(self):
        # Moderate symptoms requiring 2+ resources
        res = self.triage.assess(
            recognized_signs=["fever", "vomit"],
            interpreted_text="I have fever and vomiting",
            pain_score=5,
            facial_distress=False,
        )
        assert res.esi_level == 3
        assert "ESI-3" in res.category
        assert res.is_critical is False

    def test_esi_level_4_less_urgent(self):
        # Low acuity presentation
        res = self.triage.assess(
            recognized_signs=["cough", "sore throat"],
            interpreted_text="I have a cough and mild sore throat",
            pain_score=2,
            facial_distress=False,
        )
        assert res.esi_level == 4
        assert "ESI-4" in res.category
        assert res.is_critical is False

    def test_esi_level_5_non_urgent(self):
        # Routine or comfort inquiry
        res = self.triage.assess(
            recognized_signs=["water", "please"],
            interpreted_text="May I have water please?",
            pain_score=1,
            facial_distress=False,
        )
        assert res.esi_level == 5
        assert "ESI-5" in res.category
        assert res.is_critical is False


class TestDrugSafetyAgent:
    def test_asthma_contraindication(self):
        agent = DrugSafetyAgent(patient_conditions=["asthma"])
        alerts = agent.evaluate("I need ibuprofen for my headache")
        assert len(alerts) > 0
        alert = alerts[0]
        assert alert.severity == "CONTRAINDICATED"
        assert "AERD" in alert.message or "bronchospasm" in alert.message
        assert "Acetaminophen" in alert.recommended_alternative

    def test_pregnancy_contraindication(self):
        agent = DrugSafetyAgent(patient_conditions=["pregnancy"])
        alerts = agent.evaluate("Can you give me aspirin?")
        assert len(alerts) > 0
        alert = alerts[0]
        assert alert.severity == "CONTRAINDICATED"
        assert "3rd trimester" in alert.message

    def test_peptic_ulcer_contraindication(self):
        agent = DrugSafetyAgent(patient_conditions=["peptic_ulcer"])
        alerts = agent.evaluate("I want an NSAID painkiller")
        assert len(alerts) > 0
        assert alerts[0].severity == "CONTRAINDICATED"
        assert "gastrointestinal" in alerts[0].message.lower()

    def test_penicillin_allergy(self):
        agent = DrugSafetyAgent(patient_allergies=["penicillin_allergy"])
        alerts = agent.evaluate("Doctor prescribes amoxicillin")
        assert len(alerts) > 0
        assert alerts[0].severity == "CONTRAINDICATED"
        assert "beta-lactam" in alerts[0].message

    def test_safe_medication_no_alert(self):
        agent = DrugSafetyAgent(patient_conditions=["asthma"])
        # Acetaminophen is safe in asthma
        alerts = agent.evaluate("May I take acetaminophen please?")
        assert len(alerts) == 0


class TestClinicalScribeAgent:
    def test_soap_note_generation(self):
        scribe = ClinicalScribeAgent()
        scribe.record_utterance(
            raw_signs=["chest", "pain", "severe"],
            interpreted="I have severe chest pain",
            pain_score=8,
            distress=True,
            question_marker=None,
        )
        triage = TriageAgent().assess(
            ["chest", "pain"], "I have severe chest pain", pain_score=8, facial_distress=True
        )
        drug_agent = DrugSafetyAgent(patient_conditions=["asthma"])
        drug_alerts = drug_agent.evaluate("I took aspirin earlier")

        soap = scribe.generate_soap_note(triage=triage, drug_alerts=drug_alerts)
        assert soap is not None
        assert "I have severe chest pain" in soap.subjective
        assert "MediaPipe" in soap.objective
        assert "8/10" in soap.objective or "8/10" in soap.subjective
        assert "ESI" in soap.assessment
        assert "DRUG SAFETY" in soap.assessment
        assert "Aspirin" in soap.assessment

        md_text = soap.to_markdown()
        assert "**S (Subjective):**" in md_text
        assert "**O (Objective):**" in md_text
        assert "**A (Assessment):**" in md_text
        assert "**P (Plan):**" in md_text


class TestTwoWayDoctorAgent:
    def test_preset_cards(self):
        doc = TwoWayDoctorAgent()
        cards = doc.get_card_list()
        assert len(cards) >= 8

        # Test selecting medication card
        card1 = doc.select_card(1)
        assert card1 is not None
        assert card1.icon == "MED"
        assert "Medication" in card1.headline
        assert "Medication" in card1.headline_en
        assert doc.active_card == card1

        # Test clearing
        doc.clear_card()
        assert doc.active_card is None


class TestFaceMeshAnalyzer:
    def setup_method(self):
        self.analyzer = FaceMeshAnalyzer(alpha=0.90)

    def test_none_or_missing_face(self):
        res = self.analyzer.update(None)
        assert res["has_face"] is False
        assert res["pain_score"] == 1
        assert res["brow_raised"] is False

    def test_neutral_face(self):
        # Create synthetic 468 landmarks for normal face
        face = np.zeros((468, 3), dtype=np.float32)
        # Inner eye corners (33 and 263)
        face[33] = [0.40, 0.40, 0.0]
        face[263] = [0.60, 0.40, 0.0]
        # Inner brows (66 and 296)
        face[66] = [0.44, 0.35, 0.0]
        face[296] = [0.56, 0.35, 0.0]
        # Eyelids
        face[159] = [0.43, 0.38, 0.0]
        face[145] = [0.43, 0.42, 0.0]
        face[386] = [0.57, 0.38, 0.0]
        face[374] = [0.57, 0.42, 0.0]
        # Mouth corners
        face[61] = [0.43, 0.65, 0.0]
        face[291] = [0.57, 0.65, 0.0]
        face[13] = [0.50, 0.63, 0.0]
        face[1] = [0.50, 0.50, 0.0]

        res = self.analyzer.update(face)
        assert res["has_face"] is True
        assert res["pain_score"] <= 4

    def test_eyebrow_raise_yes_no_question(self):
        face = np.zeros((468, 3), dtype=np.float32)
        face[33] = [0.40, 0.50, 0.0]
        face[263] = [0.60, 0.50, 0.0]
        # Brows raised high above eyes
        face[66] = [0.43, 0.30, 0.0]
        face[296] = [0.57, 0.30, 0.0]
        # Eye midpoints around y=0.50 -> distance ~ 0.20 / 0.20 = 1.0 > 0.33
        face[159] = [0.43, 0.49, 0.0]
        face[145] = [0.43, 0.51, 0.0]
        face[386] = [0.57, 0.49, 0.0]
        face[374] = [0.57, 0.51, 0.0]
        face[61] = [0.45, 0.65, 0.0]
        face[291] = [0.55, 0.65, 0.0]
        face[13] = [0.50, 0.63, 0.0]
        face[1] = [0.50, 0.50, 0.0]

        res = self.analyzer.update(face)
        assert res["brow_raised"] is True
        assert res["question_marker"] == "yes_no"


class TestClinicalSessionExporter:
    def test_session_lifecycle_and_exports(self):
        session = ClinicalSession(patient_id="TEST-PT-99", clinic_unit="ER Bed 4")
        session.record_sign("chest", 0.96)
        session.record_sign("pain", 0.94)
        session.record_sentence("I have chest pain", pain_score=7, facial_distress=True)

        triage = TriageAgent().assess(["chest", "pain"], "I have chest pain", pain_score=7, facial_distress=True)
        session.record_triage(triage)

        scribe = ClinicalScribeAgent()
        soap = scribe.generate_soap_note(triage)
        session.set_soap_note(soap)

        diag = DiagnosticAgent().analyze(["chest", "pain"], "I have chest pain", pain_score=7)
        session.set_diagnosis(diag)

        session.record_doctor_reply("Medication is Coming", 1)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test Markdown export
            md_path = session.export_markdown(output_dir=tmpdir)
            assert md_path.exists()
            content = md_path.read_text(encoding="utf-8")
            assert "ASLMed Clinical Encounter" in content
            assert "TEST-PT-99" in content
            assert "SOAP" in content
            assert "Differential Diagnosis" in content
            assert "Chronological Interaction Audit Trail" in content

            # Test JSON export
            json_path = session.export_json(output_dir=tmpdir)
            assert json_path.exists()
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert data["patient_id"] == "TEST-PT-99"
            assert data["clinic_unit"] == "ER Bed 4"
            assert data["triage"]["esi_level"] in (1, 2)
            assert data["diagnosis"] is not None
            assert len(data["events"]) >= 4


class TestDiagnosticAgent:
    def setup_method(self):
        self.diagnostic_agent = DiagnosticAgent()

    def test_chest_pain_differential_diagnosis(self):
        dx = self.diagnostic_agent.analyze(["chest", "pain"], "I have severe chest pain", pain_score=8)
        assert "Acute Coronary Syndrome" in dx.primary_ddx
        assert dx.is_critical is True
        assert len(dx.differentials) >= 2
        assert any("EKG" in w for w in dx.suggested_workup)
        assert "arm" in dx.suggested_doctor_question.lower()

    def test_respiratory_differential_diagnosis(self):
        dx = self.diagnostic_agent.analyze(["can't breathe", "asthma"], "I cannot breathe well", pain_score=6)
        assert "Respiratory" in dx.primary_ddx or "Asthma" in dx.primary_ddx
        assert dx.is_critical is True
        assert any("SpO2" in w for w in dx.suggested_workup)

    def test_neurological_differential_diagnosis(self):
        dx = self.diagnostic_agent.analyze(["headache", "dizziness"], "I have severe headache and feel dizzy", pain_score=8)
        assert "Cephalea" in dx.primary_ddx
        assert dx.is_critical is True
        assert any("CT" in w for w in dx.suggested_workup)

    def test_baseline_routine_diagnosis(self):
        dx = self.diagnostic_agent.analyze(["hello", "doctor"], "Hello doctor please", pain_score=1)
        assert "Observation" in dx.primary_ddx
        assert dx.is_critical is False
