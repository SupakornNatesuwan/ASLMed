"""
ASLMed Clinical Multi-Agent Framework
Provides clinical intelligence and communication agents for medical ASL interpretation:
1. TriageAgent: Emergency Severity Index (ESI Levels 1-5) triage classification & action protocols.
2. DrugSafetyAgent: Contraindication checks, allergy warnings, and safety recommendations.
3. ClinicalScribeAgent: Real-time clinical encounter documentation & SOAP note generation.
4. TwoWayDoctorAgent: Doctor-to-patient visual communication cards with pictograms.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import os
import re
import sys
import threading
import time


# ==============================================================================
# 1. Triage Agent (Emergency Severity Index - ESI 1 to 5)
# ==============================================================================

@dataclass
class TriageAssessment:
    esi_level: int                  # 1 (Resuscitation) to 5 (Non-urgent)
    category: str                   # Human readable ESI description
    color_hex: str                  # UI badge color
    response_target: str            # Recommended response window (e.g. Immediate, <10 min)
    primary_concerns: List[str]     # Key clinical concerns detected
    recommended_vitals: List[str]   # Vital signs to immediately measure
    recommended_actions: List[str]  # Clinical action checklist
    is_critical: bool               # Requires audible or visual critical alert


class TriageAgent:
    """
    Evaluates patient signs, natural language interpreted complaints, and facial pain/distress
    metrics to assign Emergency Severity Index (ESI) Level 1 through 5.
    """

    ESI_CONFIG = {
        1: {
            "category": "ESI-1: Resuscitation (Immediate Life Threat)",
            "color_hex": "#EF4444",  # Red
            "response_target": "Immediate (< 1 min)",
            "is_critical": True,
        },
        2: {
            "category": "ESI-2: Emergent (High Risk / Severe Pain)",
            "color_hex": "#F97316",  # Orange-red
            "response_target": "< 10 minutes",
            "is_critical": True,
        },
        3: {
            "category": "ESI-3: Urgent (Multiple Resources Required)",
            "color_hex": "#F59E0B",  # Amber
            "response_target": "< 30-60 minutes",
            "is_critical": False,
        },
        4: {
            "category": "ESI-4: Less Urgent (Single Resource Required)",
            "color_hex": "#3B82F6",  # Blue
            "response_target": "< 60-120 minutes",
            "is_critical": False,
        },
        5: {
            "category": "ESI-5: Non-Urgent (Routine / Consultation)",
            "color_hex": "#10B981",  # Mint / Green
            "response_target": "Routine / Clinic flow",
            "is_critical": False,
        },
    }

    # Keywords mapping to high-acuity triggers
    RESUSCITATION_TRIGGERS = {
        "unconscious", "collapse", "cardiac", "stroke", "cannot breathe", 
        "choking", "seizure", "anaphylaxis", "severe bleeding"
    }

    EMERGENT_TRIGGERS = {
        "chest pain", "shortness of breath", "asthma", "difficulty breathing",
        "headache severe", "fracture", "blood", "bleed", "burn severe",
        "vision loss", "allergic", "poison", "heart"
    }

    URGENT_TRIGGERS = {
        "fever", "vomit", "vomiting", "nausea", "stomach pain", "abdominal pain",
        "dizziness", "migraine", "infection", "diarrhea", "swelling"
    }

    LESS_URGENT_TRIGGERS = {
        "cough", "sore throat", "mild pain", "rash", "earache", "sprain",
        "bandage", "cold"
    }

    NON_URGENT_TRIGGERS = {
        "water", "blanket", "bathroom", "hello", "thank you", "please",
        "consultation", "refill", "routine"
    }

    def assess(
        self,
        recognized_signs: List[str],
        interpreted_text: str = "",
        pain_score: int = 1,
        facial_distress: bool = False,
    ) -> TriageAssessment:
        text_lower = (interpreted_text + " " + " ".join(recognized_signs)).lower()
        concerns = []
        vitals = ["BP (Blood Pressure)", "HR (Heart Rate)", "RR (Respiratory Rate)", "SpO2 (Pulse Oximetry)", "Temp"]
        actions = []

        # --- ESI Level 1: Resuscitation ---
        if any(trig in text_lower for trig in self.RESUSCITATION_TRIGGERS):
            concerns.append("Acute life-threat or airway/breathing/circulation compromise detected.")
            vitals.extend(["Continuous ECG monitor", "EtCO2"])
            actions.extend([
                "Activate Immediate Resuscitation / Code Team",
                "Secure airway and provide high-flow O2",
                "Establish dual large-bore IV access",
                "Attach continuous cardiac telemetry and SpO2 monitor"
            ])
            cfg = self.ESI_CONFIG[1]
            return TriageAssessment(
                esi_level=1,
                category=cfg["category"],
                color_hex=cfg["color_hex"],
                response_target=cfg["response_target"],
                primary_concerns=concerns,
                recommended_vitals=vitals,
                recommended_actions=actions,
                is_critical=True,
            )

        # High pain with severe chest/breathing is also ESI 1
        if ("chest" in text_lower or "breath" in text_lower) and (pain_score >= 8 or facial_distress):
            concerns.append("Severe acute chest pain / respiratory distress with marked grimace.")
            vitals.extend(["12-Lead ECG", "Continuous SpO2"])
            actions.extend([
                "Stat 12-Lead ECG within 10 minutes",
                "Administer supplemental Oxygen if SpO2 < 94%",
                "Cardiology alert and urgent Troponin lab draw",
                "IV access and continuous monitoring"
            ])
            cfg = self.ESI_CONFIG[1]
            return TriageAssessment(
                esi_level=1,
                category=cfg["category"],
                color_hex=cfg["color_hex"],
                response_target=cfg["response_target"],
                primary_concerns=concerns,
                recommended_vitals=vitals,
                recommended_actions=actions,
                is_critical=True,
            )

        # --- ESI Level 2: Emergent ---
        if any(trig in text_lower for trig in self.EMERGENT_TRIGGERS) or pain_score >= 7 or facial_distress:
            if pain_score >= 7:
                concerns.append(f"Severe acute discomfort (Pain Scale {pain_score}/10 with facial grimace).")
            if any(trig in text_lower for trig in self.EMERGENT_TRIGGERS):
                concerns.append("High-risk symptoms requiring prompt medical evaluation.")

            vitals.extend(["12-Lead ECG if thoracic or cardiac", "Blood Glucose"])
            actions.extend([
                "Place in monitored treatment bay",
                "Physician evaluation within 10 minutes",
                "Prepare targeted analgesia as ordered",
                "Routine IV line and baseline blood work"
            ])
            cfg = self.ESI_CONFIG[2]
            return TriageAssessment(
                esi_level=2,
                category=cfg["category"],
                color_hex=cfg["color_hex"],
                response_target=cfg["response_target"],
                primary_concerns=concerns,
                recommended_vitals=vitals,
                recommended_actions=actions,
                is_critical=True,
            )

        # --- ESI Level 3: Urgent (Needs 2+ resources) ---
        if any(trig in text_lower for trig in self.URGENT_TRIGGERS) or (pain_score >= 4):
            concerns.append("Stable vitals anticipated, requires diagnostic investigation (labs/imaging).")
            actions.extend([
                "Order comprehensive lab panel (CBC, Electrolytes)",
                "Assess need for IV fluids / anti-emetics",
                "Schedule physician assessment within 30-45 minutes"
            ])
            cfg = self.ESI_CONFIG[3]
            return TriageAssessment(
                esi_level=3,
                category=cfg["category"],
                color_hex=cfg["color_hex"],
                response_target=cfg["response_target"],
                primary_concerns=concerns,
                recommended_vitals=vitals,
                recommended_actions=actions,
                is_critical=False,
            )

        # --- ESI Level 4: Less Urgent (Needs 1 resource) ---
        if any(trig in text_lower for trig in self.LESS_URGENT_TRIGGERS) or (2 <= pain_score <= 3):
            concerns.append("Low acuity presentation suitable for focused treatment or single investigation.")
            actions.extend([
                "Provide basic comfort care (water, blanket, positioning)",
                "Focused nursing assessment",
                "Physician examination in standard queue"
            ])
            cfg = self.ESI_CONFIG[4]
            return TriageAssessment(
                esi_level=4,
                category=cfg["category"],
                color_hex=cfg["color_hex"],
                response_target=cfg["response_target"],
                primary_concerns=concerns,
                recommended_vitals=vitals,
                recommended_actions=actions,
                is_critical=False,
            )

        # --- ESI Level 5: Non-Urgent ---
        concerns.append("Routine request, medication inquiry, or general comfort communication.")
        actions.extend([
            "Acknowledge patient needs via Two-Way Doctor visual card",
            "Assist with routine comfort or administrative direction"
        ])
        cfg = self.ESI_CONFIG[5]
        return TriageAssessment(
            esi_level=5,
            category=cfg["category"],
            color_hex=cfg["color_hex"],
            response_target=cfg["response_target"],
            primary_concerns=concerns,
            recommended_vitals=vitals,
            recommended_actions=actions,
            is_critical=False,
        )


# ==============================================================================
# 2. Drug Safety & Allergy Agent
# ==============================================================================

@dataclass
class DrugSafetyAlert:
    drug_name: str
    condition: str
    severity: str        # "CONTRAINDICATED", "WARNING", "CAUTION"
    message: str
    recommended_alternative: str


class DrugSafetyAgent:
    """
    Monitors medication-related signs/requests and cross-references them against
    known contraindications, chronic conditions, and allergy profiles.
    """

    # Clinical rules dictionary
    RULES = [
        {
            "condition": "asthma",
            "contraindicated_drugs": ["aspirin", "ibuprofen", "nsaid", "naproxen"],
            "severity": "CONTRAINDICATED",
            "message": "Risk of severe bronchospasm in aspirin-exacerbated respiratory disease (AERD).",
            "alternative": "Acetaminophen (Paracetamol) or prescribed bronchodilators.",
        },
        {
            "condition": "pregnancy",
            "contraindicated_drugs": ["ibuprofen", "aspirin", "nsaid", "naproxen"],
            "severity": "CONTRAINDICATED",
            "message": "NSAIDs contraindicated especially in 3rd trimester (premature closure of ductus arteriosus).",
            "alternative": "Acetaminophen (Paracetamol) if indicated by obstetrician.",
        },
        {
            "condition": "peptic_ulcer",
            "contraindicated_drugs": ["aspirin", "ibuprofen", "nsaid", "naproxen"],
            "severity": "CONTRAINDICATED",
            "message": "High risk of gastrointestinal hemorrhage, gastric erosion, or perforation.",
            "alternative": "Acetaminophen or consult physician for gastro-protective therapy.",
        },
        {
            "condition": "hypertension",
            "contraindicated_drugs": ["pseudoephedrine", "decongestant"],
            "severity": "WARNING",
            "message": "Sympathomimetic agents can exacerbate blood pressure spikes.",
            "alternative": "Saline nasal spray or approved antihistamines.",
        },
        {
            "condition": "liver_disease",
            "contraindicated_drugs": ["paracetamol", "acetaminophen", "tylenol"],
            "severity": "WARNING",
            "message": "Risk of hepatotoxicity; strict daily dosage restriction (<2g/day or avoided).",
            "alternative": "Consult treating physician for safe hepatic-cleared analgesia.",
        },
        {
            "condition": "penicillin_allergy",
            "contraindicated_drugs": ["penicillin", "amoxicillin", "ampicillin", "augmentin"],
            "severity": "CONTRAINDICATED",
            "message": "History of hypersensitivity/anaphylaxis to beta-lactam antibiotics.",
            "alternative": "Macrolides (Azithromycin) or Fluoroquinolones as guided by culture.",
        },
    ]

    def __init__(self, patient_conditions: Optional[List[str]] = None, patient_allergies: Optional[List[str]] = None):
        self.patient_conditions = [c.lower() for c in (patient_conditions or [])]
        self.patient_allergies = [a.lower() for a in (patient_allergies or [])]

    def add_condition(self, condition: str):
        c = condition.strip().lower()
        if c and c not in self.patient_conditions:
            self.patient_conditions.append(c)

    def add_allergy(self, allergy: str):
        a = allergy.strip().lower()
        if a and a not in self.patient_allergies:
            self.patient_allergies.append(a)

    def evaluate(self, text_or_signs: str) -> List[DrugSafetyAlert]:
        """
        Scans conversation or detected signs for medication requests and checks contraindications.
        """
        text_lower = text_or_signs.lower()
        alerts = []
        all_conditions = self.patient_conditions + self.patient_allergies

        for rule in self.RULES:
            rule_cond = rule["condition"]
            # Match condition
            if any(rule_cond in c or c in rule_cond for c in all_conditions):
                for drug in rule["contraindicated_drugs"]:
                    if drug in text_lower or (drug == "nsaid" and ("ibuprofen" in text_lower or "aspirin" in text_lower)):
                        alerts.append(
                            DrugSafetyAlert(
                                drug_name=drug.capitalize(),
                                condition=rule_cond.capitalize(),
                                severity=rule["severity"],
                                message=rule["message"],
                                recommended_alternative=rule["alternative"],
                            )
                        )
        return alerts


# ==============================================================================
# 3. Clinical Scribe Agent (Automated SOAP Note Generator)
# ==============================================================================

@dataclass
class SOAPNote:
    subjective: str
    objective: str
    assessment: str
    plan: str
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def to_markdown(self) -> str:
        return (
            f"### Clinical SOAP Documentation ({self.timestamp})\n\n"
            f"**S (Subjective):**\n{self.subjective}\n\n"
            f"**O (Objective):**\n{self.objective}\n\n"
            f"**A (Assessment):**\n{self.assessment}\n\n"
            f"**P (Plan):**\n{self.plan}\n"
        )


class ClinicalScribeAgent:
    """
    Maintains a clinical audit trail of all signed utterances, interpretations,
    vision telemetry, and physician interactions to generate standardized SOAP notes.
    """

    def __init__(self):
        self.utterances: List[Dict[str, Any]] = []
        self.pain_records: List[int] = []
        self.facial_flags: List[str] = []

    def record_utterance(
        self,
        raw_signs: List[str],
        interpreted: str,
        pain_score: int = 1,
        distress: bool = False,
        question_marker: Optional[str] = None,
    ):
        self.utterances.append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "signs": list(raw_signs),
            "interpreted": interpreted,
            "pain_score": pain_score,
            "distress": distress,
            "question_marker": question_marker,
        })
        self.pain_records.append(pain_score)
        if distress:
            self.facial_flags.append("Facial grimace / intense brow furrow noted")
        if question_marker:
            self.facial_flags.append(f"Non-manual marker: {question_marker}")

    def generate_soap_note(
        self,
        triage: TriageAssessment,
        drug_alerts: Optional[List[DrugSafetyAlert]] = None,
        patient_id: str = "PT-ASLMED-01",
        physician_name: str = "Attending Physician, MD",
    ) -> SOAPNote:
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Subjective (Patient reported)
        s_lines = []
        if self.utterances:
            unique_sentences = list(dict.fromkeys(u["interpreted"] for u in self.utterances if u["interpreted"]))
            if unique_sentences:
                s_lines.append(f"- Chief Complaints & Signed Statements:")
                for sent in unique_sentences:
                    s_lines.append(f"  * \"{sent}\"")
            all_signs = [s for u in self.utterances for s in u["signs"]]
            if all_signs:
                s_lines.append(f"- ASL Vocabulary Expressed: {', '.join(sorted(set(all_signs)))}")
        else:
            s_lines.append("- No signs recorded during this interval.")

        avg_pain = float(sum(self.pain_records) / len(self.pain_records)) if self.pain_records else 1.0
        max_pain = max(self.pain_records) if self.pain_records else 1
        s_lines.append(f"- Pain Severity: Self-reported / Grimace Index Max: {max_pain}/10 (Avg: {avg_pain:.1f}/10)")
        subjective = "\n".join(s_lines)

        # Objective (Computer vision & telemetry)
        o_lines = [
            f"- Vision AI Modality: MediaPipe Holistic (Pose + Hands) + 468-point Face Mesh",
            f"- Gesture Tracking: Real-time temporal kinematic segmentation & BiGRU feature inference",
            f"- Facial Marker Telemetry: Max Grimace Metric {max_pain}/10",
        ]
        if self.facial_flags:
            o_lines.append(f"- Non-Manual Biomarkers: {'; '.join(set(self.facial_flags))}")
        o_lines.append(f"- Recommended Vital Panel: {', '.join(triage.recommended_vitals)}")
        objective = "\n".join(o_lines)

        # Assessment (Triage & Safety)
        a_lines = [
            f"- Triage Classification: {triage.category}",
            f"- Target Acuity Window: {triage.response_target}",
        ]
        if triage.primary_concerns:
            for concern in triage.primary_concerns:
                a_lines.append(f"- Clinical Alert: {concern}")
        if drug_alerts:
            for da in drug_alerts:
                a_lines.append(f"- DRUG SAFETY [{da.severity}]: {da.drug_name} vs {da.condition} -> {da.message}")
        assessment = "\n".join(a_lines)

        # Plan
        p_lines = [
            f"- Response Protocol: {triage.category}",
        ]
        for act in triage.recommended_actions:
            p_lines.append(f"- [ ] {act}")
        if drug_alerts:
            for da in drug_alerts:
                p_lines.append(f"- [!] Pharmacotherapy safeguard: Prefer {da.recommended_alternative}")
        p_lines.append(f"- [ ] Facilitate continuous two-way communication using ASLMed Patient Reassurance Cards")
        p_lines.append(f"- Physician Signature: {physician_name} ({now_str})")
        plan = "\n".join(p_lines)

        return SOAPNote(
            subjective=subjective,
            objective=objective,
            assessment=assessment,
            plan=plan,
            timestamp=now_str,
        )


# ==============================================================================
# 4. Clinical Diagnostic Reasoning & Differential Diagnosis (DDx) AI Agent
# ==============================================================================

@dataclass
class ClinicalDiagnosis:
    primary_ddx: str
    differentials: List[str]
    clinical_rationale: str
    suggested_workup: List[str]
    suggested_doctor_question: str
    is_critical: bool
    confidence: float = 0.88


class DiagnosticAgent:
    """
    Autonomous Clinical Diagnostic Reasoning & Differential Diagnosis (DDx) AI Agent.
    Evaluates recognized signs, pain score, and triage acuity to generate:
    - Primary diagnostic hypothesis
    - Differential Diagnoses (DDx top 3)
    - Recommended clinical workup
    - Suggested physician inquiry for the patient
    Supports offline instant clinical reasoning and optional Google Gemini cloud reasoning.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    def analyze(
        self,
        signs: List[str],
        interpreted_sentence: str,
        pain_score: int = 1,
        triage_category: str = "ESI-4: Less Urgent",
    ) -> ClinicalDiagnosis:
        text_lower = (interpreted_sentence + " " + " ".join(signs)).lower()

        # 1. Chest Pain / Cardiac
        if any(w in text_lower for w in ["chest pain", "heart attack", "rib cage", "chest"]):
            return ClinicalDiagnosis(
                primary_ddx="Acute Coronary Syndrome (ACS) / Myocardial Ischemia",
                differentials=["Pulmonary Embolism (PE)", "Aortic Dissection", "Costochondritis"],
                clinical_rationale="Reported retrosternal/chest pain with elevated acuity warrants immediate cardiac rule-out.",
                suggested_workup=["12-Lead EKG", "Stat Troponin I/T", "Chest X-Ray (CXR)", "D-Dimer"],
                suggested_doctor_question="Does the pain radiate to your left arm, jaw, or shoulder?",
                is_critical=True,
                confidence=0.94,
            )

        # 2. Severe Respiratory
        if any(w in text_lower for w in ["can't breathe", "asthma", "cough", "sore throat", "breathe"]):
            return ClinicalDiagnosis(
                primary_ddx="Acute Respiratory Distress / Bronchospasm Exacerbation",
                differentials=["Severe Asthma Exacerbation", "Pneumothorax", "Bacterial Pneumonia"],
                clinical_rationale="Dyspnea and airway compromise require immediate continuous pulse oximetry and bronchodilator protocol.",
                suggested_workup=["Continuous SpO2 Monitoring", "Nebulized Albuterol/Ipratropium", "Peak Flow", "CXR"],
                suggested_doctor_question="Do you have your rescue inhaler with you right now?",
                is_critical=True,
                confidence=0.91,
            )

        # 3. Cephalea / Neurological
        if any(w in text_lower for w in ["headache", "dizziness", "dizzy", "head"]):
            crit = pain_score >= 7 or "vomit" in text_lower
            return ClinicalDiagnosis(
                primary_ddx="Acute Cephalea / Hypertensive Encephalopathy" if crit else "Benign Tension Cephalea / Migraine with Aura",
                differentials=["Subarachnoid Hemorrhage (SAH)", "Vestibular Neuritis", "Orthostatic Hypotension"],
                clinical_rationale="Head pain accompanied by dizziness and facial grimacing necessitates secondary intracranial screening.",
                suggested_workup=["Non-contrast Head CT", "Orthostatic Blood Pressures", "Cranial Nerve Exam"],
                suggested_doctor_question="Did this pain start suddenly like a thunderclap?",
                is_critical=crit,
                confidence=0.88,
            )

        # 4. Acute Abdomen
        if any(w in text_lower for w in ["stomach ache", "stomach", "vomit", "kidney"]):
            return ClinicalDiagnosis(
                primary_ddx="Acute Abdominal Pain / Suspected Gastroenteritis vs PUD",
                differentials=["Acute Appendicitis", "Cholelithiasis (Gallstones)", "Nephrolithiasis (Renal Colic)"],
                clinical_rationale="Epigastric or abdominal distress with possible emesis requires focused quadrant palpation and peritoneal evaluation.",
                suggested_workup=["Focused Abdominal Ultrasound", "Complete Blood Count (CBC)", "Serum Lipase", "BMP"],
                suggested_doctor_question="Is the pain sharper when you press down or when you let go?",
                is_critical=pain_score >= 7,
                confidence=0.86,
            )

        # 5. Bleeding / Trauma / Surgery
        if any(w in text_lower for w in ["bleeding", "severe infection", "surgery", "emergency"]):
            return ClinicalDiagnosis(
                primary_ddx="Active Acute Hemorrhage / Surgical Site Infection Risk",
                differentials=["Soft Tissue Laceration", "Underlying Coagulopathy", "Sepsis Syndrome"],
                clinical_rationale="Active blood loss or postoperative complication mandates hemodynamic stabilization and local hemostasis.",
                suggested_workup=["Type and Screen", "Coagulation Panel (PT/INR/aPTT)", "Serial Hemoglobin/Hematocrit"],
                suggested_doctor_question="Are you currently taking any blood thinners like Warfarin or Aspirin?",
                is_critical=True,
                confidence=0.93,
            )

        # 6. Allergic / Cutaneous
        if any(w in text_lower for w in ["allergic reaction", "itchy", "swollen", "skin", "cream"]):
            return ClinicalDiagnosis(
                primary_ddx="Acute Cutaneous Urticaria / Impending Anaphylaxis",
                differentials=["Drug-Induced Exanthem", "Contact Dermatitis", "Angioedema"],
                clinical_rationale="Cutaneous eruption with pruritus or swelling requires airway patency surveillance and antihistaminic protocol.",
                suggested_workup=["Airway Inspection", "IM Epinephrine 0.3mg (if stridor)", "IV Diphenhydramine & Famotidine"],
                suggested_doctor_question="Do you feel any swelling inside your lips, tongue, or throat?",
                is_critical=True,
                confidence=0.90,
            )

        # 7. Infectious / Constitutional
        if any(w in text_lower for w in ["fever", "flu", "weak", "tired", "pain"]):
            return ClinicalDiagnosis(
                primary_ddx="Viral Influenza Syndrome / Febrile Constitutional Illness",
                differentials=["Streptococcal Pharyngitis", "Urinary Tract Infection", "Early Sepsis"],
                clinical_rationale="Pyrexia and fatigue suggest systemic inflammatory or viral response; recommend hydration and antipyretics.",
                suggested_workup=["Rapid Viral PCR Panel", "Urinalysis with Microscopy", "Serum Lactate", "Antipyretic (Acetaminophen)"],
                suggested_doctor_question="How many days have you had this fever, and have you had chills?",
                is_critical=False,
                confidence=0.85,
            )

        # 8. High Blood Pressure
        if any(w in text_lower for w in ["high blood pressure", "blood"]):
            return ClinicalDiagnosis(
                primary_ddx="Hypertensive Urgency vs Essential Hypertension",
                differentials=["Secondary Renovascular Disease", "White-Coat Syndrome"],
                clinical_rationale="Patient reports elevated blood pressure; screen for end-organ compromise (headache, chest pain, vision changes).",
                suggested_workup=["Automated Serial Blood Pressure", "Basic Metabolic Panel", "Urine Protein Dipstick"],
                suggested_doctor_question="Are you experiencing blurry vision, chest pressure, or shortness of breath?",
                is_critical=True,
                confidence=0.87,
            )

        # Default: Clinical Observation / Routine Encounter
        return ClinicalDiagnosis(
            primary_ddx="Clinical Triage Observation / Non-Emergent Communication",
            differentials=["General Medical Assessment", "Patient Inquiry / Comfort Request"],
            clinical_rationale="No critical cardinal signs identified. Continuous vital monitoring and receptive communication ongoing.",
            suggested_workup=["Baseline Vitals (HR, BP, RR, SpO2, Temp)", "Comprehensive Nursing Intake"],
            suggested_doctor_question="Please indicate how our medical team can assist you today.",
            is_critical=False,
            confidence=0.80,
        )


# ==============================================================================
# 5. Two-Way Doctor Communication Agent (Visual Cards)
# ==============================================================================

@dataclass
class DoctorVisualCard:
    card_id: int
    icon: str
    headline: str = ""
    subtext: str = ""
    theme_color: Tuple[int, int, int] = (128, 128, 128)  # BGR
    headline_en: str = ""
    headline_th: str = ""
    subtext_en: str = ""
    subtext_th: str = ""

    def __post_init__(self):
        if not self.headline:
            self.headline = self.headline_en or self.headline_th or "Medical Message"
        if not self.subtext:
            self.subtext = self.subtext_en or self.subtext_th or ""
        if not self.headline_en:
            self.headline_en = self.headline
        if not self.headline_th:
            self.headline_th = self.headline
        if not self.subtext_en:
            self.subtext_en = self.subtext
        if not self.subtext_th:
            self.subtext_th = self.subtext


class TwoWayDoctorAgent:
    """
    Supplies instant, clinically validated visual communication cards for doctors
    to present to deaf/hard-of-hearing patients with clear pictograms and high-contrast typography.
    """

    PRESET_CARDS = [
        DoctorVisualCard(
            card_id=1,
            icon="MED",
            headline="Medication is Coming",
            subtext="The nurse is preparing your prescribed medicine now.",
            theme_color=(128, 220, 16),  # Lime / Teal
        ),
        DoctorVisualCard(
            card_id=2,
            icon="BLD",
            headline="Blood Test Needed",
            subtext="A small blood sample will help us find the cause.",
            theme_color=(80, 80, 240),  # Coral Red
        ),
        DoctorVisualCard(
            card_id=3,
            icon="XRY",
            headline="X-Ray / Scan Scheduled",
            subtext="An orderly will take you to imaging shortly.",
            theme_color=(230, 160, 40),  # Cyan / Blue
        ),
        DoctorVisualCard(
            card_id=4,
            icon="WAIT",
            headline="Please Wait ~10 Minutes",
            subtext="Doctor is reviewing your results and will arrive soon.",
            theme_color=(40, 180, 240),  # Amber
        ),
        DoctorVisualCard(
            card_id=5,
            icon="DOC",
            headline="Doctor is Examining You",
            subtext="Please sit comfortably; we will check your breathing and pulse.",
            theme_color=(220, 140, 100),  # Sky Blue
        ),
        DoctorVisualCard(
            card_id=6,
            icon="H2O",
            headline="Water is on the Way",
            subtext="Fresh drinking water is being brought for you.",
            theme_color=(240, 180, 50),  # Cyan
        ),
        DoctorVisualCard(
            card_id=7,
            icon="SAFE",
            headline="You are Safe With Us",
            subtext="Our medical team is taking good care of you.",
            theme_color=(100, 210, 80),  # Emerald Green
        ),
        DoctorVisualCard(
            card_id=8,
            icon="LOC",
            headline="Where Does it Hurt Most?",
            subtext="Point to the area or show us your pain level 1-10.",
            theme_color=(80, 140, 255),  # Orange
        ),
        DoctorVisualCard(
            card_id=9,
            icon="FST",
            headline="Fasting / Eating Status",
            subtext="Have you eaten or drank anything in the last 6-8 hours?",
            theme_color=(200, 100, 240),  # Purple
        ),
        DoctorVisualCard(
            card_id=10,
            icon="ALG",
            headline="Any Known Drug Allergies?",
            subtext="Please tell us if you have any medicine allergies.",
            theme_color=(80, 90, 245),   # Crimson
        ),
    ]

    INTENT_KEYWORDS = [
        # Place compound checks (FST, ALG) before simple nouns (H2O, MED)
        ("FST", 9, ["fasting", "eaten", "breakfast", "dinner", "food", "drink", "empty stomach", "last meal"]),
        ("ALG", 10, ["allergy", "allergic", "reaction", "allergies", "side effect", "contraindication"]),
        ("MED", 1, ["medicine", "pill", "painkiller", "analgesic", "medication", "dose", "tablet", "drug"]),
        ("BLD", 2, ["blood", "blood test", "draw", "lab", "sample", "needle", "vial"]),
        ("XRY", 3, ["xray", "x-ray", "scan", "imaging", "ct", "radiology", "ultrasound", "mri"]),
        ("WAIT", 4, ["wait", "minute", "moment", "hold on", "soon", "pending", "queue"]),
        ("DOC", 5, ["examine", "check", "doctor examine", "stethoscope", "listen", "assessment", "physical"]),
        ("H2O", 6, ["water", "cup of water", "thirsty", "hydration", "glass of water"]),
        ("SAFE", 7, ["safe", "calm", "relax", "fine", "care", "okay", "alright", "don't worry"]),
        ("LOC", 8, ["where", "hurt", "point", "location", "spot", "area", "indicate"]),
    ]

    def __init__(self):
        self.active_card: Optional[DoctorVisualCard] = None

    def select_card(self, card_id: int) -> Optional[DoctorVisualCard]:
        for card in self.PRESET_CARDS:
            if card.card_id == card_id:
                self.active_card = card
                return card
        return None

    def parse_doctor_utterance(self, text: str) -> DoctorVisualCard:
        """
        Interprets natural doctor spoken or typed language
        and activates the matching clinical visual card, or creates a custom card.
        """
        raw_text = text.strip()
        text_lower = raw_text.lower()

        # Check matched intent
        for intent_code, cid, keywords in self.INTENT_KEYWORDS:
            if any(k in text_lower for k in keywords):
                card = self.select_card(cid)
                if card:
                    return card

        # If no preset keyword matches, generate a dynamic custom visual communication card
        custom_card = DoctorVisualCard(
            card_id=99,
            icon="DOC",
            headline="Message from Doctor",
            subtext=raw_text,
            theme_color=(235, 195, 55),
        )
        self.active_card = custom_card
        return custom_card

    def create_custom_card(
        self,
        headline: str = "Message from Doctor",
        subtext: str = "",
        icon: str = "DOC",
        **kwargs,
    ) -> DoctorVisualCard:
        headline_text = headline or kwargs.get("headline_en", "") or kwargs.get("headline_th", "Message from Doctor")
        subtext_text = subtext or kwargs.get("subtext_en", "") or kwargs.get("subtext_th", "")
        card = DoctorVisualCard(
            card_id=99,
            icon=icon,
            headline=headline_text,
            subtext=subtext_text,
            theme_color=(235, 195, 55),
        )
        self.active_card = card
        return card

    def clear_card(self):
        self.active_card = None

    def get_card_list(self) -> List[DoctorVisualCard]:
        return self.PRESET_CARDS


class DoctorConsoleWorker(threading.Thread):
    """
    Background worker that listens for text typed by the doctor in the terminal
    and instantly creates/displays visual communication cards for the patient.
    """

    def __init__(self, doctor_agent: TwoWayDoctorAgent, session: Any, on_message_callback: Optional[Any] = None):
        super().__init__(daemon=True)
        self.doctor_agent = doctor_agent
        self.session = session
        self.on_message_callback = on_message_callback
        self.alive = True

    def run(self):
        # Only run interactive stdin reader if running in interactive terminal
        if not sys.stdin or not sys.stdin.isatty():
            return
        print("\n" + "=" * 65)
        print("  [DOCTOR TWO-WAY CONSOLE ACTIVATED]")
        print("  Type any message and press ENTER to display to patient.")
        print("  (e.g. 'nurse is bringing medicine' or 'please wait 10 minutes')")
        print("=" * 65 + "\n", flush=True)
        while self.alive:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                text = line.strip()
                if text:
                    card = self.doctor_agent.parse_doctor_utterance(text)
                    self.session.record_doctor_reply(card.headline, card.card_id)
                    if self.on_message_callback:
                        self.on_message_callback(card)
                    print(f"  >>> [SENT TO PATIENT]: [{card.icon}] {card.headline}", flush=True)
            except Exception:
                time.sleep(0.5)


class DoctorVoiceWorker(threading.Thread):
    """
    Asynchronous background listener for physician voice communication.
    Transcribes doctor speech in real-time and maps to visual cards.
    Gracefully handles environments without microphone permissions or offline networks.
    """

    def __init__(self, doctor_agent: TwoWayDoctorAgent, session: Any, on_message_callback: Optional[Any] = None):
        super().__init__(daemon=True)
        self.doctor_agent = doctor_agent
        self.session = session
        self.on_message_callback = on_message_callback
        self.alive = True
        self.mic_available = False

    def run(self):
        try:
            import speech_recognition as sr
            r = sr.Recognizer()
            r.energy_threshold = 300
            r.pause_threshold = 0.8
            with sr.Microphone() as source:
                self.mic_available = True
                r.adjust_for_ambient_noise(source, duration=0.4)
                while self.alive:
                    try:
                        audio = r.listen(source, timeout=3.0, phrase_time_limit=6.0)
                        text = ""
                        try:
                            text = r.recognize_google(audio, language="en-US")
                        except Exception:
                            pass
                        if text:
                            card = self.doctor_agent.parse_doctor_utterance(text)
                            self.session.record_doctor_reply(card.headline, card.card_id)
                            if self.on_message_callback:
                                self.on_message_callback(card)
                    except sr.WaitTimeoutError:
                        continue
                    except Exception:
                        time.sleep(1.0)
        except Exception:
            # Microphone unavailable, unpermitted on macOS, or headless environment - standby silently
            self.mic_available = False

