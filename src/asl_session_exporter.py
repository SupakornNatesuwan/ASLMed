"""
ASLMed Clinical Session Manager & Medical Encounter Exporter
Manages encounter lifecycle, records timeline events, and exports structured reports:
- Markdown Clinical Encounter (.md) with standardized SOAP documentation
- JSON EHR/EMR Integration Payload (.json)
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

from src.asl_clinical_agents import TriageAssessment, SOAPNote, DrugSafetyAlert


@dataclass
class SessionEvent:
    timestamp: str
    event_type: str  # "SIGN_DETECTED", "SENTENCE_INTERPRETED", "PAIN_UPDATE", "TRIAGE_TRIGGER", "DOCTOR_REPLY"
    payload: Dict[str, Any]


class ClinicalSession:
    """
    Manages active clinical session history, telemetry logs, and encounter exports.
    """

    def __init__(self, patient_id: str = "PT-ASLMED-01", clinic_unit: str = "Emergency Dept / Triage"):
        self.session_id: str = f"ENCOUNTER-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.patient_id: str = patient_id
        self.clinic_unit: str = clinic_unit
        self.start_time: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.end_time: Optional[str] = None
        self.events: List[SessionEvent] = []
        self.current_triage: Optional[TriageAssessment] = None
        self.latest_soap: Optional[SOAPNote] = None
        self.drug_alerts: List[DrugSafetyAlert] = []
        self.latest_diagnosis: Optional[Any] = None

    def set_diagnosis(self, diagnosis: Any):
        self.latest_diagnosis = diagnosis

    def record_sign(self, sign: str, confidence: float):
        ts = datetime.now().strftime("%H:%M:%S")
        self.events.append(SessionEvent(
            timestamp=ts,
            event_type="SIGN_DETECTED",
            payload={"sign": sign, "confidence": round(float(confidence), 3)},
        ))

    def record_sentence(self, sentence: str, pain_score: int, facial_distress: bool):
        ts = datetime.now().strftime("%H:%M:%S")
        self.events.append(SessionEvent(
            timestamp=ts,
            event_type="SENTENCE_INTERPRETED",
            payload={"sentence": sentence, "pain_score": pain_score, "facial_distress": facial_distress},
        ))

    def record_triage(self, triage: TriageAssessment):
        self.current_triage = triage
        ts = datetime.now().strftime("%H:%M:%S")
        self.events.append(SessionEvent(
            timestamp=ts,
            event_type="TRIAGE_TRIGGER",
            payload={
                "esi_level": triage.esi_level,
                "category": triage.category,
                "is_critical": triage.is_critical,
                "concerns": triage.primary_concerns,
            },
        ))

    def record_doctor_reply(self, card_headline: str, card_id: int):
        ts = datetime.now().strftime("%H:%M:%S")
        self.events.append(SessionEvent(
            timestamp=ts,
            event_type="DOCTOR_REPLY",
            payload={"card_id": card_id, "headline": card_headline},
        ))

    def set_soap_note(self, soap: SOAPNote):
        self.latest_soap = soap

    def set_drug_alerts(self, alerts: List[DrugSafetyAlert]):
        self.drug_alerts = alerts

    def export_markdown(self, output_dir: str = "exports") -> Path:
        """
        Exports full encounter summary as a standardized Clinical Markdown document.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        filename = f"{self.session_id.lower().replace('-', '_')}.md"
        target = out_path / filename

        self.end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Triage string
        triage_str = "Not Evaluated"
        response_win = "N/A"
        if self.current_triage:
            triage_str = f"**{self.current_triage.category}**"
            response_win = self.current_triage.response_target

        md = []
        md.append(f"# ASLMed Clinical Encounter Documentation")
        md.append(f"**Session Identifier:** `{self.session_id}`  ")
        md.append(f"**Patient ID:** `{self.patient_id}` | **Unit:** `{self.clinic_unit}`  ")
        md.append(f"**Start Time:** {self.start_time} | **Report Time:** {self.end_time}  ")
        md.append(f"**Emergency Triage Level:** {triage_str} (Target: {response_win})\n")
        md.append("---\n")

        # SOAP Note
        if self.latest_soap:
            md.append(self.latest_soap.to_markdown())
            md.append("---\n")

        # Differential Diagnosis & AI Clinical Reasoning
        if self.latest_diagnosis:
            md.append("### AI Differential Diagnosis & Clinical Reasoning (DDx)\n")
            md.append(f"- **Primary Diagnostic Hypothesis:** {self.latest_diagnosis.primary_ddx} (Confidence: {int(self.latest_diagnosis.confidence * 100)}%)")
            md.append(f"- **Differential Diagnoses:** {', '.join(self.latest_diagnosis.differentials)}")
            md.append(f"- **Clinical Rationale:** {self.latest_diagnosis.clinical_rationale}")
            md.append(f"- **Recommended Workup:** {', '.join(self.latest_diagnosis.suggested_workup)}")
            md.append(f"- **AI Suggested Inquiry:** \"{self.latest_diagnosis.suggested_doctor_question}\"")
            md.append("\n---\n")

        # Drug Safety Alerts
        if self.drug_alerts:
            md.append("### Active Pharmacotherapy / Safety Flags\n")
            for alert in self.drug_alerts:
                md.append(f"- **[{alert.severity}] {alert.drug_name}** with **{alert.condition}**: {alert.message}")
                md.append(f"  *Suggested Alternative:* {alert.recommended_alternative}")
            md.append("\n---\n")

        # Chronological Interaction Audit Trail
        md.append("### Chronological Interaction Audit Trail\n")
        md.append("| Time | Event Type | Details |")
        md.append("| :--- | :--- | :--- |")
        for ev in self.events:
            if ev.event_type == "SIGN_DETECTED":
                det = f"Signed **{ev.payload.get('sign')}** (Confidence: {ev.payload.get('confidence'):.2f})"
            elif ev.event_type == "SENTENCE_INTERPRETED":
                det = f"Interpreted: \"{ev.payload.get('sentence')}\" [Pain: {ev.payload.get('pain_score')}/10]"
            elif ev.event_type == "TRIAGE_TRIGGER":
                det = f"Triage Alert: **{ev.payload.get('category')}**"
            elif ev.event_type == "DOCTOR_REPLY":
                det = f"Doctor Visual Card: **{ev.payload.get('headline')}** (Card #{ev.payload.get('card_id')})"
            else:
                det = str(ev.payload)
            md.append(f"| {ev.timestamp} | `{ev.event_type}` | {det} |")

        target.write_text("\n".join(md), encoding="utf-8")
        return target

    def export_json(self, output_dir: str = "exports") -> Path:
        """
        Exports machine-readable JSON structure for EHR / EMR integration.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        filename = f"{self.session_id.lower().replace('-', '_')}.json"
        target = out_path / filename

        self.end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        payload = {
            "session_id": self.session_id,
            "patient_id": self.patient_id,
            "clinic_unit": self.clinic_unit,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "triage": asdict(self.current_triage) if self.current_triage else None,
            "diagnosis": asdict(self.latest_diagnosis) if self.latest_diagnosis else None,
            "soap_note": asdict(self.latest_soap) if self.latest_soap else None,
            "drug_safety_alerts": [asdict(a) for a in self.drug_alerts],
            "events": [
                {
                    "timestamp": e.timestamp,
                    "event_type": e.event_type,
                    "payload": e.payload,
                }
                for e in self.events
            ],
        }

        with open(target, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        return target
