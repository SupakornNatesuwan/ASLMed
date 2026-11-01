# ASLMed: Two-Way Clinical AI & Multi-Agent Sign Language Interpreter

A standalone desktop American Sign Language (ASL) medical interpreter and two-way communication system designed for real-life hospital emergency departments, outpatient clinics, and everyday communication. Runs directly from VS Code (`asl_main.py` or `./run.sh`) with zero web-server overhead.

---

## 🌟 Key Upgrades & Two-Way Features

### 1. Two-Way Doctor-Patient Communication Suite (`src/asl_clinical_agents.py`)
- **Split Clinical Mode (`--mode split`, Default)**:
  - Divides the screen into two live panes: **Patient View** on the left and **Doctor Clinical HUD** on the right.
  - Doctor and deaf/hard-of-hearing patient can look at the screen simultaneously and communicate with zero delay!
  - Cycle views anytime by pressing **`D`**: `Split Mode` $\rightarrow$ `Doctor Mode` $\rightarrow$ `Patient Mode` $\rightarrow$ `Split Mode`.
- **Doctor Speech-to-Card & Text-to-Card AI (`DoctorVoiceWorker` & `DoctorConsoleWorker`)**:
  - Doctor can **speak naturally** into the microphone or **type any sentence in Terminal**.
  - The AI parses the doctor's words and displays the corresponding visual card on the patient's screen immediately:
    - 💊 *"The nurse will bring medicine now"* $\rightarrow$ **[MED] Medication is Coming**
    - 💉 *"We need to draw a blood sample"* $\rightarrow$ **[BLD] Blood Test Needed**
    - 🩻 *"Taking you to imaging for x-ray"* $\rightarrow$ **[XRY] X-Ray / Scan Scheduled**
    - ⏱️ *"Please wait about 10 minutes"* $\rightarrow$ **[WAIT] Please Wait ~10 Minutes**
    - 🩺 *"Doctor is examining your breathing"* $\rightarrow$ **[DOC] Doctor is Examining You**
    - 💧 *"Water is being brought for you"* $\rightarrow$ **[H2O] Water is on the Way**
    - 👍 *"You are safe with our team"* $\rightarrow$ **[SAFE] You are Safe With Us**
    - ❓ *"Where does it hurt the most? Point here"* $\rightarrow$ **[LOC] Where Does it Hurt Most?**
    - 🍽️ *"Are you fasting or have you eaten?"* $\rightarrow$ **[FST] Fasting / Eating Status**
    - ⚠️ *"Do you have any drug allergies?"* $\rightarrow$ **[ALG] Any Known Drug Allergies?**
  - **Dynamic Custom Cards**: Any custom sentence typed or spoken by the doctor is automatically converted into an on-screen doctor card!

### 2. Patient Fingerspelling & Manual Alphabet Engine (`src/asl_fingerspelling.py`)
- Allows patients to spell out specific medication names, proper nouns, or symptoms letter-by-letter (A–Z) using 3D hand landmarks or keyboard.
- Automatically matches spelled words against clinical databases (e.g., `ASPIRIN`, `TYLENOL`, `PANADOL`, `AMOXICILLIN`, `PENICILLIN`) and routes them to the **Drug Safety Agent**!

### 3. Clinical Multi-Agent Intelligence & Diagnostic Co-Pilot
- **Diagnostic Reasoning Agent (DDx)**: Generates real-time Differential Diagnosis (top 3 medical hypotheses), recommended diagnostic workup (e.g. 12-Lead EKG, Troponin, CXR), and physician inquiry prompts.
- **Triage Agent (ESI Levels 1-5)**: Real-time Emergency Severity Index classification, target response windows, and vital panels.
- **Drug Safety & Allergy Agent**: Automated contraindication checks (Asthma/AERD with NSAIDs, Pregnancy, Peptic Ulcers, Penicillin).
- **Clinical Scribe Agent**: Generates standardized medical **SOAP Documentation** (Subjective, Objective, Assessment, Plan).

### 4. Neural Softmax & Facial Biometric AI Vectors
- **Live Neural Confidence Distribution**: Visualizes real-time softmax probability bars from BiGRU+Attention, Roboflow, and Contextual Markov Intent models.
- **MediaPipe 468-Point FaceMesh Biometrics**: Real-time feature vectors (Brow Tension, Eye Squint, Mouth Grimace, Pain Scale 1-10, and ASL Non-Manual grammatical markers).

### 5. Large-Screen High-Definition Interface
- Automatically launches in a high-resolution, widescreen window (`1280x920`), fully resizable and maximizable to any display or projector.

### 6. Medical Encounter Exporter (`src/asl_session_exporter.py`)
- Press **`E`** to export a complete clinical session to `exports/`:
  - **Clinical Markdown (`.md`)**: Full formatted SOAP note, differential diagnosis, triage classification, safety alerts, and interaction audit trail.
  - **EHR/EMR JSON (`.json`)**: Structured payload for hospital information systems.

---

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Using uv (fastest) or standard venv:
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env
```

### 2. Launch Application
```bash
# Easy launcher:
./run.sh

# Or with custom options (Default opens in Two-Way Split Mode):
python asl_main.py --mode split --conditions asthma --allergies penicillin
```

### 3. Keyboard Controls
| Key | Action |
| :--- | :--- |
| **`D`** | **Cycle Display Mode** (`Split Two-Way View` $\rightarrow$ `Doctor Dashboard` $\rightarrow$ `Patient View`) |
| **`T`** | **Open Doctor Quick-Reply Menu** (Press `1`–`9`, `0` to send visual cards to patient) |
| **`/`** | **Type Custom Doctor Message** (Type any English/Thai message in terminal to send to patient) |
| **`E`** | **Export Encounter Report** to `exports/` (`.md` and `.json`) |
| **`1` / `2` / `3`** | **Smart Intent Quick Pick** (Accept top suggested next sign) |
| **`ENTER`** | Speak current sentence aloud |
| **`TAB`** | Cycle alternative natural phrasing styles |
| **`BACKSPACE`** | Delete last signed word |
| **`C`** | Clear current words and spelled letters |
| **`R`** | Repeat last spoken sentence |
| **`A`** | Accept uncertain / borderline sign |
| **`S`** | Toggle glowing skeleton overlay |
| **`V`** | Toggle Text-to-Speech voice on / off |
| **`Q` / `ESC`** | Quit application |

---

## 🧪 Testing

Run the automated test suite (64 unit tests covering two-way communication, fingerspelling, clinical agents, triage, face mesh, session export, and models):

```bash
pytest tests/ -v
```

---

## 📂 Project Structure

```text
ASLMed/
├── asl_main.py                 # Main Application with Two-Way Split Mode & Multi-Agent HUD
├── run.sh                      # One-click execution script
├── .env                        # Local secrets / API keys (never pushed to git)
├── .env.example                # Template for environment configuration
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── src/                        # Core Application Modules & Multi-Agent Engine
│   ├── asl_clinical_agents.py  # Two-Way Doctor Agent, Triage (ESI 1-5), Drug Safety, SOAP Scribe
│   ├── asl_fingerspelling.py   # Fingerspelling & Manual Alphabet Engine (A-Z, Drug matching)
│   ├── asl_session_exporter.py # Clinical Session & Report Exporter (.md, .json)
│   ├── asl_face_mesh.py        # Facial Pain Grimace (1-10) & Non-Manual Marker Analyzer
│   ├── asl_intent.py           # Smart Intent Engine (Context Priors & Softened Thresholds)
│   ├── asl_ui.py               # Two-Way Split HUD, Dark Glassmorphism & Visual Cards
│   ├── asl_dataset.py          # 50-Class Vocabulary, Landmark Extractor & Smoother
│   ├── asl_model.py            # Neural Architectures (BiGRU, Transformer, Hybrid)
│   └── asl_sentence.py         # AI Sentence Interpreter (realise & realise_variants)
│
├── models/                     # Trained Model Weights
│   ├── asl_sign_model.pt       # Production PyTorch Ensemble Weights (50 classes)
│   └── asl_sign_model.torchscript # High-speed CPU Deployment Model
│
├── tests/                      # Automated Unit Test Suite (64 tests)
│   ├── test_asl_two_way.py     # Two-Way Doctor Cards, Intent Parsing, Fingerspelling, Split View
│   ├── test_asl_clinical.py    # Clinical Agents, Triage, Face Mesh, Session Exports
│   ├── test_asl_main_logic.py  # UI Dashboard, Segmenter, Intent Engine Logic
│   ├── test_asl_model.py       # PyTorch Forward, Features, Dimensions, TorchScript
│   ├── test_asl_dataset.py     # Smoothing, Hands, Normalization, Trim, Speed
│   └── test_asl_sentence.py    # Multi-clause, Negations, Symptoms, Phrasings
│
└── exports/                    # (Auto-generated) Exported Medical Encounter Reports (.md, .json)
```
