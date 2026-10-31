# ASLMed: Real-Life Medical & Daily ASL Interpreter

A standalone desktop American Sign Language (ASL) interpreter designed for real-life clinical and everyday communication. Run directly from VS Code (`asl_main.py`) with zero web-server overhead.

---

## 🌟 Key Upgrades & Features

1. **Direct Standalone Desktop App (`asl_main.py`)**:
   - No web server required—just press **Run Python File** on `asl_main.py` in VS Code or run `python asl_main.py`.
   - Features a clean split-screen OpenCV HUD with live pose/hand skeleton overlay (`S` key), real-time top-model confidence indicators (`MED`, `DAILY`, `RF`), and a multi-clause sentence interpreter panel.

2. **50 Medical Signs + 18 Everyday ASL Signs (3-Way Fusion)**:
   - **50-Class Medical Model (`MedicalModel`)**: Trained on all 1,046 clips across classes `1..50` from `dataset_2` using a Deep BiGRU + Self-Attention ensemble with Bilateral Test-Time Augmentation (TTA) for left- and right-handed signers.
   - **Local 3D Kinematic Daily ASL Model (`DailyASLModel`)**: Uses MediaPipe 3D hand and body kinematics to recognize everyday conversation signs offline with zero latency:
     - `"I"`, `"want"`, `"feel"`, `"hello"`, `"need"`, `"please"`, `"thank-you"`, `"sorry"`, `"yes"`, `"no"`, `"help"`, `"good"`, `"water"`, `"bathroom"`, `"my"`, `"you"`, `"where"`, `"more"`
   - **Roboflow Cloud ASL Verifier (`RoboflowWorker`)**: Queries `asl-avoix/1` asynchronously and fuses its predictions with the local `DailyASLModel` and `MedicalModel` in `decide()`.

3. **Expert AI Sentence Interpreter (`asl_sentence.py`)**:
   - **Flexible, Non-Rigid Grammar**: Accepts signs in any natural ASL order without dropping words.
   - **Context-Aware Linking**: Automatically connects desires/requests (`want`, `need`) with treatments (`pill`, `medicine`, `water`, `doctor`) and reasons (`headache`, `fever`, `pain`):
     - `["I", "want", "pill", "headache"]` $\rightarrow$ *"I want a pill for my headache."* (or press `TAB` for *"I have a headache, so I want a pill."*)
     - `["I", "feel", "headache", "want", "pill"]` $\rightarrow$ *"I feel a headache, and I want a pill."*
     - `["hello", "doctor", "I", "feel", "dizzy", "need", "water"]` $\rightarrow$ *"Hello, Doctor — I feel dizzy, and I need water."*
   - **Alternative Phrasings (`TAB` Key)**: Press `TAB` at any time to cycle through smooth alternative English interpretations of the current signed words.

---

## 🚀 Quick Start (Run in VS Code)

### 1. Configuration (Roboflow API Key)
Copy `.env.example` to `.env` and insert your Roboflow API key:
```powershell
Copy-Item .env.example .env
```
Open `.env` and set your key:
```env
ROBOFLOW_API_KEY="your_roboflow_api_key_here"
ROBOFLOW_API_URL="https://serverless.roboflow.com"
```
*(If no API key is provided, the app will still run smoothly using the offline 50-sign Medical Model and 18-sign local kinematic Daily ASL model).*

### 2. Launch Application
Open `asl_main.py` in VS Code and click **Run Python File**, or run from the terminal:

```powershell
& "c:\Users\atiwa\OneDrive\Desktop\ASLMed\.venv\Scripts\python.exe" "c:\Users\atiwa\OneDrive\Desktop\ASLMed\asl_main.py"
```

### Keyboard Controls (`asl_main.py`)
| Key | Action |
| :--- | :--- |
| **`SPACE`** | Manual start / stop sign capture |
| **`TAB`** | Cycle alternative natural sentence phrasings |
| **`BACKSPACE`** | Delete the last signed word |
| **`C`** or **`ENTER`** | Clear the current word list and start a new sentence |
| **`S`** | Toggle live MediaPipe hand & pose skeleton overlay |
| **`[` / `]`** | Decrease / increase Medical vs. Daily model priority (`alpha`) |
| **`Q` / `ESC`** | Quit the application |

---

## 🛠️ Training & Dataset CLI (`asl_sign_train.py`)

```powershell
# 1. Extract MediaPipe landmarks from video clips (Classes 1-50)
python asl_sign_train.py extract --data dataset_2 --max_cls 50

# 2. Train final 50-class BiGRU+Attention ensemble with bilateral hand augmentation
python asl_sign_train.py train --final --n_cls 50 --arch bigru --bilateral --epochs 80 --ensemble 2

# 3. Record custom sign clips via webcam
python asl_sign_train.py record --name Me --n_cls 50
```

---

## 📂 Project Structure

```text
ASLMed/
├── asl_main.py                # Main Desktop Application (Medical + Daily + Roboflow + Interpreter HUD)
├── asl_sentence.py            # Expert AI Sentence Interpreter (realise & realise_variants)
├── asl_model.py               # Neural Architectures (BiGRU, Transformer, Hybrid) & Bilateral Flip
├── asl_dataset.py             # 50-Class Vocabulary, Landmark Extractor & Anti-Flicker Smoother
├── asl_sign_train.py          # Unified CLI for Extract, Record, Train, CV, and Live testing
├── asl_sign_model.pt          # Trained 50-Class PyTorch Ensemble Weights
├── asl_sign_model.torchscript # Fast CPU TorchScript Deployment Model
├── cache_landmarks/           # 1,046 Cached MediaPipe Landmark Sequences (.npy)
└── dataset_2/                 # 50-Class Video Dataset (.mp4)
```
