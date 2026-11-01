"""
ASLMed Deep Learning Model Architectures, Bilateral Augmentation, and ONNX Runtime.

Supported Models:
- Net (BiGRU + Attention) : Default robust recurrent model (backward-compatible)
- SignTransformerNet     : Multi-head temporal self-attention transformer
- HybridConvGRU          : 1D-CNN temporal feature extractor + BiGRU

Features:
- Bilateral Hand/Pose Symmetry Augmentation (Left/Right hand invariance)
- Dynamic TTA (Test-Time Augmentation)
- ONNX Export and low-latency ONNX Runtime engine
"""

import math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

N_CLS_DEFAULT = 40
SEQ_LEN_DEFAULT = 48
N_LM = 25 + 21 + 21  # 25 pose + 21 left hand + 21 right hand

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_DIR / "models"
MODEL_PATH = str(MODELS_DIR / "asl_sign_model.pt")
ONNX_PATH = str(MODELS_DIR / "asl_sign_model.onnx")
TTA_CROPS = [(0.0, 0.0), (0.06, 0.0), (0.0, 0.06), (0.06, 0.06)]

# Body & Hand Anchors for Relational Features
ANCHORS = [0, 2, 5, 9, 10, 11, 12, 13, 14, 23, 24]  # nose, eyes, mouth, shoulders, elbows, hips
HAND_PTS = [0, 8, 12]  # wrist, index tip, middle tip

# Pose Symmetric Pairs for Horizontal Mirroring (Bilateral Flip)
POSE_PAIRS = [
    (1, 4), (2, 5), (3, 6),   # eyes
    (7, 8),                   # ears
    (9, 10),                  # mouth corners
    (11, 12),                 # shoulders
    (13, 14),                 # elbows
    (15, 16),                 # wrists
    (17, 18),                 # pinkies
    (19, 20),                 # index fingers
    (21, 22),                 # thumbs
    (23, 24),                 # hips
]


# ============================ Bilateral Hand Flip ============================
def flip_bilateral(seq):
    """
    Mirror the landmark sequence horizontally:
    1. Invert the X coordinate: x -> -x
    2. Swap left pose points with right pose points
    3. Swap entire Left Hand (25:46) with Right Hand (46:67)
    
    Allows left-handed signers to be recognized as accurately as right-handed signers.
    """
    flipped = seq.copy()
    # Invert X axis
    flipped[..., 0] = -flipped[..., 0]
    
    # Swap symmetric pose landmarks
    for left_idx, right_idx in POSE_PAIRS:
        temp = flipped[:, left_idx].copy()
        flipped[:, left_idx] = flipped[:, right_idx]
        flipped[:, right_idx] = temp
        
    # Swap Left Hand (25:46) and Right Hand (46:67)
    temp_lh = flipped[:, 25:46].copy()
    flipped[:, 25:46] = flipped[:, 46:67]
    flipped[:, 46:67] = temp_lh
    
    return flipped


def resample(seq, train=False, crop=(0.0, 0.0), seq_len=SEQ_LEN_DEFAULT):
    """Resample sequence dynamically to a fixed number of frames."""
    n = len(seq)
    if train:  # random crop + jittered speed
        lo = np.random.uniform(0, 0.1 * n)
        hi = n - 1 - np.random.uniform(0, 0.1 * n)
        idx = np.sort(np.random.uniform(lo, max(hi, lo + 1), seq_len)).round()
    else:
        idx = np.linspace(crop[0] * (n - 1), (n - 1) * (1 - crop[1]), seq_len).round()
    return seq[np.clip(idx.astype(int), 0, n - 1)]


# ============================ Feature Extraction ============================
def hand_shape(h):
    """Hand pose relative to its own wrist, scaled by hand size."""
    with np.errstate(invalid="ignore", divide="ignore"):
        size = np.linalg.norm(h[:, 9] - h[:, 0], axis=-1)
        size = np.where(size < 1e-3, np.nan, size)
        return (h - h[:, :1]) / size[:, None, None]


def rel_feats(seq):
    """Vectors from key hand points to face/body anchors."""
    T = len(seq)
    pose = seq[:, ANCHORS, :2]
    outs = []
    for s in (25, 46):
        pts = seq[:, [s + i for i in HAND_PTS], :2]
        d = pts[:, :, None, :] - pose[:, None, :, :]
        outs.append(d.reshape(T, -1))
    return np.concatenate(outs, 1)


def to_features(seq):
    """Compute dense feature vector per frame (shape, positions, relations, velocities)."""
    T = len(seq)
    pos = seq.reshape(T, -1)
    shape = np.concatenate([
        hand_shape(seq[:, 25:46]).reshape(T, -1),
        hand_shape(seq[:, 46:67]).reshape(T, -1)
    ], 1)
    rel = rel_feats(seq)
    base = np.concatenate([pos, rel], 1)
    vel = np.diff(base, axis=0, prepend=base[:1])
    mask = np.stack([
        ~np.isnan(seq[:, 0, 0]),
        ~np.isnan(seq[:, 25, 0]),
        ~np.isnan(seq[:, 46, 0])
    ], 1).astype(np.float32)
    out = np.nan_to_num(np.concatenate([pos, shape, rel, vel, mask], 1))
    return np.clip(out, -6.0, 6.0).astype(np.float32)


# Compute Input Feature Dimension
D_IN = to_features(np.zeros((3, N_LM, 3), np.float32)).shape[1]


# ============================ Neural Models ==================================
class Net(nn.Module):
    """2-layer Bidirectional GRU with Attention pooling (Fully backward-compatible)."""
    def __init__(self, d_in=None, n_cls=N_CLS_DEFAULT, h=192, p=0.3):
        super().__init__()
        self.d_in = d_in or D_IN
        self.n_cls = n_cls
        self.inp = nn.Sequential(
            nn.Linear(self.d_in, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Dropout(p)
        )
        self.gru = nn.GRU(h, h, num_layers=2, batch_first=True, bidirectional=True, dropout=p)
        self.att = nn.Linear(2 * h, 1)
        self.head = nn.Sequential(
            nn.Dropout(p),
            nn.Linear(2 * h, n_cls)
        )

    def forward(self, x):
        # x: (B, T, D_IN)
        z, _ = self.gru(self.inp(x))
        w = torch.softmax(self.att(z), 1)
        pooled = (w * z).sum(1)
        return self.head(pooled)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class SignTransformerNet(nn.Module):
    """Multi-Head Self-Attention Temporal Transformer for sign recognition."""
    def __init__(self, d_in=None, n_cls=N_CLS_DEFAULT, d_model=192, nhead=4, num_layers=3, p=0.25):
        super().__init__()
        self.d_in = d_in or D_IN
        self.n_cls = n_cls
        self.inp = nn.Sequential(
            nn.Linear(self.d_in, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(p)
        )
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=p, activation="gelu", batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.att = nn.Linear(d_model, 1)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(p),
            nn.Linear(d_model, n_cls)
        )

    def forward(self, x):
        h = self.pos_enc(self.inp(x))
        z = self.transformer(h)
        w = torch.softmax(self.att(z), 1)
        pooled = (w * z).sum(1)
        return self.head(pooled)


class HybridConvGRU(nn.Module):
    """1D Temporal Convolutions + BiGRU + Attention for fine finger transitions."""
    def __init__(self, d_in=None, n_cls=N_CLS_DEFAULT, h=192, p=0.3):
        super().__init__()
        self.d_in = d_in or D_IN
        self.n_cls = n_cls
        self.conv1 = nn.Conv1d(self.d_in, h, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm1d(h)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv1d(h, h, kernel_size=5, padding=2)
        self.norm2 = nn.BatchNorm1d(h)
        self.act2 = nn.GELU()
        self.drop = nn.Dropout(p)
        self.gru = nn.GRU(h, h, num_layers=2, batch_first=True, bidirectional=True, dropout=p)
        self.att = nn.Linear(2 * h, 1)
        self.head = nn.Sequential(
            nn.Dropout(p),
            nn.Linear(2 * h, n_cls)
        )

    def forward(self, x):
        # x: (B, T, D_IN) -> conv expects (B, D_IN, T)
        t = x.transpose(1, 2)
        c1 = self.act1(self.norm1(self.conv1(t)))
        c2 = self.drop(self.act2(self.norm2(self.conv2(c1))))
        c_out = c2.transpose(1, 2)
        z, _ = self.gru(c_out)
        w = torch.softmax(self.att(z), 1)
        pooled = (w * z).sum(1)
        return self.head(pooled)


def build_model(arch="bigru", d_in=None, n_cls=N_CLS_DEFAULT):
    if arch == "transformer":
        return SignTransformerNet(d_in=d_in, n_cls=n_cls)
    elif arch == "hybrid":
        return HybridConvGRU(d_in=d_in, n_cls=n_cls)
    return Net(d_in=d_in, n_cls=n_cls)


def get_device():
    """Detect acceleration device: MPS (Apple Silicon), CUDA, or fallback to CPU."""
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    return "mps" if mps and mps.is_available() else "cpu"


def model_from_state(state, dev=None):
    """Load model dynamically from state_dict (auto-detects 40/50 classes & architecture)."""
    if dev is None:
        dev = "cpu"
    n_cls = 40
    for k, v in state.items():
        if k.endswith("head.1.weight") or k.endswith("head.2.weight"):
            n_cls = v.shape[0]
            break
    arch = "bigru"
    if any(k.startswith("transformer") for k in state):
        arch = "transformer"
    elif any(k.startswith("conv1") for k in state):
        arch = "hybrid"
    m = build_model(arch=arch, d_in=D_IN, n_cls=n_cls).to(dev)
    m.load_state_dict(state)
    return m.eval()


# ============================ High-Performance Inference =====================
def export_to_torchscript(model, ts_path, seq_len=SEQ_LEN_DEFAULT, d_in=None):
    """Export model to standalone TorchScript format for low-latency CPU inference."""
    model.eval()
    dev = next(model.parameters()).device
    dummy = torch.randn(1, seq_len, d_in or D_IN, dtype=torch.float32, device=dev)
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy)
        traced.save(str(ts_path))
    print(f"Exported TorchScript model -> {ts_path}")


class TorchScriptPredictor:
    """Zero-dependency high-speed CPU inference engine using TorchScript."""
    def __init__(self, ts_path):
        self.model = torch.jit.load(str(ts_path), map_location="cpu")
        self.model.eval()

    def predict_probs(self, feature_batch_np):
        """feature_batch_np: (B, T, D_IN) -> probabilities (B, N_CLS)"""
        with torch.no_grad():
            tensor_in = torch.from_numpy(feature_batch_np.astype(np.float32))
            logits = self.model(tensor_in)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs


def export_to_onnx(model, onnx_path, seq_len=SEQ_LEN_DEFAULT, d_in=None):
    """Export PyTorch sign model to ONNX format (if onnx is installed)."""
    try:
        model.eval()
        dev = next(model.parameters()).device
        dummy = torch.randn(1, seq_len, d_in or D_IN, dtype=torch.float32, device=dev)
        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["landmarks_features"],
            output_names=["logits"],
            dynamo=False
        )
        print(f"Exported ONNX model -> {onnx_path}")
    except Exception as e:
        print(f"Note: ONNX export skipped ({e}). TorchScript is available as high-speed engine.")


class ONNXPredictor:
    """Ultra-fast ONNX Runtime inference engine."""
    def __init__(self, onnx_path):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4
        self.session = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict_probs(self, feature_batch_np):
        """feature_batch_np: (B, T, D_IN) -> probabilities (B, N_CLS)"""
        logits = self.session.run([self.output_name], {self.input_name: feature_batch_np.astype(np.float32)})[0]
        e_x = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)


if __name__ == "__main__":
    net = build_model("bigru")
    sample_input = torch.randn(2, SEQ_LEN_DEFAULT, D_IN)
    out = net(sample_input)
    print("Net output shape:", out.shape)
    assert out.shape == (2, N_CLS_DEFAULT)
    print("Model architecture verification passed.")
