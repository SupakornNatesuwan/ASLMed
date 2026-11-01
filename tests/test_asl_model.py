from pathlib import Path
import numpy as np
import pytest
import torch

from src.asl_model import (
    ANCHORS,
    D_IN,
    HAND_PTS,
    N_CLS_DEFAULT,
    N_LM,
    POSE_PAIRS,
    SEQ_LEN_DEFAULT,
    HybridConvGRU,
    Net,
    SignTransformerNet,
    TorchScriptPredictor,
    build_model,
    flip_bilateral,
    hand_shape,
    rel_feats,
    resample,
    to_features,
)


class TestASLModel:
    def test_dimensions(self):
        assert D_IN == 795
        assert N_LM == 67
        assert SEQ_LEN_DEFAULT == 48

    def test_bigru_forward(self):
        model = Net(d_in=D_IN, n_cls=50)
        x = torch.randn(2, SEQ_LEN_DEFAULT, D_IN)
        out = model(x)
        assert out.shape == (2, 50)

    def test_transformer_forward(self):
        model = SignTransformerNet(d_in=D_IN, n_cls=50)
        x = torch.randn(2, SEQ_LEN_DEFAULT, D_IN)
        out = model(x)
        assert out.shape == (2, 50)

    def test_hybrid_forward(self):
        model = HybridConvGRU(d_in=D_IN, n_cls=50)
        x = torch.randn(2, SEQ_LEN_DEFAULT, D_IN)
        out = model(x)
        assert out.shape == (2, 50)

    def test_build_model_factory(self):
        m1 = build_model("bigru", d_in=D_IN, n_cls=40)
        assert isinstance(m1, Net)

        m2 = build_model("transformer", d_in=D_IN, n_cls=40)
        assert isinstance(m2, SignTransformerNet)

        m3 = build_model("hybrid", d_in=D_IN, n_cls=40)
        assert isinstance(m3, HybridConvGRU)

    def test_flip_bilateral(self):
        seq = np.zeros((10, N_LM, 3), dtype=np.float32)
        # Give distinct landmark values to check swap
        seq[:, 11, :] = [1.0, 2.0, 3.0]   # Left shoulder
        seq[:, 12, :] = [4.0, 5.0, 6.0]   # Right shoulder
        seq[:, 25, :] = [7.0, 8.0, 9.0]   # Left hand wrist
        seq[:, 46, :] = [10.0, 11.0, 12.0] # Right hand wrist

        flipped = flip_bilateral(seq)

        # X coordinate should be inverted:
        # Flipped left shoulder should have right shoulder's original values with inverted X
        assert np.isclose(flipped[0, 11, 0], -4.0)
        assert np.isclose(flipped[0, 11, 1], 5.0)

        # Flipped right shoulder should have left shoulder's original values with inverted X
        assert np.isclose(flipped[0, 12, 0], -1.0)
        assert np.isclose(flipped[0, 12, 1], 2.0)

        # Hand swap:
        # Left hand wrist (25) should now hold right hand wrist (46) values with inverted X
        assert np.isclose(flipped[0, 25, 0], -10.0)
        assert np.isclose(flipped[0, 46, 0], -7.0)

    def test_to_features_shape_and_bounds(self):
        seq = np.random.randn(20, N_LM, 3).astype(np.float32)
        feats = to_features(seq)
        assert feats.shape == (20, D_IN)
        assert np.all(feats >= -6.0) and np.all(feats <= 6.0)
        assert not np.isnan(feats).any()

    def test_resample(self):
        seq = np.random.randn(30, N_LM, 3).astype(np.float32)
        resampled_eval = resample(seq, train=False, seq_len=48)
        assert len(resampled_eval) == 48

        resampled_train = resample(seq, train=True, seq_len=48)
        assert len(resampled_train) == 48

    def test_checkpoint_and_torchscript(self):
        pt_path = Path("models/asl_sign_model.pt")
        ts_path = Path("models/asl_sign_model.torchscript")

        assert pt_path.exists(), "models/asl_sign_model.pt must exist"
        assert ts_path.exists(), "models/asl_sign_model.torchscript must exist"

        ck = torch.load(str(pt_path), map_location="cpu")
        assert ck["d_in"] == D_IN
        assert ck["n_cls"] == 50
        assert len(ck["states"]) >= 1

        predictor = TorchScriptPredictor(ts_path)
        dummy_input = np.random.randn(1, SEQ_LEN_DEFAULT, D_IN).astype(np.float32)
        probs = predictor.predict_probs(dummy_input)
        assert probs.shape == (1, 50)
        assert np.isclose(probs.sum(), 1.0, atol=1e-5)
