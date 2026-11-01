import numpy as np
import pytest

from src.asl_dataset import (
    MEANINGS,
    LandmarkSmoother,
    clean_hands,
    normalize,
    trim,
    augment,
    parse_filename,
    hand_speed,
)


class TestASLDatasetUtils:
    def test_meanings_vocabulary(self):
        assert len(MEANINGS) == 50
        assert MEANINGS[1] == "Body"
        assert MEANINGS[40] == "Stomach"
        assert MEANINGS[41] == "High blood pressure"
        assert MEANINGS[50] == "Call ambulance"

    def test_parse_filename(self):
        res1 = parse_filename("Me_1_a.mp4", max_cls=50)
        assert res1 == ("Me", 1, "a")

        res2 = parse_filename("SubjectA_50_c.npy", max_cls=50)
        assert res2 == ("SubjectA", 50, "c")

        # Exceeds max_cls
        assert parse_filename("Me_51_a.mp4", max_cls=50) is None

        # Invalid pattern
        assert parse_filename("random_video.mp4") is None

    def test_landmark_smoother(self):
        smoother = LandmarkSmoother(alpha=0.6)
        frame1 = np.ones((67, 3), dtype=np.float32)
        out1 = smoother.update(frame1)
        np.testing.assert_array_equal(out1, frame1)

        frame2 = np.full((67, 3), 2.0, dtype=np.float32)
        out2 = smoother.update(frame2)
        # Expected: 0.6 * 2.0 + 0.4 * 1.0 = 1.6
        expected = np.full((67, 3), 1.6, dtype=np.float32)
        np.testing.assert_allclose(out2, expected, rtol=1e-5)

        smoother.reset()
        assert smoother.prev is None

    def test_clean_hands(self):
        arr = np.zeros((5, 67, 3), dtype=np.float32)
        # Set shoulders 11 and 12 with normal distance
        arr[:, 11, :2] = [-0.2, 0.0]
        arr[:, 12, :2] = [0.2, 0.0]
        # Shoulder width = 0.4

        # Normal hand
        arr[:, 25, :2] = [0.0, 0.0]  # wrist
        arr[:, 37, :2] = [0.0, 0.2]  # middle tip (span = 0.2 / 0.4 = 0.5 -> plausible)

        cleaned = clean_hands(arr, need_shoulders=True)
        assert not np.isnan(cleaned[:, 25, 0]).any()

    def test_normalize(self):
        raw = np.zeros((10, 67, 3), dtype=np.float32)
        raw[:, 11, :2] = [0.4, 0.5]
        raw[:, 12, :2] = [0.6, 0.5]
        # Midpoint is [0.5, 0.5], distance = 0.2

        norm = normalize(raw)
        assert norm.shape == raw.shape
        assert norm.dtype == np.float32
        # Midpoint of shoulders in normalized coordinates should be near 0
        mid = (norm[:, 11, :2] + norm[:, 12, :2]) / 2.0
        np.testing.assert_allclose(mid, 0.0, atol=1e-5)

    def test_trim(self):
        seq = np.full((20, 67, 3), np.nan, dtype=np.float32)
        # Hands only present in frames 5 to 12
        seq[5:13, 25, 0] = 0.5
        trimmed = trim(seq, pad=2)
        # Expected frames: max(5 - 2, 0) = 3 to min(12 + 2 + 1, 20) = 15
        assert len(trimmed) == 12

    def test_augment(self):
        seq = np.random.randn(15, 67, 3).astype(np.float32)
        augmented = augment(seq, allow_bilateral=True)
        assert augmented.shape == seq.shape

    def test_hand_speed(self):
        f1 = np.zeros((67, 3), dtype=np.float32)
        f2 = np.zeros((67, 3), dtype=np.float32)
        # Shoulders
        f1[11:13, :2] = [[-0.2, 0.0], [0.2, 0.0]]
        f2[11:13, :2] = [[-0.2, 0.0], [0.2, 0.0]]

        # Hand moves
        f1[25:46, :2] = 0.0
        f2[25:46, :2] = 0.1

        spd = hand_speed(f1, f2, dt=0.033)
        assert spd > 0.0
