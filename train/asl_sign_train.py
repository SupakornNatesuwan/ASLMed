"""
ASL Medical Sign Trainer & Webcam Recognizer (v3 - Clinical & Production Edition)

Key Upgrades:
- Supports 40 Core Signs & 50 Full Signs (including dataset_2 41..50)
- Architectures: BiGRU+Attention, Temporal Transformer, Hybrid 1D-CNN+BiGRU
- Bilateral (Left/Right Hand) Data Augmentation for ambidextrous robustness
- Temporal anti-flicker smoothing for webcam noise reduction
- Automatic ONNX Export for ultra-fast, low-latency CPU inference
- Interactive Dataset Recorder / Visual Sign Studio

Usage:
    python asl_sign_train.py extract --data dataset_2 --max_cls 50
    python asl_sign_train.py record  --name Me --n_cls 40
    python asl_sign_train.py train   --arch bigru --bilateral --export_onnx
    python asl_sign_train.py train   --final --arch bigru --export_onnx
    python asl_sign_train.py cv      --arch bigru
    python asl_sign_train.py live    --onnx
"""

import argparse
from collections import Counter, deque
from pathlib import Path
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.asl_dataset import (
    CACHE_DIR, MEANINGS, ASLDataset, LandmarkSmoother, LatestFrame,
    clean_hands, hand_speed, landmarks_from_bgr, load_landmarks_cache,
    make_holistic, normalize, parse_filename, prepare_sequence, resample, trim
)
from src.asl_model import (
    D_IN, MODEL_PATH, ONNX_PATH, SEQ_LEN_DEFAULT, TTA_CROPS,
    ONNXPredictor, TorchScriptPredictor, build_model, export_to_onnx,
    export_to_torchscript, get_device, model_from_state, to_features
)


# ============================ Landmark Extraction ============================
def cmd_extract(args):
    CACHE_DIR.mkdir(exist_ok=True)
    files = sorted(p for p in Path(args.data).rglob("*") if p.suffix.lower() == ".mp4")
    print(f"Found {len(files)} mp4 video clips in {args.data}")
    extracted, skipped = 0, 0

    with make_holistic() as hol:
        for i, p in enumerate(files, 1):
            info = parse_filename(p, max_cls=args.max_cls)
            if info is None:
                skipped += 1
                continue
            out_file = CACHE_DIR / f"{p.stem}.npy"
            if out_file.exists() and not args.force:
                continue

            cap = cv2.VideoCapture(str(p))
            frames = []
            smoother = LandmarkSmoother(alpha=0.7)
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                if bgr.shape[1] > 640:
                    scale = 640.0 / bgr.shape[1]
                    bgr = cv2.resize(bgr, None, fx=scale, fy=scale)
                lm = landmarks_from_bgr(bgr, hol)
                frames.append(smoother.update(lm))
            cap.release()

            if not frames:
                print(f"  Warning: could not read frames from {p.name}")
                continue

            arr = np.stack(frames)
            seen = (~np.isnan(arr[:, 25, 0]) | ~np.isnan(arr[:, 46, 0])).mean()
            np.save(out_file, arr)
            extracted += 1
            print(f"[{i}/{len(files)}] {p.name}: {len(arr)} frames (hands in {seen:.0%})")

    print(f"Extraction complete: {extracted} new files saved to cache, {skipped} files skipped.")


# ============================ Model Training =================================
def evaluate(model, ds, dev):
    model.eval()
    all_probs, all_targets = [], []
    with torch.no_grad():
        for x, y in DataLoader(ds, batch_size=64, shuffle=False):
            logits = model(x.to(dev))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_targets.extend(y.tolist())
    return np.concatenate(all_probs), np.array(all_targets)


def train_one(tr, va, epochs, seed, arch="bigru", n_cls=40, bilateral=True, verbose=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = get_device()
    
    tl = DataLoader(
        ASLDataset(tr, train=True, allow_bilateral=bilateral),
        batch_size=32,
        shuffle=True,
        drop_last=len(tr) > 32
    )
    vds = ASLDataset(va, train=False, allow_bilateral=False) if va else None
    
    model = build_model(arch=arch, d_in=D_IN, n_cls=n_cls).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=epochs * len(tl))
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()

        if verbose and (ep == 1 or ep % 10 == 0 or ep == epochs):
            msg = f"epoch {ep:3d}/{epochs} | loss {tot / len(tl):.3f}"
            if vds:
                P, Y = evaluate(model, vds, dev)
                msg += f" | val acc {(P.argmax(1) == Y).mean():.1%}"
            print(msg)

    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def cmd_train(args):
    data = load_landmarks_cache(CACHE_DIR, max_cls=args.n_cls)
    if not data:
        raise SystemExit("No cached landmarks found. Run: python asl_sign_train.py extract --data dataset_2")
    
    signers = sorted({d[2] for d in data})
    print(f"Loaded {len(data)} clips across {len(signers)} signers: {signers} | Feature dimension: {D_IN}")
    
    val_signers = [] if args.final else args.val_signers
    for s in val_signers:
        if s not in signers:
            print(f"Note: validation signer '{s}' not present in cached clips.")
            
    tr = [d for d in data if d[2] not in val_signers]
    va = [d for d in data if d[2] in val_signers]
    print(f"Training clips: {len(tr)} | Validation clips: {len(va)} | Device: {get_device()}")
    print(f"Architecture: {args.arch} | Bilateral Augmentation: {args.bilateral} | Ensemble size: {args.ensemble}")

    states, probs = [], []
    for k in range(args.ensemble):
        print(f"\n--- Training Model {k + 1}/{args.ensemble} ---")
        st = train_one(
            tr, va, args.epochs, seed=k, arch=args.arch,
            n_cls=args.n_cls, bilateral=args.bilateral,
            verbose=(k == 0 or not va)
        )
        states.append(st)
        if va:
            eval_model = build_model(arch=args.arch, d_in=D_IN, n_cls=args.n_cls)
            eval_model.load_state_dict(st)
            P, Y = evaluate(eval_model, ASLDataset(va, train=False, allow_bilateral=False), "cpu")
            probs.append(P)
            print(f"Model {k + 1} validation accuracy: {(P.argmax(1) == Y).mean():.1%}")

    if va:
        ens_probs = np.mean(probs, axis=0)
        ens_preds = ens_probs.argmax(1)
        acc = (ens_preds == Y).mean()
        print(f"\n>>> Ensemble Accuracy on unseen signers {val_signers}: {acc:.1%} <<<")
        conf = Counter((y, p) for y, p in zip(Y, ens_preds) if y != p)
        if conf:
            print("Top confused signs:")
            for (y, p), count in conf.most_common(6):
                name_y = MEANINGS.get(y + 1, f"Sign {y+1}")
                name_p = MEANINGS.get(p + 1, f"Sign {p+1}")
                print(f"  {name_y} -> misclassified as {name_p} ({count}x)")

    # Save PyTorch Checkpoint
    checkpoint = {
        "states": states,
        "d_in": D_IN,
        "seq_len": SEQ_LEN_DEFAULT,
        "n_cls": args.n_cls,
        "arch": args.arch
    }
    torch.save(checkpoint, MODEL_PATH)
    print(f"Checkpoint saved -> {MODEL_PATH}")

    # Export high-speed TorchScript & ONNX models
    export_model = build_model(arch=args.arch, d_in=D_IN, n_cls=args.n_cls)
    export_model.load_state_dict(states[0])
    ts_path = Path(MODEL_PATH).with_suffix(".torchscript")
    export_to_torchscript(export_model, ts_path, seq_len=SEQ_LEN_DEFAULT, d_in=D_IN)

    if args.export_onnx:
        export_to_onnx(export_model, ONNX_PATH, seq_len=SEQ_LEN_DEFAULT, d_in=D_IN)


def cmd_cv(args):
    data = load_landmarks_cache(CACHE_DIR, max_cls=args.n_cls)
    signers = sorted({d[2] for d in data})
    print(f"Running Leave-One-Signer-Out Cross Validation across {len(signers)} signers...")
    accs = {}
    for s in signers:
        tr = [d for d in data if d[2] != s]
        va = [d for d in data if d[2] == s]
        st = train_one(tr, va, args.epochs, seed=0, arch=args.arch, n_cls=args.n_cls,
                       bilateral=args.bilateral, verbose=False)
        eval_model = build_model(arch=args.arch, d_in=D_IN, n_cls=args.n_cls)
        eval_model.load_state_dict(st)
        P, Y = evaluate(eval_model, ASLDataset(va, train=False, allow_bilateral=False), "cpu")
        acc = float((P.argmax(1) == Y).mean())
        accs[s] = acc
        print(f"Hold-out signer {s:10s} : {acc:.1%}")
    print(f"\nMean Leave-One-Signer-Out Accuracy: {np.mean(list(accs.values())):.1%}")


# ============================ Dataset Recording ==============================
def cmd_record(args):
    out = Path(args.data)
    out.mkdir(exist_ok=True)
    todo = [(n, t) for n in range(1, args.n_cls + 1) for t in "abc"
            if not (out / f"{args.name}_{n}_{t}.mp4").exists()]
    if not todo:
        print(f"All signs 1..{args.n_cls} already recorded for {args.name}")
        return

    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    i, mode, t0, writer = 0, "wait", 0.0, None

    print("\n--- Visual Sign Studio ---")
    print("Press SPACE to start 2-second countdown, then perform sign.")
    print("Press S to skip, R to redo, Q or ESC to exit.\n")

    while i < len(todo):
        ok, frame = cap.read()
        if not ok:
            break
        n, t = todo[i]
        now = time.time()
        view = cv2.flip(frame, 1)
        h, w = view.shape[:2]
        meaning = MEANINGS.get(n, f"Sign {n}")
        head = f"[{i + 1}/{len(todo)}] Sign {n}: {meaning}  (Take {t.upper()})"
        col = (255, 255, 255)

        if mode == "wait":
            sub = "SPACE = Record | S = Skip | R = Redo Previous | Q = Quit"
        elif mode == "count":
            left = 2.0 - (now - t0)
            if left <= 0:
                path = out / f"{args.name}_{n}_{t}.mp4"
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
                mode, t0 = "rec", now
            sub = f"Get ready... starting in {int(max(left, 0)) + 1}"
            col = (0, 255, 255)
        else:
            writer.write(frame)
            sub = "RECORDING - Sign now, then lower hands to sides"
            col = (0, 0, 255)
            if now - t0 >= args.seconds:
                writer.release()
                writer, mode = None, "wait"
                i += 1

        cv2.rectangle(view, (0, 0), (w, 90), (25, 25, 25), -1)
        cv2.putText(view, head, (15, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (80, 255, 120), 2)
        cv2.putText(view, sub, (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)
        cv2.imshow("ASLMed - Sign Studio Recorder", view)
        
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        elif k == ord(" ") and mode == "wait":
            mode, t0 = "count", now
        elif k == ord("s") and mode == "wait":
            i += 1
        elif k == ord("r") and mode == "wait":
            i = max(i - 1, 0)

    if writer:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nRecording finished. Extract landmarks using:")
    print(f"  python asl_sign_train.py extract --data {args.data}")


# ============================ Live Webcam Recognizer =========================
def cmd_live(args):
    ts_path = Path(MODEL_PATH).with_suffix(".torchscript")
    if args.onnx and Path(ONNX_PATH).exists():
        print(f"Using ONNX Runtime engine -> {ONNX_PATH}")
        predictor = ONNXPredictor(ONNX_PATH)
        engine_type = "ONNX (CPU)"
    elif ts_path.exists():
        print(f"Using TorchScript high-speed engine -> {ts_path}")
        predictor = TorchScriptPredictor(ts_path)
        engine_type = "TorchScript (CPU)"
    else:
        if not Path(MODEL_PATH).exists():
            raise SystemExit(f"Model file {MODEL_PATH} not found. Run: python asl_sign_train.py train")
        ck = torch.load(MODEL_PATH, map_location="cpu")
        arch = ck.get("arch", "bigru")
        n_cls = ck.get("n_cls", 40)
        models = []
        for st in ck["states"]:
            m = build_model(arch=arch, d_in=D_IN, n_cls=n_cls)
            m.load_state_dict(st)
            m.eval()
            models.append(m)
        predictor = None
        engine_type = "PyTorch Ensemble"

    def classify_body(seg):
        x = trim(normalize(np.stack(seg), need_shoulders=True))
        crops = [to_features(resample(x, False, c)) for c in TTA_CROPS]
        if args.bilateral_tta:
            # Also test horizontally flipped sequence for ambidextrous robustness
            from src.asl_model import flip_bilateral
            x_flip = flip_bilateral(x)
            crops += [to_features(resample(x_flip, False, c)) for c in TTA_CROPS]
            
        feats = np.stack(crops)
        if predictor is not None:
            p = predictor.predict_probs(feats).mean(axis=0)
        else:
            with torch.no_grad():
                tensor_feats = torch.from_numpy(feats)
                p = torch.stack([torch.softmax(m(tensor_feats), 1) for m in models]).mean((0, 1)).numpy()
        order = p.argsort()[::-1][:3]
        return [(MEANINGS.get(i + 1, f"Sign {i+1}"), float(p[i])) for i in order]

    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    hol = make_holistic(args.complexity)
    smoother = LandmarkSmoother(alpha=0.65)
    
    seg, seg_t, active = [], [], False
    preroll = deque(maxlen=6)
    last_hand = last_move = 0.0
    result, result_until, history = None, 0.0, []

    print("\nStarting ASLMed Live Recognizer... (Press Q to quit)\n")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()
        small = cv2.resize(frame, (480, 360))
        raw_lm = landmarks_from_bgr(small, hol)
        lm = smoother.update(clean_hands(raw_lm[None], need_shoulders=True)[0])
        hands_now = not (np.isnan(lm[25, 0]) and np.isnan(lm[46, 0]))

        if hands_now:
            last_hand = now

        if not active:
            preroll.append((now, lm))
            if hands_now:
                active = True
                seg = [x for _, x in preroll]
                seg_t = [t for t, _ in preroll]
                last_move = now
                result_until = 0.0
        else:
            seg.append(lm)
            seg_t.append(now)
            if hands_now:
                last_move = now
            finished = (now - last_hand > 0.40 or now - last_move > 0.65 or now - seg_t[0] > 4.5)
            if finished:
                body = seg
                dur = seg_t[-1] - seg_t[0]
                if len(body) >= 8 and dur >= 0.4:
                    top = classify_body(body)
                    ok_pred = top[0][1] >= args.threshold and (top[0][1] - top[1][1]) >= args.margin
                    result, result_until = (top, ok_pred), now + 2.5
                    if ok_pred:
                        history = (history + [top[0][0]])[-6:]
                active, seg, seg_t = False, [], []
                preroll.clear()

        # UI Overlay
        view = cv2.flip(frame, 1)
        h, w = view.shape[:2]
        cv2.rectangle(view, (0, h - 110), (w, h), (20, 20, 20), -1)
        if active:
            cv2.putText(view, "Signing...", (20, h - 65), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        elif result and now < result_until:
            top, ok_pred = result
            txt = top[0][0] if ok_pred else f"Not sure ({top[0][0]}?)"
            cv2.putText(view, txt, (20, h - 65), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        (80, 255, 120) if ok_pred else (0, 165, 255), 3)
            alt = "   ".join(f"{nm}: {c * 100:.0f}%" for nm, c in top)
            cv2.putText(view, alt, (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        else:
            cv2.putText(view, "Ready - sign in camera view", (20, h - 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (160, 160, 160), 2)

        if history:
            cv2.putText(view, "History: " + ", ".join(history), (20, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 200, 120), 1)

        engine_str = f"Engine: {engine_type}"
        cv2.putText(view, engine_str, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
        cv2.imshow("ASLMed Live Recognizer", view)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


# ============================ CLI Parser =====================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ASLMed Sign Language Recognition & Training")
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="Extract MediaPipe landmarks from video clips")
    e.add_argument("--data", default="dataset_2", help="Folder with training mp4 clips")
    e.add_argument("--max_cls", type=int, default=40, help="Maximum class ID to extract (40 or 50)")
    e.add_argument("--force", action="store_true", help="Re-extract even if cached")
    e.set_defaults(fn=cmd_extract)

    r = sub.add_parser("record", help="Record webcam clips for new signs")
    r.add_argument("--name", required=True, help="Your signer name (e.g. Me)")
    r.add_argument("--data", default="dataset_2")
    r.add_argument("--n_cls", type=int, default=40)
    r.add_argument("--seconds", type=float, default=3.5)
    r.add_argument("--cam", type=int, default=0)
    r.set_defaults(fn=cmd_record)

    t = sub.add_parser("train", help="Train sign recognition neural network")
    t.add_argument("--val_signers", nargs="*", default=["Joel", "Sue_mum"])
    t.add_argument("--epochs", type=int, default=120)
    t.add_argument("--ensemble", type=int, default=3, help="Number of models to ensemble")
    t.add_argument("--arch", choices=["bigru", "transformer", "hybrid"], default="bigru")
    t.add_argument("--n_cls", type=int, default=40)
    t.add_argument("--bilateral", action="store_true", default=True, help="Enable left/right hand flip augmentation")
    t.add_argument("--export_onnx", action="store_true", default=True, help="Export to ONNX after training")
    t.add_argument("--final", action="store_true", help="Train on all signers without holdout validation")
    t.set_defaults(fn=cmd_train)

    c = sub.add_parser("cv", help="Leave-one-signer-out cross validation")
    c.add_argument("--epochs", type=int, default=80)
    c.add_argument("--arch", choices=["bigru", "transformer", "hybrid"], default="bigru")
    c.add_argument("--n_cls", type=int, default=40)
    c.add_argument("--bilateral", action="store_true", default=True)
    c.set_defaults(fn=cmd_cv)

    l = sub.add_parser("live", help="Webcam recognition demo")
    l.add_argument("--cam", type=int, default=0)
    l.add_argument("--complexity", type=int, default=1, choices=[0, 1, 2])
    l.add_argument("--threshold", type=float, default=0.5)
    l.add_argument("--margin", type=float, default=0.12)
    l.add_argument("--n_cls", type=int, default=40)
    l.add_argument("--onnx", action="store_true", default=True, help="Use ONNX Runtime engine if available")
    l.add_argument("--bilateral_tta", action="store_true", default=True, help="Test both original and flipped hands")
    l.set_defaults(fn=cmd_live)

    args = ap.parse_args()
    args.fn(args)
