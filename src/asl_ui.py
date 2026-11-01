"""
ASLMed Modern Clinical UI & Glassmorphism Dashboard Engine
Renders high-contrast, beautiful, accessible clinical HUD:
- Doctor Clinical Dashboard (ESI Triage Badge, Facial Pain Grimace Gauge, Drug Safety Alerts)
- Patient Reassurance Mode (Dual-Screen high contrast, large typography, visual cards)
- Doctor-to-Patient Two-Way Communication Card Overlay
- Anti-aliased glowing skeletons, word chips, and intent prediction buttons.
"""

from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX

# Clinical Dark Palette (BGR format)
COLOR_BG_DARK = (20, 24, 30)
COLOR_CARD_BG = (30, 36, 46)
COLOR_CARD_BORDER = (55, 65, 80)
COLOR_TEXT_PRIMARY = (250, 252, 255)
COLOR_TEXT_MUTED = (160, 170, 185)
COLOR_TEXT_DIM = (110, 120, 135)

COLOR_ACCENT_CYAN = (235, 195, 55)     # Sky Cyan (RGB: 55, 195, 235)
COLOR_ACCENT_MINT = (110, 235, 130)    # Mint Emerald (RGB: 130, 235, 110)
COLOR_ACCENT_AMBER = (60, 180, 255)    # Warm Amber (RGB: 255, 180, 60)
COLOR_ACCENT_INDIGO = (245, 120, 165)  # Soft Purple / Indigo (RGB: 165, 120, 245)
COLOR_ACCENT_RED = (90, 90, 245)       # Coral Red

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]
POSE_CONNECTIONS = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]


def draw_rounded_rect(
    img: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = -1,
    radius: int = 8
) -> None:
    """Draw smooth rounded rectangle on image."""
    x1, y1 = pt1
    x2, y2 = pt2
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    w = x2 - x1
    h = y2 - y1
    r = min(radius, w // 2, h // 2)

    if r <= 2 or thickness > 0:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        return

    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
    cv2.circle(img, (x1 + r, y1 + r), r, color, -1, cv2.LINE_AA)
    cv2.circle(img, (x2 - r, y1 + r), r, color, -1, cv2.LINE_AA)
    cv2.circle(img, (x1 + r, y2 - r), r, color, -1, cv2.LINE_AA)
    cv2.circle(img, (x2 - r, y2 - r), r, color, -1, cv2.LINE_AA)


def draw_chip(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    bg_color: Tuple[int, int, int],
    text_color: Tuple[int, int, int] = COLOR_TEXT_PRIMARY,
    tag: Optional[str] = None,
    scale: float = 0.50,
) -> int:
    """Draw sleek chip badge with optional tag. Returns width."""
    main_txt = text.upper()
    (w_main, h_main), _ = cv2.getTextSize(main_txt, FONT, scale, 2)
    tag_w = 0
    if tag:
        (tag_w, _), _ = cv2.getTextSize(tag.upper(), FONT, scale * 0.75, 1)
        tag_w += 12

    total_w = w_main + tag_w + 22
    total_h = max(h_main + 14, 26)

    draw_rounded_rect(img, (x, y), (x + total_w, y + total_h), bg_color, thickness=-1, radius=6)
    draw_rounded_rect(img, (x, y), (x + total_w, y + total_h), COLOR_CARD_BORDER, thickness=1, radius=6)
    cv2.putText(img, main_txt, (x + 10, y + total_h - 8), FONT, scale, text_color, 2, cv2.LINE_AA)
    if tag:
        tag_x = x + w_main + 16
        cv2.putText(img, tag.upper(), (tag_x, y + total_h - 9), FONT, scale * 0.75, COLOR_TEXT_DIM, 1, cv2.LINE_AA)
    return total_w


def draw_neural_bar(
    img: np.ndarray,
    label: str,
    score: float,
    src: str,
    x: int,
    y: int,
    max_w: int,
    bar_h: int = 10,
    color: Tuple[int, int, int] = COLOR_ACCENT_CYAN,
) -> None:
    """Draws a sleek softmax probability bar with percentage and model badge."""
    lbl_txt = label.upper()[:12]
    cv2.putText(img, lbl_txt, (x, y + bar_h - 1), FONT, 0.32, COLOR_TEXT_PRIMARY, 1, cv2.LINE_AA)

    # Bar background
    bar_x = x + 85
    bar_w = max_w - 85 - 58
    if bar_w > 15:
        draw_rounded_rect(img, (bar_x, y), (bar_x + bar_w, y + bar_h), (25, 30, 42), -1, 3)
        # Filled bar
        fill_w = max(4, int(bar_w * float(np.clip(score, 0.0, 1.0))))
        draw_rounded_rect(img, (bar_x, y), (bar_x + fill_w, y + bar_h), color, -1, 3)

    # Score text + Source badge
    pct_txt = f"{int(score * 100)}%"
    cv2.putText(img, pct_txt, (x + max_w - 52, y + bar_h - 1), FONT, 0.32, color, 1, cv2.LINE_AA)
    src_tag = f"[{src.upper()[:4]}]"
    cv2.putText(img, src_tag, (x + max_w - 24, y + bar_h - 1), FONT, 0.28, COLOR_TEXT_DIM, 1, cv2.LINE_AA)


def draw_skeleton(view: np.ndarray, lm: np.ndarray) -> None:
    """Draw high-contrast glowing hand and body skeletons."""
    h, w = view.shape[:2]

    def pt(idx):
        x, y = lm[idx, 0], lm[idx, 1]
        if np.isnan(x) or np.isnan(y):
            return None
        return int((1.0 - x) * w), int(y * h)

    # Pose (Torso / Shoulders)
    for i, j in POSE_CONNECTIONS:
        p1, p2 = pt(i), pt(j)
        if p1 and p2:
            cv2.line(view, p1, p2, COLOR_ACCENT_AMBER, 2, cv2.LINE_AA)

    # Hands: Left Hand (Mint), Right Hand (Sky Cyan)
    hand_specs = [
        (25, COLOR_ACCENT_MINT, "L"),
        (46, COLOR_ACCENT_CYAN, "R")
    ]
    for base, col, _ in hand_specs:
        if np.isnan(lm[base, 0]):
            continue
        for a, b in HAND_CONNECTIONS:
            p1, p2 = pt(base + a), pt(base + b)
            if p1 and p2:
                cv2.line(view, p1, p2, col, 2, cv2.LINE_AA)
        for k in range(21):
            p = pt(base + k)
            if p:
                cv2.circle(view, p, 3, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(view, p, 4, col, 1, cv2.LINE_AA)


def wrap_text(text: str, scale: float, thick: int, max_w: int) -> List[str]:
    """Wrap text cleanly within pixel width."""
    lines, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if cur and cv2.getTextSize(t, FONT, scale, thick)[0][0] > max_w:
            lines.append(cur)
            cur = w
        else:
            cur = t
    return lines + ([cur] if cur else [])


def draw_pain_gauge(
    img: np.ndarray,
    x: int,
    y: int,
    score: int,
    distress: bool = False,
    scale: float = 0.44
) -> int:
    """Renders a clinical 10-point pain scale gauge with dynamic color coding."""
    w_gauge, h_gauge = 180, 22
    draw_rounded_rect(img, (x, y), (x + w_gauge, y + h_gauge), (24, 28, 36), -1, 4)
    draw_rounded_rect(img, (x, y), (x + w_gauge, y + h_gauge), COLOR_CARD_BORDER, 1, 4)

    # Fill segments based on score
    clamped_score = max(1, min(10, score))
    fill_w = int((clamped_score / 10.0) * (w_gauge - 4))
    
    # Determine color by severity
    if clamped_score <= 3:
        bar_col = COLOR_ACCENT_MINT
    elif clamped_score <= 6:
        bar_col = COLOR_ACCENT_AMBER
    elif clamped_score <= 8:
        bar_col = (40, 130, 245)  # Orange
    else:
        bar_col = COLOR_ACCENT_RED

    if fill_w > 0:
        cv2.rectangle(img, (x + 2, y + 2), (x + 2 + fill_w, y + h_gauge - 2), bar_col, -1)

    label_txt = f"PAIN: {clamped_score}/10"
    if distress:
        label_txt += " [DISTRESS]"
    cv2.putText(img, label_txt, (x + 6, y + h_gauge - 6), FONT, scale, COLOR_TEXT_PRIMARY, 1, cv2.LINE_AA)
    return w_gauge


def render_doctor_reply_overlay(panel: np.ndarray, vw: int, vh: int) -> None:
    """Renders a sleek modal menu for quick doctor visual card selection."""
    modal_w, modal_h = min(vw - 40, 580), 245
    mx1 = (vw - modal_w) // 2
    my1 = 20
    mx2, my2 = mx1 + modal_w, my1 + modal_h

    # Dim background
    draw_rounded_rect(panel, (mx1, my1), (mx2, my2), (18, 22, 28), -1, 10)
    draw_rounded_rect(panel, (mx1, my1), (mx2, my2), COLOR_ACCENT_CYAN, 2, 10)

    # Modal Header
    cv2.putText(panel, "DOCTOR QUICK CARDS (Press 1-9, 0, or [/] to Type)", (mx1 + 16, my1 + 24), FONT, 0.44, COLOR_ACCENT_CYAN, 2, cv2.LINE_AA)
    cv2.putText(panel, "[T]/[ESC] Dismiss", (mx2 - 130, my1 + 24), FONT, 0.36, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

    # 10 Options in 2 columns
    cards = [
        ("1", "MED", "Medication Coming"),
        ("2", "BLD", "Blood Test Needed"),
        ("3", "XRY", "X-Ray / Scan Scheduled"),
        ("4", "WAIT", "Please Wait ~10 Mins"),
        ("5", "DOC", "Doctor Examining You"),
        ("6", "H2O", "Water is on the Way"),
        ("7", "SAFE", "You are Safe With Us"),
        ("8", "LOC", "Where Does it Hurt?"),
        ("9", "FST", "Fasting Status Check"),
        ("0", "ALG", "Drug Allergy Check"),
    ]

    col_w = modal_w // 2 - 16
    for idx, (key, tag, title) in enumerate(cards):
        col = idx // 5
        row = idx % 5
        cx = mx1 + 14 + col * col_w
        cy = my1 + 38 + row * 38

        draw_rounded_rect(panel, (cx, cy), (cx + col_w - 8, cy + 32), (28, 34, 44), -1, 6)
        draw_rounded_rect(panel, (cx, cy), (cx + col_w - 8, cy + 32), COLOR_CARD_BORDER, 1, 6)

        # Key badge
        cv2.rectangle(panel, (cx + 4, cy + 4), (cx + 28, cy + 28), COLOR_ACCENT_CYAN, -1)
        cv2.putText(panel, key, (cx + 10, cy + 22), FONT, 0.48, (20, 24, 30), 2, cv2.LINE_AA)

        # Text
        txt_str = f"[{tag}] {title}"
        cv2.putText(panel, txt_str, (cx + 34, cy + 22), FONT, 0.38, COLOR_TEXT_PRIMARY, 1, cv2.LINE_AA)


def render_modern_dashboard(
    frame: np.ndarray,
    lm: Optional[np.ndarray],
    show_skel: bool,
    now: float,
    status: str,
    status_type: str,           # "ready", "signing", "recognized", "inferred", "unsure"
    cand_txt: str,
    words: List[Dict],
    sentence: str,
    alt_sentence: str,
    suggestions: List[Tuple[str, float]],
    log: List[str],
    speaker_active: bool,
    fps: float,
    speed: float,
    progress: float,
    scale: float = 1.0,
    # Extended Clinical Parameters
    triage: Optional[Any] = None,
    pain_info: Optional[Dict] = None,
    drug_alerts: Optional[List[Any]] = None,
    active_card: Optional[Any] = None,
    ui_mode: str = "split",      # "split", "doctor", or "patient"
    doctor_reply_menu: bool = False,
    toast_message: Optional[str] = None,
    spelled_text: Optional[str] = None,
    diagnosis: Optional[Any] = None,
    neural_cands: Optional[List[Tuple[str, float, str]]] = None,
) -> np.ndarray:
    """Renders the complete modern medical dashboard with Dual-Screen & Multi-Agent HUD."""
    view = cv2.flip(frame, 1)
    if show_skel and lm is not None:
        draw_skeleton(view, lm)

    vh, vw = view.shape[:2]

    # ---- Top Glass Overlay Header (Inside Camera View) ----
    header_h = 44
    overlay = view.copy()
    cv2.rectangle(overlay, (0, 0), (vw, header_h), (15, 18, 22), -1)
    cv2.addWeighted(overlay, 0.78, view, 0.22, 0, view)

    # Logo / Title + Mode Indicator
    if ui_mode == "split":
        mode_tag = "TWO-WAY CLINICAL (SPLIT)"
    elif ui_mode == "doctor":
        mode_tag = "DOCTOR DASHBOARD"
    else:
        mode_tag = "PATIENT VIEW"

    cv2.putText(view, "+ ASLMed", (14, 28), FONT, 0.68, COLOR_ACCENT_CYAN, 2, cv2.LINE_AA)
    cv2.putText(view, f"| {mode_tag}", (125, 28), FONT, 0.44, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

    # Status Pill Center
    status_colors = {
        "ready": (COLOR_CARD_BG, COLOR_TEXT_MUTED, (0, 220, 240)),
        "signing": (COLOR_CARD_BG, COLOR_ACCENT_AMBER, (0, 220, 255)),
        "recognized": (COLOR_CARD_BG, COLOR_ACCENT_MINT, (80, 255, 120)),
        "inferred": (COLOR_CARD_BG, COLOR_ACCENT_INDIGO, (240, 140, 200)),
        "unsure": (COLOR_CARD_BG, COLOR_ACCENT_RED, (100, 120, 255)),
    }
    bg_col, txt_col, dot_col = status_colors.get(status_type, status_colors["ready"])

    stat_txt = f"{status.upper()}"
    (stat_w, _), _ = cv2.getTextSize(stat_txt, FONT, 0.48, 2)
    pill_x1 = (vw - stat_w - 40) // 2
    pill_x2 = pill_x1 + stat_w + 40
    draw_rounded_rect(view, (pill_x1, 8), (pill_x2, 36), bg_col, -1, 6)
    draw_rounded_rect(view, (pill_x1, 8), (pill_x2, 36), dot_col, 1, 6)
    cv2.circle(view, (pill_x1 + 14, 22), 4, dot_col, -1, cv2.LINE_AA)
    cv2.putText(view, stat_txt, (pill_x1 + 26, 28), FONT, 0.48, txt_col, 2, cv2.LINE_AA)

    # Right Indicators: FPS + Voice
    voice_txt = "VOICE ON" if speaker_active else "MUTED (V)"
    right_info = f"{fps:.0f} FPS  |  {voice_txt}"
    (r_w, _), _ = cv2.getTextSize(right_info, FONT, 0.44, 1)
    cv2.putText(view, right_info, (vw - r_w - 14, 28), FONT, 0.44, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

    # Motion Progress Bar at boundary between camera and console
    if progress > 0.05:
        cv2.rectangle(view, (0, vh - 5), (int(vw * progress), vh), COLOR_ACCENT_MINT, -1)

    # ---- Bottom Console Panel ----
    panel_h = 300
    panel = np.full((panel_h, vw, 3), COLOR_BG_DARK, np.uint8)

    if ui_mode == "split":
        # ======================================================================
        # TWO-WAY SPLIT CLINICAL VIEW (Patient on Left, Doctor on Right)
        # ======================================================================
        half_w = vw // 2

        # --- LEFT PANE: PATIENT SCREEN ---
        draw_rounded_rect(panel, (6, 6), (half_w - 4, panel_h - 32), (25, 31, 40), -1, 6)
        draw_rounded_rect(panel, (6, 6), (half_w - 4, panel_h - 32), (45, 55, 70), 1, 6)
        cv2.putText(panel, "PATIENT SCREEN (REASSURANCE VIEW)", (14, 22), FONT, 0.38, COLOR_ACCENT_CYAN, 1, cv2.LINE_AA)

        # What patient signed box
        draw_rounded_rect(panel, (12, 28), (half_w - 10, 112), (32, 40, 52), -1, 4)
        cv2.putText(panel, "WHAT YOU SIGNED (PATIENT INPUT):", (18, 44), FONT, 0.34, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
        disp_sentence = sentence or "Sign facing camera..."
        sent_col = COLOR_TEXT_PRIMARY if sentence else COLOR_TEXT_DIM
        p_lines = wrap_text(disp_sentence, 0.54, 2, half_w - 36)
        py = 66
        for line in p_lines[:2]:
            cv2.putText(panel, line, (18, py), FONT, 0.54, sent_col, 2, cv2.LINE_AA)
            py += 20

        if spelled_text:
            sp_txt = f"Fingerspelling: [{spelled_text}]"
            cv2.putText(panel, sp_txt[:35], (18, 104), FONT, 0.36, COLOR_ACCENT_AMBER, 1, cv2.LINE_AA)

        # Message from Medical Team box
        draw_rounded_rect(panel, (12, 120), (half_w - 10, panel_h - 38), (28, 36, 46), -1, 4)
        cv2.putText(panel, "FROM MEDICAL TEAM (DOCTOR REPLY):", (18, 136), FONT, 0.34, COLOR_ACCENT_MINT, 1, cv2.LINE_AA)
        if active_card:
            cv2.putText(panel, f"[{active_card.icon}] {active_card.headline[:32]}", (18, 164), FONT, 0.52, COLOR_TEXT_PRIMARY, 2, cv2.LINE_AA)
            sub_lines = wrap_text(active_card.subtext, 0.38, 2, half_w - 36)
            sy = 190
            for sl in sub_lines[:2]:
                cv2.putText(panel, sl, (18, sy), FONT, 0.38, COLOR_ACCENT_MINT, 1, cv2.LINE_AA)
                sy += 20
        else:
            cv2.putText(panel, "[OK] Attending medical team on standby", (18, 164), FONT, 0.46, COLOR_TEXT_PRIMARY, 1, cv2.LINE_AA)
            cv2.putText(panel, "Medical staff is monitoring your signs.", (18, 190), FONT, 0.38, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

        # --- RIGHT PANE: DOCTOR CLINICAL & AI CO-PILOT HUD ---
        draw_rounded_rect(panel, (half_w + 4, 6), (vw - 6, panel_h - 32), (22, 26, 36), -1, 6)
        draw_rounded_rect(panel, (half_w + 4, 6), (vw - 6, panel_h - 32), (50, 60, 75), 1, 6)
        cv2.putText(panel, "DOCTOR CLINICAL & AI CO-PILOT", (half_w + 12, 22), FONT, 0.36, COLOR_ACCENT_MINT, 1, cv2.LINE_AA)

        # Triage Badge + Pain Score on right side
        if triage:
            t_col = COLOR_ACCENT_RED if triage.is_critical else COLOR_ACCENT_MINT
            cv2.putText(panel, f"[{triage.category.split(':')[0]}]", (vw - 125, 22), FONT, 0.36, t_col, 2, cv2.LINE_AA)

        # Multi-Agent Status Chips
        cv2.putText(panel, "AI CO-PILOT:", (half_w + 12, 36), FONT, 0.28, COLOR_TEXT_DIM, 1, cv2.LINE_AA)
        cv2.putText(panel, "[VISION 30FPS] [TRIAGE] [SAFETY] [DDx] [SCRIBE]", (half_w + 74, 36), FONT, 0.28, COLOR_ACCENT_CYAN, 1, cv2.LINE_AA)

        # Section 1: Neural Softmax Probabilities
        cv2.putText(panel, "NEURAL SOFTMAX PROBABILITIES (BiGRU+ATTN / CONTEXT):", (half_w + 12, 50), FONT, 0.28, COLOR_TEXT_DIM, 1, cv2.LINE_AA)
        if neural_cands and len(neural_cands) > 0:
            ny = 62
            colors = [COLOR_ACCENT_CYAN, COLOR_ACCENT_AMBER, COLOR_ACCENT_MINT]
            for idx, c in enumerate(neural_cands[:3]):
                draw_neural_bar(panel, c[0], c[1], c[2], half_w + 12, ny, half_w - 24, bar_h=8, color=colors[idx % len(colors)])
                ny += 12
        elif words:
            cur_gx = half_w + 12
            for item in words[-3:]:
                w_len = draw_chip(panel, item["w"], cur_gx, 58, COLOR_CARD_BG, COLOR_TEXT_PRIMARY, tag=item.get("src", "med"), scale=0.34)
                cur_gx += w_len + 4
        else:
            cv2.putText(panel, "(Awaiting gesture input - 50 Medical BiGRU active)", (half_w + 14, 72), FONT, 0.32, COLOR_TEXT_DIM, 1, cv2.LINE_AA)

        # Section 2: Biometric AI Vector & Drug Safety
        if drug_alerts:
            first_alert = drug_alerts[0]
            alert_msg = f"[!] {first_alert.severity}: {first_alert.drug_name} for {first_alert.condition}"
            cv2.putText(panel, alert_msg[:44], (half_w + 12, 110), FONT, 0.34, COLOR_ACCENT_RED, 2, cv2.LINE_AA)
        elif pain_info:
            bt = pain_info.get("brow_tension", 0.0)
            sq = pain_info.get("squint_ratio", 0.0)
            gr = pain_info.get("mouth_grimace", 0.0)
            ps = pain_info.get("pain_score", 1)
            dist_tag = " [DISTRESS]" if pain_info.get("facial_distress") else ""
            cv2.putText(panel, f"AI BIOMETRIC: Brow {bt:.2f} | Squint {sq:.2f} | Grimace {gr:.2f} | Pain {ps}/10{dist_tag}", (half_w + 12, 110), FONT, 0.30, COLOR_ACCENT_INDIGO, 1, cv2.LINE_AA)

        # Section 3: AI Clinical Reasoning & Differential Diagnosis (DDx) Box
        draw_rounded_rect(panel, (half_w + 10, 118), (vw - 12, 204), (28, 34, 46), -1, 4)
        draw_rounded_rect(panel, (half_w + 10, 118), (vw - 12, 204), COLOR_CARD_BORDER, 1, 4)
        cv2.putText(panel, "AI CLINICAL REASONING & DIFFERENTIAL DIAGNOSIS (DDx):", (half_w + 16, 130), FONT, 0.28, COLOR_ACCENT_MINT, 1, cv2.LINE_AA)
        if diagnosis:
            cv2.putText(panel, f"Primary DDx: {diagnosis.primary_ddx[:38]}", (half_w + 16, 146), FONT, 0.32, COLOR_TEXT_PRIMARY, 1, cv2.LINE_AA)
            diff_str = ", ".join(diagnosis.differentials[:2])
            cv2.putText(panel, f"Diff: {diff_str[:42]}", (half_w + 16, 162), FONT, 0.30, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
            workup_str = ", ".join(diagnosis.suggested_workup[:2])
            cv2.putText(panel, f"Workup: {workup_str[:42]}", (half_w + 16, 178), FONT, 0.30, COLOR_ACCENT_CYAN, 1, cv2.LINE_AA)
            cv2.putText(panel, f"Inquiry: \"{diagnosis.suggested_doctor_question[:38]}\"", (half_w + 16, 194), FONT, 0.30, COLOR_ACCENT_AMBER, 1, cv2.LINE_AA)
        else:
            cv2.putText(panel, "Primary DDx: Awaiting clinical symptoms...", (half_w + 16, 154), FONT, 0.32, COLOR_TEXT_DIM, 1, cv2.LINE_AA)
            cv2.putText(panel, "Workup: Baseline Vitals & Continuous Monitoring", (half_w + 16, 174), FONT, 0.30, COLOR_TEXT_DIM, 1, cv2.LINE_AA)

        # Section 4: Doctor Communication Hint Box
        draw_rounded_rect(panel, (half_w + 10, 212), (vw - 12, panel_h - 38), (25, 31, 42), -1, 4)
        draw_rounded_rect(panel, (half_w + 10, 212), (vw - 12, panel_h - 38), COLOR_CARD_BORDER, 1, 4)
        cv2.putText(panel, "COMMUNICATE WITH PATIENT:", (half_w + 16, 226), FONT, 0.30, COLOR_ACCENT_CYAN, 1, cv2.LINE_AA)
        cv2.putText(panel, "[T] Quick Cards 1-9  |  [/] Type Terminal  |  [E] Export SOAP", (half_w + 16, 244), FONT, 0.32, COLOR_TEXT_PRIMARY, 1, cv2.LINE_AA)

    elif ui_mode == "patient":
        # ======================================================================
        # PATIENT VIEW MODE (High contrast, reassuring, large visual cards)
        # ======================================================================
        # Patient Card 1: What the patient signed
        card1_y1, card1_y2 = 12, 112
        draw_rounded_rect(panel, (12, card1_y1), (vw - 12, card1_y2), (28, 38, 48), -1, 8)
        draw_rounded_rect(panel, (12, card1_y1), (vw - 12, card1_y2), COLOR_ACCENT_CYAN, 2, 8)
        cv2.putText(panel, "WHAT YOU SIGNED (PATIENT VIEW)", (24, 34), FONT, 0.44, COLOR_ACCENT_CYAN, 1, cv2.LINE_AA)

        disp_sentence = sentence or "Please sign facing the camera..."
        sent_col = COLOR_TEXT_PRIMARY if sentence else COLOR_TEXT_DIM
        sent_lines = wrap_text(disp_sentence, 0.76, 2, vw - 60)
        sy = 64
        for line in sent_lines[:2]:
            cv2.putText(panel, line, (24, sy), FONT, 0.76, sent_col, 2, cv2.LINE_AA)
            sy += 26

        if spelled_text:
            cv2.putText(panel, f"Fingerspelling: [{spelled_text}]", (24, 102), FONT, 0.42, COLOR_ACCENT_AMBER, 1, cv2.LINE_AA)

        # Patient Card 2: Doctor Visual Card / Message
        card2_y1, card2_y2 = 122, 260
        draw_rounded_rect(panel, (12, card2_y1), (vw - 12, card2_y2), (26, 32, 42), -1, 8)
        draw_rounded_rect(panel, (12, card2_y1), (vw - 12, card2_y2), COLOR_CARD_BORDER, 1, 8)

        cv2.putText(panel, "MESSAGE FROM MEDICAL TEAM", (24, 144), FONT, 0.42, COLOR_ACCENT_MINT, 1, cv2.LINE_AA)

        if active_card:
            cv2.putText(panel, f"[{active_card.icon}] {active_card.headline}", (24, 184), FONT, 0.68, COLOR_TEXT_PRIMARY, 2, cv2.LINE_AA)
            sub_lines = wrap_text(active_card.subtext, 0.48, 2, vw - 60)
            sy = 216
            for sl in sub_lines[:2]:
                cv2.putText(panel, sl, (24, sy), FONT, 0.48, COLOR_ACCENT_MINT, 1, cv2.LINE_AA)
                sy += 24
        else:
            cv2.putText(panel, "[OK] You are in good hands. Doctor is reviewing your signs.", (24, 184), FONT, 0.58, COLOR_TEXT_PRIMARY, 2, cv2.LINE_AA)
            cv2.putText(panel, "Medical team is actively attending to your care.", (24, 218), FONT, 0.48, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

    else:
        # ======================================================================
        # DOCTOR CLINICAL DASHBOARD MODE
        # ======================================================================
        # Card 1: Hero Translation Card (Top of Panel)
        card1_y1, card1_y2 = 10, 105
        draw_rounded_rect(panel, (12, card1_y1), (vw - 12, card1_y2), COLOR_CARD_BG, -1, 8)
        draw_rounded_rect(panel, (12, card1_y1), (vw - 12, card1_y2), COLOR_CARD_BORDER, 1, 8)

        # Translation Header & Triage Badge
        cv2.putText(panel, "CLINICAL INTERPRETATION", (24, 28), FONT, 0.40, COLOR_ACCENT_CYAN, 1, cv2.LINE_AA)

        # Right side of Hero Card: Triage & Pain Score
        if triage:
            triage_txt = f"[{triage.category.split(':')[0]}]"
            t_col = COLOR_ACCENT_RED if triage.is_critical else COLOR_ACCENT_MINT
            cv2.putText(panel, triage_txt, (vw - 130, 28), FONT, 0.44, t_col, 2, cv2.LINE_AA)

        if pain_info:
            p_score = pain_info.get("pain_score", 1)
            p_distress = pain_info.get("facial_distress", False)
            draw_pain_gauge(panel, vw - 220, 75, p_score, distress=p_distress)

        # Live English Sentence
        disp_sentence = sentence or "Start signing... (e.g. 'I have chest pain' or 'Doctor please')"
        sent_col = COLOR_TEXT_PRIMARY if sentence else COLOR_TEXT_DIM
        sent_lines = wrap_text(disp_sentence, 0.68, 2, vw - 240)
        sy = 54
        for line in sent_lines[:2]:
            cv2.putText(panel, line, (24, sy), FONT, 0.68, sent_col, 2, cv2.LINE_AA)
            sy += 22

        if spelled_text:
            cv2.putText(panel, f"Fingerspelling: [{spelled_text}]", (24, 96), FONT, 0.38, COLOR_ACCENT_AMBER, 1, cv2.LINE_AA)
        elif alt_sentence and alt_sentence != sentence:
            alt_txt = f"Alt style (TAB): {alt_sentence}"
            cv2.putText(panel, alt_txt[:65], (24, 96), FONT, 0.38, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

        # Card 2: Recognized Words Stream (Chips)
        y_chips = 114
        cv2.putText(panel, "SIGNED GLOSSES:", (14, y_chips + 18), FONT, 0.42, COLOR_TEXT_DIM, 1, cv2.LINE_AA)
        cur_x = 140
        if not words:
            cv2.putText(panel, "(no signs yet)", (cur_x, y_chips + 18), FONT, 0.44, COLOR_TEXT_DIM, 1, cv2.LINE_AA)
        else:
            for item in words[-6:]:
                w_str = item["w"]
                src_str = item.get("src", "med")
                chip_col = COLOR_CARD_BG
                if "intent" in src_str:
                    chip_col = (45, 35, 60)
                elif "daily" in src_str:
                    chip_col = (25, 45, 55)

                w_len = draw_chip(panel, w_str, cur_x, y_chips, chip_col, COLOR_TEXT_PRIMARY, tag=src_str, scale=0.42)
                cur_x += w_len + 8
                if cur_x > vw - 80:
                    break

        # Card 3: Smart Next-Sign Intent Predictions
        y_intent = 152
        cv2.putText(panel, "SMART INTENT:", (14, y_intent + 18), FONT, 0.42, COLOR_ACCENT_INDIGO, 1, cv2.LINE_AA)
        cur_ix = 140
        if suggestions:
            for idx, (sug_w, prior) in enumerate(suggestions[:3], 1):
                btn_txt = f"[{idx}] {sug_w.upper()}"
                sug_len = draw_chip(panel, btn_txt, cur_ix, y_intent, (38, 30, 48), COLOR_ACCENT_INDIGO, scale=0.40)
                cur_ix += sug_len + 8
        else:
            cv2.putText(panel, "Context-aware transitions active...", (cur_ix, y_intent + 18), FONT, 0.40, COLOR_TEXT_DIM, 1, cv2.LINE_AA)

        # Card 4: Clinical Triage & Safety Alert Row
        y_triage_row = 194
        if triage:
            t_summary = f"{triage.category} | Target: {triage.response_target}"
            t_col = COLOR_ACCENT_RED if triage.is_critical else COLOR_ACCENT_MINT
            cv2.putText(panel, t_summary[:85], (14, y_triage_row), FONT, 0.40, t_col, 1, cv2.LINE_AA)
        elif cand_txt:
            cv2.putText(panel, cand_txt[:95], (14, y_triage_row), FONT, 0.38, COLOR_TEXT_DIM, 1, cv2.LINE_AA)

        # Drug Safety Warning Row if any
        y_safe_row = 218
        if drug_alerts:
            first_alert = drug_alerts[0]
            alert_msg = f"[!] DRUG ALERT: {first_alert.severity} - {first_alert.drug_name} for {first_alert.condition}"
            cv2.putText(panel, alert_msg[:90], (14, y_safe_row), FONT, 0.40, COLOR_ACCENT_RED, 2, cv2.LINE_AA)
        elif log:
            cv2.putText(panel, f"Last Spoken: \"{log[-1][:75]}\"", (14, y_safe_row), FONT, 0.40, COLOR_ACCENT_MINT, 1, cv2.LINE_AA)
        else:
            cv2.putText(panel, "Drug & Allergy Safety Agent: Standby (Active monitoring)", (14, y_safe_row), FONT, 0.38, COLOR_TEXT_DIM, 1, cv2.LINE_AA)

        # Active Doctor Card Preview
        y_card_prev = 242
        if active_card:
            card_prev_txt = f"Active Visual Card: [{active_card.icon}] {active_card.headline_en}"
            cv2.putText(panel, card_prev_txt[:85], (14, y_card_prev), FONT, 0.40, COLOR_ACCENT_CYAN, 1, cv2.LINE_AA)
        else:
            cv2.putText(panel, "Press [T] Quick-Cards  |  [/] Type Custom Message to Patient", (14, y_card_prev), FONT, 0.38, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

    # Toast Message Banner (e.g. Encounter Export Notification)
    if toast_message:
        t_box_w = min(vw - 40, 520)
        tx1 = (vw - t_box_w) // 2
        draw_rounded_rect(panel, (tx1, 215), (tx1 + t_box_w, 255), (15, 60, 30), -1, 6)
        draw_rounded_rect(panel, (tx1, 215), (tx1 + t_box_w, 255), COLOR_ACCENT_MINT, 1, 6)
        cv2.putText(panel, toast_message[:65], (tx1 + 14, 240), FONT, 0.42, COLOR_TEXT_PRIMARY, 1, cv2.LINE_AA)

    # Bottom Shortcuts Toolbar
    cv2.rectangle(panel, (0, 268), (vw, 300), (14, 17, 22), -1)
    bar_txt = "ENTER Speak | [D] View (Split/Doc/Pat) | [T] Cards | [/] Type to Patient | [E] Export | C Clear | Q Quit"
    cv2.putText(panel, bar_txt, (14, 288), FONT, 0.38, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

    # Overlay Doctor Quick-Reply Menu if opened
    if doctor_reply_menu:
        render_doctor_reply_overlay(panel, vw, panel_h)

    canvas = np.vstack([view, panel])
    if scale != 1.0:
        canvas = cv2.resize(canvas, None, fx=scale, fy=scale)
    return canvas
