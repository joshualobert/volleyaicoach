"""
VolleyAI - Ball and player tracking using OpenCV only (no PyTorch needed).

Ball detection  : Hough circle transform on blurred grayscale
Player detection: MOG2 background subtraction → large contours
Output          : annotated MP4 with red circle on ball, blue boxes on players
"""

import cv2
import numpy as np
from pathlib import Path


def track_video(input_path: str, output_path: str, progress_cb=None) -> dict:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps        = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Write output as mp4
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Background subtractor for player detection
    bg_sub = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=40, detectShadows=False)

    # Simple ball position smoother
    ball_history = []   # last N detected positions
    SMOOTH = 5

    ball_detected_count   = 0
    player_detected_count = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated = frame.copy()

        # ── Player detection ────────────────────────────────────────────────
        fg_mask = bg_sub.apply(frame)
        # Remove noise
        kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        player_num  = 1
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Players are large blobs; skip tiny noise and very large regions (whole court)
            if 3000 < area < (width * height * 0.25):
                x, y, w, h = cv2.boundingRect(cnt)
                aspect = h / max(w, 1)
                if aspect > 1.0:   # taller than wide — person-shaped
                    cv2.rectangle(annotated, (x, y), (x+w, y+h), (255, 140, 0), 2)
                    cv2.putText(annotated, f"P{player_num}", (x, y-6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 140, 0), 2)
                    player_num += 1
                    player_detected_count += 1

        # ── Ball detection ──────────────────────────────────────────────────
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 2)

        ball_radius_min = max(8,  int(min(width, height) * 0.012))
        ball_radius_max = max(35, int(min(width, height) * 0.06))

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=min(width, height) // 6,
            param1=60,
            param2=28,
            minRadius=ball_radius_min,
            maxRadius=ball_radius_max,
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))
            # Pick the most confident circle (first returned by HoughCircles)
            cx, cy, r = circles[0][0]
            ball_history.append((int(cx), int(cy)))
            if len(ball_history) > SMOOTH:
                ball_history.pop(0)
            ball_detected_count += 1

        # Draw smoothed ball position
        if ball_history:
            avg_x = int(sum(p[0] for p in ball_history) / len(ball_history))
            avg_y = int(sum(p[1] for p in ball_history) / len(ball_history))
            r_draw = max(ball_radius_min, 18)
            # Filled semi-transparent red circle
            overlay = annotated.copy()
            cv2.circle(overlay,    (avg_x, avg_y), r_draw, (0, 0, 220), -1)
            cv2.addWeighted(overlay, 0.45, annotated, 0.55, 0, annotated)
            # Solid outline
            cv2.circle(annotated,  (avg_x, avg_y), r_draw, (0, 0,255), 2)
            # Trail dots
            for i, (px, py) in enumerate(ball_history[:-1]):
                alpha = (i + 1) / len(ball_history)
                radius_dot = max(3, int(r_draw * 0.3 * alpha))
                cv2.circle(annotated, (px, py), radius_dot, (0, 60, 255), -1)

        # Watermark
        cv2.putText(annotated, "VolleyAI", (12, height - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        out.write(annotated)
        frame_idx += 1

        if progress_cb and frame_idx % max(1, total_frames // 40) == 0:
            pct = int(frame_idx / max(total_frames, 1) * 100)
            progress_cb(pct)

    cap.release()
    out.release()

    return {
        "frames":         frame_idx,
        "fps":            round(fps, 1),
        "duration_s":     round(frame_idx / fps, 1),
        "ball_detections": ball_detected_count,
        "ball_pct":       round(ball_detected_count / max(frame_idx, 1) * 100, 1),
    }
