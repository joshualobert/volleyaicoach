"""
VolleyAI
========
Run:  python3 volleyai.py
A browser tab opens automatically.

Ball   → HSV colour mask + contour detection (samples exact colour from click)
Player → MIL tracker (position-based)
"""

import os
import uuid
import cv2
import numpy as np
import gradio as gr

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cap preview frames so the image isn't enormous in the browser.
# Tracking also runs at this size, so click coords map 1-to-1.
PREVIEW_MAX = 960


# ═══════════════════════════════════════════════════════════════════════════════
#  Video utilities
# ═══════════════════════════════════════════════════════════════════════════════

def _get_path(file) -> str:
    """Extract a plain filepath string from whatever Gradio passes."""
    if isinstance(file, str):   return file
    if isinstance(file, dict):  return file.get("path") or file.get("name", "")
    if hasattr(file, "path"):   return file.path
    return str(file)


def _resize_for_preview(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    if max(h, w) <= PREVIEW_MAX:
        return frame
    s = PREVIEW_MAX / max(h, w)
    return cv2.resize(frame, (int(w * s), int(h * s)), cv2.INTER_AREA)


def get_video_info(file) -> dict:
    path = _get_path(file)
    cap  = cv2.VideoCapture(path)
    info = {
        "fps":   cap.get(cv2.CAP_PROP_FPS) or 30.0,
        "total": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return info


def read_frame(file, frame_num: int) -> np.ndarray:
    """Return RGB numpy array for one frame, capped at PREVIEW_MAX px."""
    path = _get_path(file)
    cap  = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_num)))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {frame_num}")
    frame = _resize_for_preview(frame)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# ═══════════════════════════════════════════════════════════════════════════════
#  Drawing helpers (preview markers)
# ═══════════════════════════════════════════════════════════════════════════════

BALL_R  = 16   # preview circle radius  (matches tracker search window)
PLR_HW  = 22   # player box half-width
PLR_HH  = 48   # player box half-height


def redraw_markers(base_rgb: np.ndarray, clicks: list) -> np.ndarray:
    """Stamp click markers onto base_rgb. Returns RGB."""
    if base_rgb is None:
        return None
    img = cv2.cvtColor(base_rgb.copy(), cv2.COLOR_RGB2BGR)
    pn  = 1
    for c in clicks:
        cx, cy = int(c["x"]), int(c["y"])
        if c["mode"] == "ball":
            cv2.circle(img, (cx, cy), BALL_R, (0, 0, 255), 2)
            cv2.circle(img, (cx, cy), 3,      (0, 0, 255), -1)
            cv2.putText(img, "Ball", (cx + BALL_R + 4, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        elif c["mode"] == "player":
            cv2.rectangle(img,
                          (cx - PLR_HW, cy - PLR_HH),
                          (cx + PLR_HW, cy + PLR_HH),
                          (0, 140, 255), 2)
            cv2.putText(img, f"P{pn}", (cx - PLR_HW, cy - PLR_HH - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)
            pn += 1
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ═══════════════════════════════════════════════════════════════════════════════
#  Ball tracking — Mikasa yellow HSV + Hough circles + physics prediction
# ═══════════════════════════════════════════════════════════════════════════════

# Hard-coded Mikasa yellow-and-blue volleyball HSV range (from product image).
# H≈27 (yellow), very high saturation and brightness.
_YELLOW_LO = np.array([15, 80, 80],   dtype=np.uint8)
_YELLOW_HI = np.array([40, 255, 255], dtype=np.uint8)

REACQUIRE_AT = 8   # consecutive misses before full-frame scan
HISTORY_LEN  = 12  # frames of position history kept for trajectory prediction


class BallTracker:
    """
    Tracks a Mikasa yellow volleyball.

    COLOUR  — Hard-coded yellow HSV range refined by sampling the clicked pixel.
    DETECT  — Hough circle detection on the yellow mask every frame.
              Falls back to largest-blob centroid when motion blur breaks circles.
    PREDICT — Keeps a rolling position history (HISTORY_LEN frames).
              Uses weighted linear regression over that history to predict the
              next position — much smoother than single-frame velocity.
              When the ball is in the setter's hands (nearly stationary for
              several frames), the regression slope captures the incoming
              direction and the search window is widened toward both pins
              (left and right edges) so the ball is found quickly after release.
    REACQUIRE — full-frame scan after REACQUIRE_AT consecutive misses.
    """

    def __init__(self, frame_bgr: np.ndarray, cx: int, cy: int, sample_r: int = 16):
        fh, fw = frame_bgr.shape[:2]

        # Sample colour at click to fine-tune the range for this lighting
        blurred  = cv2.GaussianBlur(frame_bgr, (5, 5), 0)
        hsv_full = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        y1 = max(0, cy - sample_r);  y2 = min(fh, cy + sample_r)
        x1 = max(0, cx - sample_r);  x2 = min(fw, cx + sample_r)
        roi = hsv_full[y1:y2, x1:x2]

        h_med = float(np.median(roi[:, :, 0]))
        s_med = float(np.median(roi[:, :, 1]))
        v_med = float(np.median(roi[:, :, 2]))

        if 14 <= h_med <= 45 and s_med >= 80 and v_med >= 80:
            self.lower = np.array([max(0,   h_med - 15),
                                   max(50,  s_med - 80),
                                   max(50,  v_med - 80)], dtype=np.uint8)
            self.upper = np.array([min(180, h_med + 15), 255, 255], dtype=np.uint8)
            print(f"[BallTracker] sampled H={h_med:.1f} S={s_med:.1f} V={v_med:.1f}")
        else:
            self.lower = _YELLOW_LO.copy()
            self.upper = _YELLOW_HI.copy()
            print(f"[BallTracker] H={h_med:.1f} off-ball — using Mikasa yellow preset")

        print(f"              lower={self.lower.tolist()}  upper={self.upper.tolist()}")

        # Rolling position history for trajectory regression
        self._hist: list[tuple[int,int]] = [(cx, cy)] * HISTORY_LEN
        self.last_pos = (cx, cy)
        self.last_r   = BALL_R
        self._lost    = 0
        self._fw      = fw   # frame width — used for pin-side widening

    # ── colour mask ───────────────────────────────────────────────────────────

    def _mask(self, bgr: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(bgr, (5, 5), 0)
        hsv  = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
        return mask

    # ── detection — Hough circles with blob fallback ──────────────────────────

    def _detect(self, frame_bgr: np.ndarray, off_x: int = 0, off_y: int = 0):
        """
        Return (abs_cx, abs_cy, r) of the best yellow circle, or None.
        Tries Hough first; falls back to largest-blob centroid.
        """
        mask = self._mask(frame_bgr)

        # ── Hough ──
        filled = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=2)
        circles = cv2.HoughCircles(
            filled, cv2.HOUGH_GRADIENT,
            dp=1, minDist=20,
            param1=30, param2=10,
            minRadius=max(4, BALL_R - 12),
            maxRadius=BALL_R + 25,
        )
        if circles is not None:
            # Pick circle whose centre has the most yellow pixels
            best, best_fill = None, -1
            for x, y, r in circles[0]:
                x, y, r = int(x), int(y), int(r)
                # Measure yellow fill inside this circle
                tmp = np.zeros(mask.shape, dtype=np.uint8)
                cv2.circle(tmp, (x, y), max(r, 4), 255, -1)
                fill = int(cv2.countNonZero(cv2.bitwise_and(mask, tmp)))
                if fill > best_fill:
                    best_fill = fill
                    best = (x + off_x, y + off_y, max(r, BALL_R))
            if best and best_fill > 10:
                return best

        # ── blob fallback ──
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        best_score, best_result = 0.0, None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 15:
                continue
            perim = cv2.arcLength(cnt, True)
            if perim == 0:
                continue
            circ = 4 * np.pi * area / (perim ** 2)
            if circ < 0.20:
                continue
            score = area * circ
            if score > best_score:
                best_score = score
                (bx, by), r = cv2.minEnclosingCircle(cnt)
                best_result = (int(bx) + off_x, int(by) + off_y, max(int(r), BALL_R))
        return best_result

    # ── trajectory prediction ─────────────────────────────────────────────────

    def _predict(self) -> tuple[int, int]:
        """
        Weighted linear regression over position history.
        Recent frames get higher weight so the prediction reacts quickly
        to direction changes (setter release, block).
        Returns predicted (x, y) for the next frame.
        """
        n  = len(self._hist)
        xs = np.array([p[0] for p in self._hist], dtype=float)
        ys = np.array([p[1] for p in self._hist], dtype=float)
        t  = np.arange(n, dtype=float)
        # Exponential weights — most recent frame has weight e^0=1, oldest e^-(n-1)
        w  = np.exp(np.linspace(-(n-1), 0, n))

        def wls(vals):
            sw  = w.sum()
            swt = (w * t).sum()
            swv = (w * vals).sum()
            swt2 = (w * t**2).sum()
            swtv = (w * t * vals).sum()
            denom = sw * swt2 - swt**2
            if abs(denom) < 1e-6:
                return float(vals[-1]), 0.0
            slope = (sw * swtv - swt * swv) / denom
            intercept = (swv - slope * swt) / sw
            return intercept, slope

        _, vx = wls(xs)
        _, vy = wls(ys)

        # Predict one frame ahead from the last known position
        px = int(self._hist[-1][0] + vx)
        py = int(self._hist[-1][1] + vy)
        return px, py, vx, vy   # also return velocity for window sizing

    # ── search window ─────────────────────────────────────────────────────────

    def _search(self, frame_bgr: np.ndarray, cx: int, cy: int, radius: int):
        fh, fw = frame_bgr.shape[:2]
        x1 = max(0, cx - radius);  x2 = min(fw, cx + radius)
        y1 = max(0, cy - radius);  y2 = min(fh, cy + radius)
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return self._detect(crop, x1, y1)

    # ── commit / glide ────────────────────────────────────────────────────────

    def _commit(self, bx: int, by: int, r: int):
        self._hist.append((bx, by))
        if len(self._hist) > HISTORY_LEN:
            self._hist.pop(0)
        self.last_pos = (bx, by)
        self.last_r   = r
        self._lost    = 0

    def _glide(self, vx: float, vy: float):
        self._lost += 1
        decay = 0.85 ** self._lost
        ex = int(self.last_pos[0] + vx * decay)
        ey = int(self.last_pos[1] + vy * decay)
        self.last_pos = (ex, ey)
        return ex, ey, self.last_r

    # ── public API ────────────────────────────────────────────────────────────

    def update(self, frame_bgr: np.ndarray):
        """Return (cx, cy, radius) in full-frame coordinates."""
        pred_x, pred_y, vx, vy = self._predict()

        speed = np.hypot(vx, vy)

        # Search radius:
        #  • normal tracking → 130 px
        #  • ball is slow/stationary (setter hold) → widen to 300 px so we
        #    catch it as soon as it leaves the setter's hands toward a pin
        #  • ball is moving fast → scale with speed so we don't miss it
        base_r = 130
        if speed < 4:
            # Ball nearly stationary → setter is holding it.
            # Widen search and bias toward both pin sides simultaneously.
            search_r = 300
        else:
            search_r = max(base_r, int(speed * 2.5))
            search_r = min(search_r, 350)

        # ── 1. Windowed search around predicted position ───────────────────
        found = self._search(frame_bgr, pred_x, pred_y, search_r)
        if found:
            if self._lost > 0:
                print(f"[BallTracker] found pos={found[:2]} lost={self._lost}")
            self._commit(*found)
            return found

        # ── 2. Glide on trajectory while briefly occluded ─────────────────
        if self._lost < REACQUIRE_AT:
            return self._glide(vx, vy)

        # ── 3. Full-frame re-acquisition ───────────────────────────────────
        found = self._detect(frame_bgr)
        if found:
            print(f"[BallTracker] re-acquired at {found[:2]} after {self._lost} frames")
            # Restart history at new position so regression isn't polluted
            self._hist = [found[:2]] * HISTORY_LEN
            self._commit(*found)
            return found

        return self._glide(vx, vy)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main tracking pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_tracking(file, output_path: str, clicks: list,
                 start_frame: int, progress_cb=None) -> dict:
    path = _get_path(file)
    cap  = cv2.VideoCapture(path)

    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start = max(0, min(int(start_frame), total - 1))

    # Read and resize the seed frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    ok, seed = cap.read()
    if not ok:
        raise RuntimeError("Cannot read start frame")
    seed = _resize_for_preview(seed)
    h, w = seed.shape[:2]

    # H.264 — plays in every browser
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"avc1"),
        fps, (w, h)
    )
    if not writer.isOpened():
        raise RuntimeError("Could not open video writer")

    # ── init trackers ─────────────────────────────────────────────────────────
    ball_tracker    = None
    player_trackers = []
    PLR_COLS = [
        (0, 140, 255), (255, 0, 200),
        (0, 210, 90),  (0, 200, 200), (180, 0, 255),
    ]

    for c in clicks:
        cx, cy = int(c["x"]), int(c["y"])
        if c["mode"] == "ball":
            ball_tracker = BallTracker(seed, cx, cy)
        elif c["mode"] == "player":
            hw, hh = PLR_HW, PLR_HH
            box = (max(0, cx - hw), max(0, cy - hh), hw * 2, hh * 2)
            t   = cv2.TrackerMIL_create()
            if t.init(seed, box):
                player_trackers.append(t)

    # ── per-frame annotation ──────────────────────────────────────────────────
    trail:    list[tuple] = []
    TRAIL_LEN = 16
    ball_hits = 0

    def annotate(frame_bgr: np.ndarray) -> np.ndarray:
        nonlocal ball_hits

        # Ball
        if ball_tracker:
            bx, by, r = ball_tracker.update(frame_bgr)
            trail.append((bx, by))
            if len(trail) > TRAIL_LEN:
                trail.pop(0)

            # Fading trail
            for i, (tx, ty) in enumerate(trail[:-1]):
                a = (i + 1) / len(trail)
                cv2.circle(frame_bgr, (tx, ty),
                           max(2, int(r * 0.35 * a)), (0, 0, 255), -1)

            # Red circle + centre dot
            cv2.circle(frame_bgr, (bx, by), r, (0, 0, 255), 2)
            cv2.circle(frame_bgr, (bx, by), 4, (0, 0, 255), -1)
            ball_hits += 1

        # Players
        for i, t in enumerate(player_trackers):
            ok2, box = t.update(frame_bgr)
            if ok2:
                x, y, bw, bh = (int(v) for v in box)
                col = PLR_COLS[i % len(PLR_COLS)]
                cv2.rectangle(frame_bgr, (x, y), (x + bw, y + bh), col, 2)
                cv2.putText(frame_bgr, f"P{i+1}", (x + 4, y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

        cv2.putText(frame_bgr, "VolleyAI", (8, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
        return frame_bgr

    writer.write(annotate(seed.copy()))
    done = 1
    left = max(total - start - 1, 1)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = _resize_for_preview(frame)
        writer.write(annotate(frame))
        done += 1
        if progress_cb and done % max(1, left // 60) == 0:
            progress_cb(min(int(done / left * 100), 99))

    cap.release()
    writer.release()
    if progress_cb:
        progress_cb(100)

    return {
        "frames":     done,
        "fps":        round(fps, 1),
        "duration_s": round(done / fps, 1),
        "ball_pct":   round(ball_hits / max(done, 1) * 100, 1),
        "players":    len(player_trackers),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Gradio UI
# ═══════════════════════════════════════════════════════════════════════════════

def on_upload(file):
    if file is None:
        return [gr.update(visible=False)] * 3 + [None, None, [], 0]
    try:
        info  = get_video_info(file)
        frame = read_frame(file, 0)
    except Exception as e:
        raise gr.Error(f"Could not read video: {e}")
    total = info["total"]
    fps   = info["fps"]
    return (
        gr.update(visible=True, minimum=0, maximum=max(total - 1, 1),
                  step=1, value=0,
                  label=f"Start frame  (0 – {total-1}  ·  {total/fps:.1f}s @ {fps:.0f} fps)"),
        gr.update(visible=True),
        gr.update(visible=False),
        frame,
        frame.copy(),
        [],
        0,
    )


def on_scrub(file, frame_num):
    if file is None:
        return None, None, []
    try:
        frame = read_frame(file, int(frame_num))
    except Exception as e:
        raise gr.Error(str(e))
    return frame, frame.copy(), []


def on_click(evt: gr.SelectData, base, clicks, mode):
    if base is None:
        return None, clicks
    clicks = list(clicks or [])
    clicks.append({"x": evt.index[0], "y": evt.index[1], "mode": mode})
    return redraw_markers(base, clicks), clicks


def on_clear(base):
    return redraw_markers(base, []), []


def on_undo(base, clicks):
    clicks = list(clicks or [])
    if clicks:
        clicks.pop()
    return redraw_markers(base, clicks), clicks


def on_track(file, clicks, frame_num, progress=gr.Progress(track_tqdm=False)):
    if file is None:
        raise gr.Error("Upload a video first.")
    if not clicks:
        raise gr.Error("Place at least one marker on the preview before tracking.")

    out_path = os.path.join(OUTPUT_DIR, f"{uuid.uuid4().hex}.mp4")

    def prog(pct):
        progress(pct / 100, desc=f"Tracking… {pct}%")

    try:
        stats = run_tracking(file, out_path, clicks, int(frame_num), prog)
    except Exception as e:
        raise gr.Error(f"Tracking failed: {e}")

    dur = stats["duration_s"]
    m, s = int(dur // 60), round(dur % 60, 1)
    summary = (
        f"**Analysed:** {m}m {s}s  ·  "
        f"**{stats['fps']} fps**  ·  "
        f"**Ball tracked:** {stats['ball_pct']}% of frames  ·  "
        f"**Players:** {stats['players']}"
    )
    return out_path, summary, gr.update(visible=True)


def on_retry(base, clicks):
    return redraw_markers(base, clicks), clicks


# ── layout ────────────────────────────────────────────────────────────────────

with gr.Blocks(title="VolleyAI") as demo:

    base_st   = gr.State(None)   # clean frame (no markers)
    clicks_st = gr.State([])
    frame_st  = gr.State(0)

    gr.Markdown(
        "# VolleyAI\n"
        "**1** Upload  ·  **2** Scrub to a clear frame  ·  "
        "**3** Mark ball & players  ·  **4** Track"
    )

    video_in = gr.File(label="Upload video", file_types=["video"])

    frame_slider = gr.Slider(minimum=0, maximum=1, step=1, value=0,
                             label="Start frame", visible=False)

    with gr.Column(visible=False) as ann_col:
        with gr.Row():
            mode_radio = gr.Radio(
                ["ball", "player"], value="ball", label="Click mode",
                info="Ball → click the ball.   Player → click each player's chest.",
            )
            with gr.Column(scale=0, min_width=140):
                undo_btn  = gr.Button("↩ Undo",      variant="secondary")
                clear_btn = gr.Button("✕ Clear all", variant="secondary")

        # No fixed height — natural image size means click coords are exact
        preview = gr.Image(
            label="Click to place markers",
            interactive=True,
        )

        track_btn = gr.Button("▶  Track video", variant="primary", size="lg")

    with gr.Column(visible=False) as result_col:
        gr.Markdown("---")
        video_out = gr.Video(label="Tracked output", interactive=False)
        stats_md  = gr.Markdown()
        again_btn = gr.Button("Start over", variant="secondary")

    # ── wire ──────────────────────────────────────────────────────────────────

    video_in.upload(
        fn=on_upload,
        inputs=video_in,
        outputs=[frame_slider, ann_col, result_col,
                 base_st, preview, clicks_st, frame_st],
    )

    frame_slider.release(          # fires only on mouse-up, not every tick
        fn=on_scrub,
        inputs=[video_in, frame_slider],
        outputs=[base_st, preview, clicks_st],
    )

    preview.select(
        fn=on_click,
        inputs=[base_st, clicks_st, mode_radio],
        outputs=[preview, clicks_st],
    )

    undo_btn.click(fn=on_undo,  inputs=[base_st, clicks_st], outputs=[preview, clicks_st])
    clear_btn.click(fn=on_clear, inputs=base_st,              outputs=[preview, clicks_st])

    track_btn.click(
        fn=on_track,
        inputs=[video_in, clicks_st, frame_slider],
        outputs=[video_out, stats_md, result_col],
    )

    again_btn.click(
        fn=lambda: (gr.update(visible=False), gr.update(visible=False)),
        outputs=[result_col, ann_col],
    )

demo.queue(max_size=2)
demo.launch(
    inbrowser=True,
    share=True,
    allowed_paths=[OUTPUT_DIR],   # lets Gradio serve our output videos
)
