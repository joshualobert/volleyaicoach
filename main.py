"""
VolleyAI — main.py
Run:  python3 main.py

Gradio UI:
  1. Upload video
  2. Scrub to a clear start frame
  3. Click the volleyball on the preview
  4. Click "Track" — processed video appears in the output panel
"""

import os
import uuid
import time
import cv2
import numpy as np
import gradio as gr
from collections import deque

from tracker.ball_tracker import BallTracker, CONF_LOW
from utils.drawing import draw_ball, draw_trail, draw_hud
from detection.court_detector import detect_court

DEBUG_TRACKING = False   # set True to overlay search ellipse, scores, etc.

# ── config ────────────────────────────────────────────────────────────────────

OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "output")
PREVIEW_MAX = 960    # longest edge of preview frames

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── video utilities ───────────────────────────────────────────────────────────

def _get_path(file) -> str:
    if isinstance(file, str):   return file
    if isinstance(file, dict):  return file.get("path") or file.get("name", "")
    if hasattr(file, "path"):   return file.path
    return str(file)


def _resize(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    if max(h, w) <= PREVIEW_MAX:
        return frame
    s = PREVIEW_MAX / max(h, w)
    return cv2.resize(frame, (int(w * s), int(h * s)), cv2.INTER_AREA)


def _read_frame(path: str, n: int) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {n}")
    return _resize(frame)


def _video_info(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    info = {
        "fps":   cap.get(cv2.CAP_PROP_FPS) or 30.0,
        "total": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return info


# ── drawing helpers (preview markers) ────────────────────────────────────────

BALL_R = 14


def _redraw(base_rgb: np.ndarray, clicks: list) -> np.ndarray:
    if base_rgb is None:
        return None
    img = cv2.cvtColor(base_rgb.copy(), cv2.COLOR_RGB2BGR)
    for c in clicks:
        if c["mode"] == "ball":
            cx, cy = int(c["x"]), int(c["y"])
            cv2.circle(img, (cx, cy), BALL_R, (0, 0, 255), 2)
            cv2.circle(img, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(img, "Ball", (cx + BALL_R + 4, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ── tracking pipeline ─────────────────────────────────────────────────────────

def run_tracking(file, clicks: list, start_frame: int,
                 progress_cb=None) -> tuple[str, dict]:

    path  = _get_path(file)
    info  = _video_info(path)
    fps   = info["fps"]
    total = info["total"]
    start = max(0, min(int(start_frame), total - 1))

    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    ok, seed = cap.read()
    if not ok:
        raise RuntimeError("Cannot read start frame")
    seed = _resize(seed)
    h, w = seed.shape[:2]

    out_path = os.path.join(OUTPUT_DIR, f"{uuid.uuid4().hex}.mp4")
    # Try H.264 first (browser-playable), fall back to mp4v
    for codec in ("avc1", "mp4v", "XVID"):
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*codec), fps, (w, h))
        if writer.isOpened():
            print(f"[Writer] using codec: {codec}")
            break
    if not writer.isOpened():
        raise RuntimeError("Could not open video writer with any codec")

    # ── detect court boundaries from seed frame ────────────────────────────
    court_mask = detect_court(seed)

    # ── init tracker ──────────────────────────────────────────────────────
    ball_tracker = None
    ball_click   = next((c for c in clicks if c["mode"] == "ball"), None)
    if ball_click:
        bx, by = int(ball_click["x"]), int(ball_click["y"])
        ball_tracker = BallTracker(seed, bx, by,
                                   debug=DEBUG_TRACKING,
                                   court_mask=court_mask)

    left = max(total - start - 1, 1)
    done = 0
    t0   = time.time()

    def process(frame_bgr: np.ndarray) -> np.ndarray:
        nonlocal done
        # Ball tracking
        if ball_tracker:
            result = ball_tracker.update(frame_bgr)
            cx, cy, conf = result["cx"], result["cy"], result["conf"]

            # Use debug frame if debug mode produced one
            if result["debug_frame"] is not None:
                frame_bgr = result["debug_frame"]
            else:
                draw_trail(frame_bgr, ball_tracker.trail)
                if result.get("above_frame"):
                    # Ball above frame — draw a "↑ waiting" arrow at top
                    cv2.putText(frame_bgr, "^ above frame",
                                (cx - 50, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                else:
                    draw_ball(frame_bgr, cx, cy)
                    if conf < CONF_LOW:
                        cv2.putText(frame_bgr, "?", (cx + 18, cy - 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 100, 255), 2)

        elapsed = time.time() - t0
        live_fps = done / elapsed if elapsed > 0 else 0
        draw_hud(frame_bgr, live_fps)
        return frame_bgr

    # ── write seed frame ──────────────────────────────────────────────────
    writer.write(process(seed.copy()))
    done = 1

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = _resize(frame)
        writer.write(process(frame))
        done += 1
        if progress_cb and done % max(1, left // 100) == 0:
            progress_cb(done, left)

    cap.release()
    writer.release()
    if progress_cb:
        progress_cb(left, left)

    elapsed = time.time() - t0
    stats = {
        "frames":   done,
        "fps":      round(fps, 1),
        "duration": round(done / fps, 1),
        "avg_fps":  round(done / elapsed, 1),
    }
    return out_path, stats


# ── Gradio event handlers ─────────────────────────────────────────────────────

def on_upload(file):
    if file is None:
        return [gr.update(visible=False)] * 3 + [None, None, []]
    path  = _get_path(file)
    info  = _video_info(path)
    total = info["total"]
    fps   = info["fps"]
    bgr   = _read_frame(path, 0)
    frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return (
        gr.update(visible=True, minimum=0, maximum=max(total - 1, 1),
                  step=1, value=0,
                  label=f"Start frame  (0–{total-1}  ·  {total/fps:.1f}s @ {fps:.0f} fps)"),
        gr.update(visible=True),
        gr.update(visible=False),
        frame,
        frame.copy(),
        [],
    )


def on_scrub(file, n):
    if file is None:
        return None, None, []
    path  = _get_path(file)
    bgr   = _read_frame(path, int(n))
    frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return frame, frame.copy(), []


def on_click(evt: gr.SelectData, base, clicks, mode):
    if base is None:
        return None, clicks
    clicks = list(clicks or [])
    # Only one ball click allowed
    if mode == "ball":
        clicks = [c for c in clicks if c["mode"] != "ball"]
    clicks.append({"x": evt.index[0], "y": evt.index[1], "mode": mode})
    return _redraw(base, clicks), clicks


def on_undo(base, clicks):
    clicks = list(clicks or [])
    if clicks:
        clicks.pop()
    return _redraw(base, clicks), clicks


def on_clear(base):
    return _redraw(base, []), []


def on_track(file, clicks, frame_num, progress=gr.Progress(track_tqdm=False)):
    if file is None:
        raise gr.Error("Upload a video first.")
    ball = next((c for c in (clicks or []) if c["mode"] == "ball"), None)
    if ball is None:
        raise gr.Error("Click on the ball in the preview before tracking.")

    info  = _video_info(_get_path(file))
    total = info["total"]

    def prog(done, left):
        pct = done / max(left, 1)
        fps_est = ""
        progress(pct, desc=f"Processing frame {done}/{left}  ({int(pct*100)}%)")

    try:
        out_path, stats = run_tracking(file, clicks, int(frame_num), prog)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise gr.Error(f"Tracking failed: {e}")

    dur = stats["duration"]
    m, s = int(dur // 60), round(dur % 60, 1)
    summary = (
        f"**{m}m {s}s analysed**  ·  "
        f"**Source:** {stats['fps']} fps  ·  "
        f"**Processing speed:** {stats['avg_fps']} fps avg"
    )
    return out_path, summary, gr.update(visible=True)


# ── Gradio layout ─────────────────────────────────────────────────────────────

with gr.Blocks(title="VolleyAI") as demo:

    base_st   = gr.State(None)
    clicks_st = gr.State([])

    gr.Markdown(
        "# VolleyAI\n"
        "**1** Upload video  ·  **2** Scrub to a clear frame  ·  "
        "**3** Click the ball  ·  **4** Track"
    )

    video_in = gr.File(label="Upload video", file_types=["video"])

    frame_slider = gr.Slider(minimum=0, maximum=1, step=1, value=0,
                             label="Start frame", visible=False)

    with gr.Column(visible=False) as ann_col:
        with gr.Row():
            mode_radio = gr.Radio(
                ["ball"], value="ball", label="Click mode",
                info="Click directly on the volleyball to seed the tracker.",
            )
            with gr.Column(scale=0, min_width=120):
                undo_btn  = gr.Button("↩ Undo",  variant="secondary")
                clear_btn = gr.Button("✕ Clear", variant="secondary")

        preview = gr.Image(label="Click the ball to place tracker seed",
                           interactive=True)

        track_btn = gr.Button("▶  Track video", variant="primary", size="lg")

    with gr.Column(visible=False) as result_col:
        gr.Markdown("---")
        video_out = gr.Video(label="Tracked output", interactive=False)
        stats_md  = gr.Markdown()
        again_btn = gr.Button("Start over", variant="secondary")

    # ── wiring ────────────────────────────────────────────────────────────

    video_in.upload(
        fn=on_upload,
        inputs=video_in,
        outputs=[frame_slider, ann_col, result_col,
                 base_st, preview, clicks_st],
    )

    frame_slider.release(
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
demo.launch(inbrowser=True, share=True, allowed_paths=[OUTPUT_DIR])

