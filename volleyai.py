"""
VolleyAI — run with:  python3 volleyai.py
A browser tab will open automatically.
"""

import tempfile
from pathlib import Path

import gradio as gr

from analyze import track_video


def analyse(video_path):
    if video_path is None:
        raise gr.Error("Please upload a video first.")

    out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    out.close()

    stats = track_video(video_path, out.name)

    dur   = stats["duration_s"]
    mins  = int(dur // 60)
    secs  = round(dur % 60, 1)
    dur_str = f"{mins}m {secs}s" if mins else f"{secs}s"

    summary = (
        f"**Duration:** {dur_str}  |  "
        f"**FPS:** {stats['fps']}  |  "
        f"**Ball detected:** {stats['ball_pct']}% of frames  |  "
        f"**Frames processed:** {stats['frames']:,}"
    )

    return out.name, summary


with gr.Blocks(title="VolleyAI", theme=gr.themes.Base()) as demo:
    gr.Markdown("# VolleyAI\nUpload a volleyball video — the ball will be tracked with a **red circle**, players with **blue boxes**.")

    with gr.Row():
        with gr.Column():
            video_in  = gr.Video(label="Upload video")
            run_btn   = gr.Button("Analyse", variant="primary")
        with gr.Column():
            video_out = gr.Video(label="Tracked output", interactive=False)
            stats_out = gr.Markdown()

    run_btn.click(fn=analyse, inputs=video_in, outputs=[video_out, stats_out])

demo.launch(inbrowser=True, share=True)
