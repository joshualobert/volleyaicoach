"""
VolleyAI - Flask backend
Run with:  python3 app.py
Then open: http://localhost:8080
"""

import os
import threading
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory, send_file

from analyze import track_video

app = Flask(__name__)

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("output")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

ALLOWED = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# In-memory job store  {job_id: {"status", "progress", "result", "error"}}
jobs = {}
jobs_lock = threading.Lock()


def run_job(job_id, input_path, output_path):
    def progress(pct):
        with jobs_lock:
            jobs[job_id]["progress"] = pct

    try:
        result = track_video(str(input_path), str(output_path), progress_cb=progress)
        with jobs_lock:
            jobs[job_id]["status"]   = "done"
            jobs[job_id]["progress"] = 100
            jobs[job_id]["result"]   = result
    except Exception as e:
        with jobs_lock:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"]  = str(e)


@app.route("/")
def index():
    return send_file("VolleyAI main.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return jsonify({"error": "No file"}), 400

    f   = request.files["video"]
    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED:
        return jsonify({"error": f"Unsupported format: {ext}"}), 400

    job_id      = uuid.uuid4().hex
    input_path  = UPLOAD_DIR / f"{job_id}{ext}"
    output_path = OUTPUT_DIR / f"{job_id}.mp4"

    f.save(str(input_path))

    with jobs_lock:
        jobs[job_id] = {"status": "processing", "progress": 0, "result": None, "error": None}

    t = threading.Thread(target=run_job, args=(job_id, input_path, output_path), daemon=True)
    t.start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Not found"}), 404
    return jsonify(job)


@app.route("/result/<job_id>")
def result_video(job_id):
    path = OUTPUT_DIR / f"{job_id}.mp4"
    if not path.exists():
        return jsonify({"error": "Not ready"}), 404
    return send_file(str(path), mimetype="video/mp4")


if __name__ == "__main__":
    print("VolleyAI running at http://localhost:8080")
    app.run(host="0.0.0.0", port=8080, debug=False)
