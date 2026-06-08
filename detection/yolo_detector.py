"""
detection/yolo_detector.py
YOLOv8 person detector via ONNX Runtime — no PyTorch required.

Model: yolov8n.onnx  (place in models/ directory)
Download: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx

Detects class 0 (person) only.
"""
import cv2
import numpy as np
import os

try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "yolov8n.onnx")

# YOLOv8 input resolution
_INPUT_W = 640
_INPUT_H = 640
_CONF_THRESH = 0.40
_NMS_THRESH  = 0.45
_PERSON_CLS  = 0


class YOLODetector:
    """
    Wraps YOLOv8n ONNX inference.
    Falls back to a MOG2 background subtractor when the model isn't present
    so the rest of the pipeline can still run.
    """

    def __init__(self, model_path: str = MODEL_PATH):
        self._session = None
        self._fallback = False

        if not _ORT_AVAILABLE:
            print("[YOLO] onnxruntime not installed — using background-sub fallback")
            self._fallback = True
            self._mog = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
            return

        if not os.path.exists(model_path):
            print(f"[YOLO] model not found at {model_path}")
            print("       Download from:")
            print("       https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx")
            print("       and place it in the models/ directory.")
            print("       Falling back to background-subtraction player detection.")
            self._fallback = True
            self._mog = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
            return

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        try:
            self._session = ort.InferenceSession(model_path, providers=providers)
            self._input_name = self._session.get_inputs()[0].name
            print(f"[YOLO] loaded {model_path}")
            print(f"       providers: {self._session.get_providers()}")
        except Exception as e:
            print(f"[YOLO] failed to load model: {e} — using fallback")
            self._fallback = True
            self._mog = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

    # ── public API ────────────────────────────────────────────────────────────

    def detect(self, frame_bgr: np.ndarray) -> list[tuple]:
        """
        Returns list of (x1, y1, x2, y2, conf) in *frame* pixel coordinates.
        """
        if self._fallback:
            return self._fallback_detect(frame_bgr)
        return self._onnx_detect(frame_bgr)

    # ── ONNX inference ────────────────────────────────────────────────────────

    def _onnx_detect(self, frame_bgr: np.ndarray) -> list[tuple]:
        h0, w0 = frame_bgr.shape[:2]

        # Letterbox to 640×640
        img, ratio, (dw, dh) = _letterbox(frame_bgr, (_INPUT_W, _INPUT_H))
        img = img[:, :, ::-1].transpose(2, 0, 1)          # BGR→RGB, HWC→CHW
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        img = img[np.newaxis]                              # add batch dim

        outputs = self._session.run(None, {self._input_name: img})
        # YOLOv8 output shape: (1, 84, 8400)  — 84 = 4 box + 80 classes
        preds = outputs[0][0].T                            # (8400, 84)

        boxes, scores = [], []
        for row in preds:
            cls_scores = row[4:]
            cls_id = int(np.argmax(cls_scores))
            conf = float(cls_scores[cls_id])
            if cls_id != _PERSON_CLS or conf < _CONF_THRESH:
                continue
            cx, cy, bw, bh = row[:4]
            # Convert from letterboxed space back to original frame
            x1 = (cx - bw / 2 - dw) / ratio
            y1 = (cy - bh / 2 - dh) / ratio
            x2 = (cx + bw / 2 - dw) / ratio
            y2 = (cy + bh / 2 - dh) / ratio
            x1 = max(0, min(w0, int(x1)))
            y1 = max(0, min(h0, int(y1)))
            x2 = max(0, min(w0, int(x2)))
            y2 = max(0, min(h0, int(y2)))
            boxes.append([x1, y1, x2, y2])
            scores.append(conf)

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(
            [[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in boxes],
            scores, _CONF_THRESH, _NMS_THRESH
        )
        result = []
        for i in (indices.flatten() if len(indices) else []):
            x1, y1, x2, y2 = boxes[i]
            result.append((x1, y1, x2, y2, scores[i]))
        return result

    # ── background-subtraction fallback ──────────────────────────────────────

    def _fallback_detect(self, frame_bgr: np.ndarray) -> list[tuple]:
        """
        Very rough person detection via MOG2 + connected components.
        Good enough for testing the pipeline without the YOLO model.
        """
        fgmask = self._mog.apply(frame_bgr)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN,  kernel)

        n, labels, stats, _ = cv2.connectedComponentsWithStats(fgmask)
        h0, w0 = frame_bgr.shape[:2]
        detections = []
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if area < 2000 or bh < 60 or bw > bh * 2:
                continue
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w0, x + bw), min(h0, y + bh)
            detections.append((x1, y1, x2, y2, 0.5))
        return detections


# ── utility ───────────────────────────────────────────────────────────────────

def _letterbox(img: np.ndarray, new_shape=(640, 640), color=(114, 114, 114)):
    h, w = img.shape[:2]
    nh, nw = new_shape
    ratio = min(nh / h, nw / w)
    rw, rh = int(round(w * ratio)), int(round(h * ratio))
    dw = (nw - rw) / 2
    dh = (nh - rh) / 2
    img = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right  = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)
