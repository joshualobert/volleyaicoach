"""
tracker/player_tracker.py
ByteTrack-style multi-object tracker — pure Python / numpy, no PyTorch.

Associates YOLO detections frame-to-frame using IoU matching + a simple
Kalman filter per track.  Assigns persistent IDs.
"""
import numpy as np
from collections import defaultdict


# ── IoU helper ────────────────────────────────────────────────────────────────

def _iou(a: tuple, b: tuple) -> float:
    """IoU between two boxes (x1,y1,x2,y2)."""
    ix1 = max(a[0], b[0]);  iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]);  iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


# ── Per-track Kalman (position + velocity) ────────────────────────────────────

class _TrackKalman:
    """
    Simple 8-state Kalman: [cx,cy,w,h, vcx,vcy,vw,vh]
    Observation: [cx,cy,w,h]
    """
    def __init__(self, box):
        cx, cy, w, h = _box_to_xywh(box)
        self.x = np.array([cx, cy, w, h, 0., 0., 0., 0.], dtype=float)
        self.P = np.eye(8) * 100.
        self.F = np.eye(8); self.F[0,4]=self.F[1,5]=self.F[2,6]=self.F[3,7]=1.
        self.H = np.eye(4, 8)
        q = 5.
        self.Q = np.diag([q,q,q*0.5,q*0.5, q*2,q*2,q,q])
        r = 15.
        self.R = np.diag([r,r,r*0.5,r*0.5])

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return _xywh_to_box(self.x[:4])

    def update(self, box):
        z = np.array(_box_to_xywh(box), dtype=float)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P
        return _xywh_to_box(self.x[:4])


def _box_to_xywh(box):
    x1,y1,x2,y2 = box[:4]
    return (x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1

def _xywh_to_box(v):
    cx,cy,w,h = v
    return int(cx-w/2), int(cy-h/2), int(cx+w/2), int(cy+h/2)


# ── Track object ──────────────────────────────────────────────────────────────

class _Track:
    _next_id = 1

    def __init__(self, box, conf):
        self.id    = _Track._next_id
        _Track._next_id += 1
        self.kf    = _TrackKalman(box)
        self.box   = tuple(int(v) for v in box[:4])
        self.conf  = conf
        self.hits  = 1
        self.misses = 0

    def predict(self):
        self.box = self.kf.predict()
        return self.box

    def update(self, box, conf):
        self.box = tuple(int(v) for v in self.kf.update(box)[:4])
        self.conf = conf
        self.hits  += 1
        self.misses = 0

    @property
    def is_confirmed(self):
        return self.hits >= 2    # need 2 hits before displaying


# ── ByteTrack-style tracker ───────────────────────────────────────────────────

class PlayerTracker:
    """
    Usage:
        tracker = PlayerTracker()
        for frame in video:
            detections = yolo.detect(frame)         # list of (x1,y1,x2,y2,conf)
            tracks = tracker.update(detections)      # list of (x1,y1,x2,y2,id)
    """

    def __init__(self, iou_thresh: float = 0.35, max_misses: int = 30):
        self.iou_thresh = iou_thresh
        self.max_misses = max_misses
        self._tracks: list[_Track] = []

    def update(self, detections: list) -> list[tuple]:
        """
        detections: list of (x1,y1,x2,y2,conf)
        Returns:    list of (x1,y1,x2,y2,id)
        """
        # ── predict all tracks ────────────────────────────────────────────
        for t in self._tracks:
            t.predict()

        # ── greedy IoU matching ───────────────────────────────────────────
        unmatched_dets = list(range(len(detections)))
        matched_tracks = set()

        if self._tracks and detections:
            iou_mat = np.zeros((len(self._tracks), len(detections)))
            for ti, t in enumerate(self._tracks):
                for di, d in enumerate(detections):
                    iou_mat[ti, di] = _iou(t.box, d[:4])

            # Greedy: assign highest-IoU pairs first
            pairs = sorted(
                [(iou_mat[ti, di], ti, di)
                 for ti in range(len(self._tracks))
                 for di in range(len(detections))],
                reverse=True
            )
            for score, ti, di in pairs:
                if score < self.iou_thresh:
                    break
                if ti in matched_tracks or di not in unmatched_dets:
                    continue
                self._tracks[ti].update(detections[di][:4], detections[di][4])
                matched_tracks.add(ti)
                unmatched_dets.remove(di)

        # ── increment misses for unmatched tracks ─────────────────────────
        for ti, t in enumerate(self._tracks):
            if ti not in matched_tracks:
                t.misses += 1

        # ── spawn new tracks for unmatched detections ─────────────────────
        for di in unmatched_dets:
            self._tracks.append(_Track(detections[di][:4], detections[di][4]))

        # ── remove dead tracks ────────────────────────────────────────────
        self._tracks = [t for t in self._tracks if t.misses <= self.max_misses]

        # ── return confirmed tracks ───────────────────────────────────────
        return [
            (t.box[0], t.box[1], t.box[2], t.box[3], t.id)
            for t in self._tracks
            if t.is_confirmed
        ]
