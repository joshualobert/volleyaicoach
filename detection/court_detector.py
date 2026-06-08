"""
detection/court_detector.py
Automatic volleyball court boundary detection.

Strategy:
  1. Detect the floor region using colour + horizontal edge clustering.
  2. Find the strongest horizontal white lines (court sidelines / end lines).
  3. From the lowest horizontal white line pair, derive left/right/bottom
     court boundaries.
  4. Extend the boundary upward to the top of the frame (covers full
     ball airspace above the court).
  5. Return a binary mask (uint8, 255=valid tracking zone).

Falls back to a small inset mask if detection fails.
"""
import cv2
import numpy as np


# Fraction of the frame to inset from each edge when court detection fails
_FALLBACK_INSET_X = 0.06   # 6% from left and right
_FALLBACK_INSET_Y = 0.02   # 2% from top, 4% from bottom


def _inset_mask(h: int, w: int,
                ix: float = _FALLBACK_INSET_X,
                iy_top: float = _FALLBACK_INSET_Y,
                iy_bot: float = _FALLBACK_INSET_Y * 2) -> np.ndarray:
    """Return a mask that is 255 inside the inset rectangle."""
    mask = np.zeros((h, w), dtype=np.uint8)
    x1 = int(w * ix)
    x2 = int(w * (1 - ix))
    y1 = int(h * iy_top)
    y2 = int(h * (1 - iy_bot))
    mask[y1:y2, x1:x2] = 255
    return mask


def detect_court(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Analyse a single (seed) frame and return a tracking zone mask.

    The mask is 255 in every pixel that is:
      - Inside the detected court boundaries (left, right columns), AND
      - Above the detected floor baseline (can reach the top of the frame
        to cover ball airspace).

    Parameters
    ----------
    frame_bgr : ndarray  — the seed frame (BGR, uint8)

    Returns
    -------
    mask : ndarray  — same H×W as frame_bgr, dtype uint8, values 0 or 255
    """
    h, w = frame_bgr.shape[:2]

    # ── Step 1: find white lines via brightness + low saturation ─────────────
    hsv  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    # White: low saturation, high value
    white_mask = cv2.inRange(hsv,
                             np.array([0,   0, 180], np.uint8),
                             np.array([180, 60, 255], np.uint8))

    # Close small gaps in line segments
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, k)

    # ── Step 2: find long near-horizontal line segments (HoughLinesP) ────────
    edges = cv2.Canny(white_mask, 30, 90)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                            threshold=60,
                            minLineLength=int(w * 0.15),
                            maxLineGap=20)

    h_lines = []   # list of y-coords for horizontal lines
    if lines is not None:
        for seg in lines:
            x1, y1, x2, y2 = seg[0]
            dx, dy = abs(x2 - x1), abs(y2 - y1)
            if dx == 0:
                continue
            angle = np.degrees(np.arctan2(dy, dx))
            if angle < 12:   # nearly horizontal
                y_mid = (y1 + y2) / 2
                h_lines.append(int(y_mid))

    # ── Step 3: find left/right court boundary columns ────────────────────────
    # Project white mask onto X axis — court sidelines make vertical spikes
    col_sum = white_mask.sum(axis=0).astype(np.float32)
    col_sum = cv2.GaussianBlur(col_sum.reshape(1, -1), (21, 1), 0).flatten()

    # Look for left boundary in left 40% and right boundary in right 40%
    left_col  = _find_boundary_col(col_sum, 0,          int(w * 0.40), from_left=True)
    right_col = _find_boundary_col(col_sum, int(w * 0.60), w,          from_left=False)

    # ── Step 4: determine floor baseline (lowest strong horizontal line) ──────
    floor_y = h   # default: very bottom
    if h_lines:
        # Use the median of lines in the bottom 40% of the frame
        bottom_lines = [y for y in h_lines if y > h * 0.50]
        if bottom_lines:
            floor_y = int(np.median(bottom_lines))
        else:
            floor_y = int(np.median(h_lines))
    # Add a small buffer below the detected line
    floor_y = min(h, floor_y + 20)

    # ── Step 5: build the mask ────────────────────────────────────────────────
    # Expand left/right boundary inward a tiny bit for safety
    margin_x = int(w * 0.01)
    x1 = max(0, left_col  + margin_x)
    x2 = min(w, right_col - margin_x)

    # Sanity check — if detection seems off, fall back
    valid = (x2 - x1) > w * 0.25 and floor_y > h * 0.30

    if not valid:
        print("[Court] detection uncertain — using inset fallback")
        return _inset_mask(h, w)

    mask = np.zeros((h, w), dtype=np.uint8)
    # Full height from top of frame down to floor baseline
    mask[0:floor_y, x1:x2] = 255
    print(f"[Court] x=[{x1},{x2}]  floor_y={floor_y}  "
          f"(frame {w}×{h})")
    return mask


def _find_boundary_col(col_sum: np.ndarray,
                        start: int, end: int,
                        from_left: bool) -> int:
    """
    Find the column index with the highest white-line density in [start, end].
    If nothing is found, return start (left) or end (right).
    """
    region = col_sum[start:end]
    if region.max() < 10:
        return start if from_left else end
    idx = int(np.argmax(region))
    return start + idx
