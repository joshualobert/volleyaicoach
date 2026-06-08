"""
utils/drawing.py
Centralised drawing helpers for all overlays.
"""
import cv2
import numpy as np
from collections import deque

# Colour palette for player IDs (BGR)
_PALETTE = [
    (0, 140, 255),   # orange
    (255, 50, 50),   # blue
    (50, 200, 50),   # green
    (0, 200, 200),   # yellow
    (200, 0, 200),   # magenta
    (255, 180, 0),   # sky-blue
    (0, 100, 255),   # deep-orange
    (180, 255, 0),   # lime
]


def player_colour(pid: int) -> tuple:
    return _PALETTE[int(pid) % len(_PALETTE)]


def draw_player(frame: np.ndarray, box: tuple, pid: int) -> None:
    """Draw bounding box + ID label for one player."""
    x1, y1, x2, y2 = (int(v) for v in box)
    col = player_colour(pid)
    cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)

    label = f"P{pid}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), col, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


def draw_ball(frame: np.ndarray, cx: int, cy: int, r: int = 14) -> None:
    """Draw ball circle + centre dot."""
    cv2.circle(frame, (cx, cy), r, (0, 0, 255), 2)
    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)


def draw_trail(frame: np.ndarray, trail: deque, max_r: int = 6) -> None:
    """
    Draw the last N ball positions as a fading red trail.
    Accepts either (x,y) tuples or TrailPoint dataclass objects.
    Older points are smaller and more transparent.
    """
    pts = list(trail)
    n = len(pts)
    for i, pt in enumerate(pts):
        # Support both (x,y) tuples and TrailPoint dataclass
        if hasattr(pt, "x"):
            tx, ty = pt.x, pt.y
        else:
            tx, ty = pt[0], pt[1]
        alpha = (i + 1) / n
        r = max(2, int(max_r * alpha))
        colour = (0, 0, int(255 * alpha))
        cv2.circle(frame, (int(tx), int(ty)), r, colour, -1)


def draw_hud(frame: np.ndarray, fps: float = 0.0) -> None:
    """Bottom-left watermark + optional FPS."""
    h = frame.shape[0]
    label = f"VolleyAI  {fps:.0f} fps" if fps > 0 else "VolleyAI"
    cv2.putText(frame, label, (8, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
