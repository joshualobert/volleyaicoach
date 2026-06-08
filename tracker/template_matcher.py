"""
tracker/template_matcher.py
Normalised cross-correlation template matcher with:
  - Multi-scale search (handles the ball changing apparent size)
  - Adaptive template update (only refresh when match is strong)
  - Blur-robust matching (blurs both template and ROI before NCC)
  - Returns position, score, and a scale hint
"""
import cv2
import numpy as np


_MIN_SCORE_UPDATE = 0.55   # minimum NCC to refresh template
_MIN_SCORE_ACCEPT = 0.25   # minimum NCC to return a result at all
_SCALES = [0.8, 1.0, 1.2]  # try template at these size ratios


class TemplateMatcher:

    def __init__(self, gray: np.ndarray, cx: int, cy: int, radius: int = 20):
        self._r = radius
        self._template: np.ndarray | None = None
        self._update(gray, cx, cy)

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _cut(gray: np.ndarray, cx: int, cy: int, r: int) -> np.ndarray | None:
        if gray is None or gray.size == 0:
            return None
        h, w = gray.shape
        cx = max(0, min(w - 1, cx))
        cy = max(0, min(h - 1, cy))
        x1, y1 = max(0, cx - r), max(0, cy - r)
        x2, y2 = min(w, cx + r), min(h, cy + r)
        if x2 - x1 < 6 or y2 - y1 < 6:
            return None
        patch = gray[y1:y2, x1:x2]
        if patch.size == 0:
            return None
        return patch.copy()

    def _update(self, gray: np.ndarray, cx: int, cy: int) -> None:
        patch = self._cut(gray, cx, cy, self._r)
        if patch is not None:
            # Slight blur — makes the template robust to minor motion blur
            self._template = cv2.GaussianBlur(patch, (3, 3), 0)

    # ── public API ────────────────────────────────────────────────────────────

    def match(self, gray: np.ndarray,
              pred_cx: int, pred_cy: int,
              search_r: int) -> tuple[int | None, int | None, float]:
        """
        Search in a window of ±search_r around (pred_cx, pred_cy).

        Returns
        -------
        cx, cy : int or None   — best match centre in full-frame coords
        score  : float [0, 1]  — NCC score of best match
        """
        if self._template is None:
            return None, None, 0.0

        h_f, w_f = gray.shape
        pred_cx = max(0, min(w_f - 1, pred_cx))
        pred_cy = max(0, min(h_f - 1, pred_cy))
        x1 = max(0, pred_cx - search_r)
        x2 = min(w_f, pred_cx + search_r)
        y1 = max(0, pred_cy - search_r)
        y2 = min(h_f, pred_cy + search_r)
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None, None, 0.0
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            return None, None, 0.0

        # Blur the ROI to match the blurred template
        roi_b = cv2.GaussianBlur(roi, (3, 3), 0)

        best_score = -1.0
        best_cx, best_cy = None, None

        for scale in _SCALES:
            th, tw = self._template.shape
            new_h = max(6, int(th * scale))
            new_w = max(6, int(tw * scale))
            tpl = cv2.resize(self._template, (new_w, new_h),
                             interpolation=cv2.INTER_LINEAR)

            if roi_b.shape[0] < tpl.shape[0] or roi_b.shape[1] < tpl.shape[1]:
                continue

            res = cv2.matchTemplate(roi_b, tpl, cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(res)
            if score > best_score:
                best_score = score
                best_cx    = x1 + loc[0] + tpl.shape[1] // 2
                best_cy    = y1 + loc[1] + tpl.shape[0] // 2

        if best_score < _MIN_SCORE_ACCEPT:
            return None, None, float(max(0.0, best_score))

        return int(best_cx), int(best_cy), float(best_score)

    def maybe_update(self, gray: np.ndarray, cx: int, cy: int,
                     score: float) -> None:
        """Refresh template only when match is strong enough."""
        if score >= _MIN_SCORE_UPDATE:
            self._update(gray, cx, cy)
