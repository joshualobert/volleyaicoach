"""
tracker/optical_flow.py
Lucas-Kanade sparse optical flow with forward-backward verification.

Forward-backward check: track points forward frame N→N+1, then track
the result backward N+1→N.  Points whose round-trip error exceeds a
threshold are rejected as unreliable.  This removes the drifting-off-ball
problem that kills naive LK tracking.
"""
import cv2
import numpy as np


# LK parameters — larger window handles bigger inter-frame motion
_LK = dict(
    winSize=(31, 31),
    maxLevel=4,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01),
)

_FB_THRESHOLD  = 2.0   # px — forward-backward round-trip tolerance
_MIN_GOOD_PTS  = 3     # minimum surviving points to trust the estimate


class OpticalFlowTracker:
    """
    Tracks a set of seed points using LK optical flow with FB verification.
    Call seed() whenever a reliable ball position is available, then
    call estimate() on each new frame.
    """

    def __init__(self):
        self._prev_gray: np.ndarray | None = None
        self._pts:       np.ndarray | None = None   # shape (N,1,2) float32

    # ── public API ────────────────────────────────────────────────────────────

    def seed(self, gray: np.ndarray, cx: int, cy: int, radius: int = 12) -> None:
        """
        Replace current points with a grid centred on (cx, cy).
        Call this every time you get a high-confidence detection.
        """
        self._prev_gray = gray.copy()
        pts = []
        steps = [-radius // 2, 0, radius // 2]
        for dx in steps:
            for dy in steps:
                pts.append([cx + dx, cy + dy])
        self._pts = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)

    def estimate(self, gray: np.ndarray) -> tuple[int | None, int | None, float]:
        """
        Estimate ball position from previous frame to this one.

        Returns
        -------
        cx, cy  : int or None — estimated ball centre
        score   : float [0, 1] — reliability of the estimate
                  (fraction of points that survived FB check,
                   scaled by how tightly they cluster)
        """
        if (self._prev_gray is None or self._pts is None
                or len(self._pts) == 0
                or self._prev_gray.size == 0
                or gray.size == 0):
            return None, None, 0.0

        # ── Forward pass ──────────────────────────────────────────────────
        new_pts, st_f, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._pts, None, **_LK)

        if new_pts is None or st_f is None:
            self._prev_gray = gray.copy()
            return None, None, 0.0

        # ── Backward pass ─────────────────────────────────────────────────
        back_pts, st_b, _ = cv2.calcOpticalFlowPyrLK(
            gray, self._prev_gray, new_pts, None, **_LK)

        if back_pts is None or st_b is None:
            self._prev_gray = gray.copy()
            return None, None, 0.0

        # ── Forward-backward consistency check ────────────────────────────
        fb_error = np.linalg.norm(
            self._pts.reshape(-1, 2) - back_pts.reshape(-1, 2),
            axis=1
        )
        good_mask = (st_f.flatten() == 1) & (st_b.flatten() == 1) & (fb_error < _FB_THRESHOLD)
        good_new  = new_pts.reshape(-1, 2)[good_mask]

        self._prev_gray = gray.copy()

        if len(good_new) < _MIN_GOOD_PTS:
            return None, None, 0.0

        # ── Estimate centre and confidence ────────────────────────────────
        median = np.median(good_new, axis=0)
        cx, cy = int(median[0]), int(median[1])

        # Spread of surviving points — tight cluster = high confidence
        spread = float(np.mean(np.linalg.norm(good_new - median, axis=1)))
        # Normalise: spread=0 → 1.0, spread=20px → 0.0
        cluster_score = max(0.0, 1.0 - spread / 20.0)
        survive_score = len(good_new) / len(self._pts)
        score = float(survive_score * 0.5 + cluster_score * 0.5)

        # Update seed points to the new positions (keep tracking the same patch)
        self._pts = good_new.reshape(-1, 1, 2)
        return cx, cy, score
