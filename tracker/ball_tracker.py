"""
tracker/ball_tracker.py
Hybrid volleyball ball tracker.

Detection priority:
  1. Colour (dominant) — yellow HSV mask, circular blob scoring
  2. Optical flow      — FB-verified LK, secondary motion estimate
  3. Template NCC      — fine-grained position correction
  4. Kalman            — prediction + smoothing through occlusion

Key design decisions:
  - Search region is an ELLIPSE oriented along the ball's velocity —
    wider in the travel direction, narrower perpendicular, so the
    tracker doesn't pick up objects above/below the trajectory.
  - Blobs are scored for CIRCULARITY, not just area — rejects players,
    jerseys, and rectangular objects that match the colour.
  - Ball is never declared "lost" quickly — Kalman glides for up to
    MAX_LOST frames while the search ellipse grows R_STEP px per miss.
  - No player tracking in this module.
"""
import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass

from .kalman           import KalmanBall
from .optical_flow     import OpticalFlowTracker
from .template_matcher import TemplateMatcher

# ── tuning ────────────────────────────────────────────────────────────────────

TRAIL_LEN   = 30
N_CONFIRM   = 2      # consecutive frames a reacq candidate must hold
MAX_LOST    = 90     # frames of Kalman glide before giving up (longer patience)

R_BASE      = 80     # search ellipse minor radius when on-track
R_STEP      = 25     # pixels added to both axes per missed frame
R_MAX       = 480    # hard cap

TPL_R       = 16     # template patch half-size (small = faster, less stale)

CONF_HIGH   = 0.60
CONF_LOW    = 0.25

# Ball considered "stationary" below this speed (px/frame)
_SLOW_SPEED    = 5.0
# Frames at slow speed before we reset Kalman velocity (setter hold)
_SLOW_FRAMES   = 3

# Mikasa yellow HSV fallback
_COL_LO_DEFAULT = np.array([12,  80,  80], np.uint8)
_COL_HI_DEFAULT = np.array([42, 255, 255], np.uint8)

# Circularity gate — 1.0 = perfect circle; fast-moving ball ≈ 0.35+
_MIN_CIRC   = 0.30


@dataclass
class TrailPoint:
    x: int;  y: int;  vx: float;  vy: float;  conf: float


# ── BallTracker ───────────────────────────────────────────────────────────────

class BallTracker:
    """
    tracker = BallTracker(frame_bgr, cx, cy, debug=False, court_mask=None)
    result  = tracker.update(frame_bgr)   # → dict, called every frame

    court_mask : optional uint8 mask (same size as frame_bgr), 255 = valid
                 tracking zone.  When supplied, colour detection is limited
                 to this zone so the ball is never latched outside the court.
    """

    def __init__(self, frame_bgr: np.ndarray, cx: int, cy: int,
                 debug: bool = False,
                 court_mask: np.ndarray | None = None):
        self.debug        = debug
        self._court_mask  = court_mask   # None or uint8 mask
        self._lost        = 0
        self._conf        = 1.0
        self._reacq_cand  = None
        self._reacq_cnt   = 0
        self._slow_cnt    = 0   # consecutive slow-speed frames (setter hold)
        self._above_frame = False  # ball has exited the top of the frame
        self.trail: deque[TrailPoint] = deque(maxlen=TRAIL_LEN)

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        self._col_lo, self._col_hi = self._sample_colour(frame_bgr, cx, cy)
        self._kf  = KalmanBall(cx, cy, process_noise=20.0, measure_noise=6.0)
        self._of  = OpticalFlowTracker()
        self._tm  = TemplateMatcher(gray, cx, cy, radius=TPL_R)
        self._of.seed(gray, cx, cy, radius=12)
        self.trail.append(TrailPoint(cx, cy, 0., 0., 1.0))

    # ── colour sampling ───────────────────────────────────────────────────────

    @staticmethod
    def _sample_colour(frame_bgr, cx, cy, r=18):
        if frame_bgr is None or frame_bgr.size == 0:
            return _COL_LO_DEFAULT.copy(), _COL_HI_DEFAULT.copy()
        fh, fw = frame_bgr.shape[:2]
        blurred  = cv2.GaussianBlur(frame_bgr, (5, 5), 0)
        hsv_full = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        roi = hsv_full[max(0,cy-r):min(fh,cy+r), max(0,cx-r):min(fw,cx+r)]
        if roi.size == 0:
            return _COL_LO_DEFAULT.copy(), _COL_HI_DEFAULT.copy()
        h_med = float(np.median(roi[:,:,0]))
        s_med = float(np.median(roi[:,:,1]))
        v_med = float(np.median(roi[:,:,2]))
        if 10 <= h_med <= 50 and s_med >= 60 and v_med >= 60:
            lo = np.array([max(8,  h_med-14), max(60, s_med-90), max(60, v_med-90)], np.uint8)
            hi = np.array([min(52, h_med+14), 255, 255], np.uint8)
            print(f"[Ball] H={h_med:.0f} S={s_med:.0f} V={v_med:.0f} "
                  f"→ lo={lo.tolist()} hi={hi.tolist()}")
        else:
            lo, hi = _COL_LO_DEFAULT.copy(), _COL_HI_DEFAULT.copy()
            print(f"[Ball] H={h_med:.0f} off-yellow — Mikasa preset")
        return lo, hi

    # ── adaptive elliptical search region ────────────────────────────────────

    def _search_ellipse(self, frame_bgr, pred_cx, pred_cy, vx, vy):
        """
        Return a binary mask (same size as frame).

        Three modes:
        1. ABOVE-FRAME  — ball exited the top; cover the full width of the
                          top half so it's found the instant it re-enters.
        2. SETTER HOLD  — ball is stationary; wide circle centred on last
                          known position so any set direction is detected.
        3. NORMAL FLIGHT — velocity-oriented ellipse, wider in travel
                           direction, grows each lost frame.
        """
        h, w = frame_bgr.shape[:2]
        mask  = np.zeros((h, w), dtype=np.uint8)
        speed = float(np.hypot(vx, vy))

        # ── Mode 1: ball above the frame ─────────────────────────────────
        if self._above_frame:
            # Full-width, top half — catches ball on the way down anywhere
            top_h = h // 2
            mask[:top_h, :] = 255
            return mask, w, top_h   # report width × half-height as axes

        # ── Mode 2: setter hold (slow for ≥ _SLOW_FRAMES frames) ─────────
        if self._slow_cnt >= _SLOW_FRAMES:
            # Large circle — direction is unknown after a set/backset
            r = min(R_MAX, R_BASE * 2 + self._lost * R_STEP)
            cv2.circle(mask, (pred_cx, pred_cy), r, 255, -1)
            return mask, r, r

        # ── Mode 3: normal flight ─────────────────────────────────────────
        minor = min(R_MAX, R_BASE + self._lost * R_STEP)
        major = min(R_MAX, R_BASE + int(speed * 2.5) + self._lost * R_STEP)

        # When velocity is low but not yet in full setter-hold, still use a
        # circle so direction-change on first touch isn't missed
        if speed < _SLOW_SPEED:
            major = minor

        angle_deg = float(np.degrees(np.arctan2(vy, vx))) if speed > 1 else 0.0
        cv2.ellipse(mask, (pred_cx, pred_cy),
                    (major, minor), angle_deg,
                    0, 360, 255, -1)
        return mask, major, minor

    # ── colour + circularity detection ────────────────────────────────────────

    def _colour_detect(self, frame_bgr, search_mask):
        """
        Find the most circular yellow blob inside the search mask.
        Scores each blob as:  circularity^2 × normalised_area
        Returns (cx, cy, radius, confidence) or (None,None,0,0).
        """
        # Apply search mask to frame before colour detection
        masked = cv2.bitwise_and(frame_bgr, frame_bgr, mask=search_mask)

        blurred = cv2.GaussianBlur(masked, (5, 5), 0)
        hsv  = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        cmask = cv2.inRange(hsv, self._col_lo, self._col_hi)

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cmask = cv2.morphologyEx(cmask, cv2.MORPH_CLOSE, k)
        cmask = cv2.morphologyEx(cmask, cv2.MORPH_OPEN,  k)

        # Only search inside the search mask.
        # Apply court mask only during normal flight — NOT when above-frame
        # (ball is in the air above court bounds) and NOT during setter hold
        # (ball may be at any horizontal position).
        effective_mask = search_mask
        if (self._court_mask is not None
                and not self._above_frame
                and self._slow_cnt < _SLOW_FRAMES):
            cm = self._court_mask
            if cm.shape[:2] != frame_bgr.shape[:2]:
                cm = cv2.resize(cm, (frame_bgr.shape[1], frame_bgr.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
            effective_mask = cv2.bitwise_and(search_mask, cm)
        cmask = cv2.bitwise_and(cmask, effective_mask)

        cnts, _ = cv2.findContours(cmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_score = 0.0
        best = (None, None, 0, 0.0)

        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < 15:
                continue
            perim = cv2.arcLength(cnt, True)
            if perim < 1:
                continue

            # Circularity (1.0 = perfect circle; jerseys/court ≈ 0.1)
            circ = 4 * np.pi * area / (perim ** 2)
            if circ < _MIN_CIRC:
                continue

            (bx, by), r = cv2.minEnclosingCircle(cnt)

            # Score: heavily weight circularity, also reward correct size
            expected_area = np.pi * 18**2   # ~18px radius expected
            size_score = min(1.0, area / expected_area)
            score = (circ ** 2) * size_score

            if score > best_score:
                best_score = score
                conf = min(1.0, circ * 1.2)
                best = (int(bx), int(by), max(int(r), 8), float(conf))

        return best

    # ── search radius for template / flow (bounding box of ellipse) ──────────

    def _search_r(self, vx, vy):
        speed = float(np.hypot(vx, vy))
        return min(R_MAX, R_BASE + int(speed * 2.5) + self._lost * R_STEP)

    # ── fuse estimates ────────────────────────────────────────────────────────

    def _fuse(self, pred_cx, pred_cy,
              col_cx, col_cy, col_conf,
              tm_cx,  tm_cy,  tm_score,
              of_cx,  of_cy,  of_score,
              direction_free: bool = False):
        """
        Weighted average of all estimates.
        Colour weight 0.65, template 0.25, flow 0.10.

        direction_free : when True (setter hold or above-frame) the Kalman
          prediction is stale / invalid so the distance penalty is disabled.
          Colour detection alone drives the result without being penalised for
          being "far" from the prediction.
        """
        pts, ws = [], []

        if col_cx is not None and col_conf > 0.05:
            pts.append((col_cx, col_cy));  ws.append(col_conf * 0.65)

        if tm_cx is not None and tm_score > 0.20:
            pts.append((tm_cx, tm_cy));    ws.append(tm_score * 0.25)

        if of_cx is not None and of_score > 0.15:
            pts.append((of_cx, of_cy));    ws.append(of_score * 0.10)

        if not pts:
            return None, None, 0.0

        tw  = sum(ws)
        cx  = int(sum(p[0]*w for p,w in zip(pts,ws)) / tw)
        cy  = int(sum(p[1]*w for p,w in zip(pts,ws)) / tw)

        if direction_free:
            # No distance penalty — trust the colour detector directly
            raw  = tw / (0.65 + 0.25 + 0.10)
            conf = float(raw)
        else:
            dist_pen = max(0.0, 1.0 - np.hypot(cx-pred_cx, cy-pred_cy) / 200.0)
            raw      = tw / (0.65 + 0.25 + 0.10)
            conf     = float(raw * (0.6 + 0.4 * dist_pen))
        return cx, cy, conf

    # ── public API ────────────────────────────────────────────────────────────

    def update(self, frame_bgr: np.ndarray) -> dict:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        h_f, w_f = frame_bgr.shape[:2]

        # Kalman predict (raw, before clamping — we need the real value to
        # detect above-frame exits)
        pred_cx_raw, pred_cy_raw = self._kf.predict()
        vx, vy = self._kf.velocity

        # ── Above-frame detection ─────────────────────────────────────────
        # Trigger when:
        #   (a) Kalman prediction exits the top of the frame, OR
        #   (b) ball was last seen in the top 8% of the frame moving upward
        #       (i.e. it's about to leave / has just left)
        top_band = int(h_f * 0.08)
        last_cy  = self.trail[-1].y if self.trail else pred_cy_raw
        exiting_top = (last_cy < top_band and vy < -2)

        if pred_cy_raw < 0 or exiting_top:
            if not self._above_frame:
                print(f"[Ball] left top of frame at y={last_cy:.0f} vy={vy:.1f}")
            self._above_frame = True
        elif self._above_frame and pred_cy_raw >= h_f * 0.05:
            # Prediction has returned well into the frame — reset and let
            # normal detection re-acquire
            self._above_frame = False
            self._kf.reset_velocity()
            print("[Ball] re-entering frame from top — velocity reset")

        # Clamp prediction to frame bounds for search purposes
        pred_cx = int(max(0, min(w_f - 1, pred_cx_raw)))
        pred_cy = int(max(0, min(h_f - 1, pred_cy_raw)))

        # ── Slow-phase (setter hold) detection ────────────────────────────
        speed = float(np.hypot(vx, vy))
        if speed < _SLOW_SPEED:
            self._slow_cnt += 1
        else:
            self._slow_cnt = 0

        # Once the ball has been held for enough frames, reset Kalman velocity
        # so the next trajectory is learned from scratch (handles backsets)
        if self._slow_cnt == _SLOW_FRAMES:
            self._kf.reset_velocity()
            vx, vy = 0.0, 0.0
            print(f"[Ball] stationary {_SLOW_FRAMES}f — velocity reset for direction change")

        # Build adaptive search region (circular when slow)
        search_mask, major, minor = self._search_ellipse(
            frame_bgr, pred_cx, pred_cy, vx, vy)

        # ── Run detectors ─────────────────────────────────────────────────
        col_cx, col_cy, col_r, col_conf = self._colour_detect(
            frame_bgr, search_mask)

        sr = self._search_r(vx, vy)
        tm_cx, tm_cy, tm_score = self._tm.match(gray, pred_cx, pred_cy, sr)
        of_cx, of_cy, of_score = self._of.estimate(gray)

        # ── Fuse ──────────────────────────────────────────────────────────
        # Disable Kalman distance penalty when the prediction is stale:
        # setter hold (velocity was just zeroed) or above-frame (prediction
        # is clamped to frame edge, not where ball actually is)
        dir_free = self._above_frame or (self._slow_cnt >= _SLOW_FRAMES)
        cx, cy, conf = self._fuse(pred_cx, pred_cy,
                                  col_cx, col_cy, col_conf,
                                  tm_cx,  tm_cy,  tm_score,
                                  of_cx,  of_cy,  of_score,
                                  direction_free=dir_free)

        # ── Reacquisition gate + lost counter ─────────────────────────────
        if cx is None or conf < CONF_LOW:
            # Don't count as "lost" while ball is legitimately above frame
            if not self._above_frame:
                self._lost += 1
                self._conf  = max(0.0, self._conf - 0.06)  # slow decay
            # else: hold confidence steady while waiting for ball to come back
            cx, cy = pred_cx, pred_cy               # glide on Kalman
            self._reacq_cand = None
            self._reacq_cnt  = 0

        else:
            if self._lost > 4:
                # Coming back from above frame: trust the first detection —
                # we know exactly where to expect it and the search area is
                # already restricted to the top half
                if self._above_frame:
                    print(f"[Ball] reacquired from above ({cx},{cy})")
                    self._reacq_cand  = None
                    self._reacq_cnt   = 0
                    self._lost        = 0
                    self._above_frame = False
                else:
                    # Normal reacq: require N_CONFIRM consistent frames
                    if self._reacq_cand is None:
                        self._reacq_cand = (cx, cy)
                        self._reacq_cnt  = 1
                    else:
                        if np.hypot(cx - self._reacq_cand[0],
                                    cy - self._reacq_cand[1]) < 45:
                            self._reacq_cnt  += 1
                            self._reacq_cand  = (cx, cy)
                        else:
                            self._reacq_cand = (cx, cy)
                            self._reacq_cnt  = 1

                    if self._reacq_cnt < N_CONFIRM:
                        cx, cy = pred_cx, pred_cy
                        conf   = 0.1
                    else:
                        print(f"[Ball] reacquired ({cx},{cy}) "
                              f"after {self._lost} frames | ellipse {major}×{minor}px")
                        self._reacq_cand  = None
                        self._reacq_cnt   = 0
                        self._lost        = 0
                        self._above_frame = False
            else:
                self._lost        = max(0, self._lost - 1)
                self._above_frame = False
                self._reacq_cand  = None
                self._reacq_cnt   = 0

            kx, ky  = self._kf.update(cx, cy)
            cx, cy  = int(kx), int(ky)
            self._conf = conf

            if conf >= CONF_HIGH:
                self._of.seed(gray, cx, cy, radius=12)
                self._tm.maybe_update(gray, cx, cy, tm_score)

        vx, vy = self._kf.velocity
        self.trail.append(TrailPoint(cx, cy, vx, vy, self._conf))
        # Half-frame interpolated midpoint — denser trail without sub-frame data
        if len(self.trail) >= 2:
            prev = self.trail[-2]
            mid_x = int((prev.x + cx) / 2)
            mid_y = int((prev.y + cy) / 2)
            last = self.trail.pop()
            self.trail.append(TrailPoint(mid_x, mid_y, vx * 0.5, vy * 0.5,
                                         self._conf * 0.8))
            self.trail.append(last)

        dbg = self._draw_debug(frame_bgr.copy(), pred_cx, pred_cy,
                               search_mask, cx, cy, col_conf,
                               of_cx, of_cy, tm_score, major, minor) \
              if self.debug else None

        return {
            "cx": cx, "cy": cy, "conf": self._conf,
            "pred_cx": pred_cx, "pred_cy": pred_cy,
            "of_cx": of_cx,     "of_cy": of_cy,
            "tm_score": round(tm_score or 0, 3),
            "col_conf": round(col_conf, 3),
            "search_major": major, "search_minor": minor,
            "lost": self._lost,
            "above_frame": self._above_frame,
            "debug_frame": dbg,
        }

    # ── debug overlay ─────────────────────────────────────────────────────────

    def _draw_debug(self, frame, pred_cx, pred_cy, search_mask,
                    cx, cy, col_conf, of_cx, of_cy, tm_score, major, minor):
        # Search ellipse boundary (grey)
        ellipse_edge = cv2.Canny(search_mask, 50, 150)
        frame[ellipse_edge > 0] = [80, 80, 80]

        # Kalman prediction (cyan cross)
        cv2.drawMarker(frame, (pred_cx, pred_cy), (220, 220, 0),
                       cv2.MARKER_CROSS, 18, 2)

        # Optical flow (blue dot)
        if of_cx is not None:
            cv2.circle(frame, (of_cx, of_cy), 6, (255, 120, 0), -1)

        # Final position (red)
        cv2.circle(frame, (cx, cy), 18, (0, 0, 255), 2)
        cv2.circle(frame, (cx, cy), 4,  (0, 0, 255), -1)

        lines = [
            f"conf  : {self._conf:.2f}",
            f"colour: {col_conf:.2f}",
            f"tm    : {tm_score:.2f}",
            f"lost  : {self._lost}",
            f"ellipse {major}x{minor}px",
        ]
        for i, t in enumerate(lines):
            cv2.putText(frame, t, (8, 20 + i*18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                        (255, 255, 255), 1, cv2.LINE_AA)
        return frame
