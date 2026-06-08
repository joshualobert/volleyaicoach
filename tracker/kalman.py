"""
tracker/kalman.py
6-state Kalman filter: [x, y, vx, vy, ax, ay]
Models constant acceleration — appropriate for a volleyball under gravity
and during fast attacks where acceleration changes abruptly.
Pure numpy, no scipy.
"""
import numpy as np


class KalmanBall:
    """
    State:       [x, y, vx, vy, ax, ay]
    Observation: [x, y]

    Tuning guide
    ------------
    process_noise  : how much we trust the constant-acceleration model.
                     Higher → filter responds faster to sudden direction
                     changes (spikes, blocks) but is noisier.
    measure_noise  : how much we trust the detector.
                     Higher → smoother output but lags on fast movement.
    """

    def __init__(self, cx: float, cy: float,
                 process_noise: float = 15.0,
                 measure_noise: float = 8.0):

        # State transition — constant acceleration model, dt=1 frame
        dt = 1.0
        self.F = np.array([
            [1, 0, dt,  0, .5*dt**2,        0],
            [0, 1,  0, dt,        0, .5*dt**2],
            [0, 0,  1,  0,       dt,        0],
            [0, 0,  0,  1,        0,       dt],
            [0, 0,  0,  0,        1,        0],
            [0, 0,  0,  0,        0,        1],
        ], dtype=float)

        # Observation: we see only (x, y)
        self.H = np.zeros((2, 6), dtype=float)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0

        # Process noise — higher for acceleration states (ball can change
        # direction abruptly on contact)
        q = process_noise
        self.Q = np.diag([q*0.5, q*0.5,   # position
                          q,     q,         # velocity
                          q*2,   q*2])      # acceleration

        # Measurement noise
        r = measure_noise
        self.R = np.diag([r, r])

        # Initial state and covariance
        self.x = np.array([cx, cy, 0., 0., 0., 0.], dtype=float)
        self.P = np.eye(6) * 500.0   # high initial uncertainty

    # ── predict ───────────────────────────────────────────────────────────────

    def predict(self) -> tuple[float, float]:
        """One step forward. Returns predicted (cx, cy)."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return float(self.x[0]), float(self.x[1])

    # ── update ────────────────────────────────────────────────────────────────

    def update(self, cx: float, cy: float) -> tuple[float, float]:
        """Incorporate observation. Returns corrected (cx, cy)."""
        z = np.array([cx, cy], dtype=float)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        return float(self.x[0]), float(self.x[1])

    # ── velocity reset ────────────────────────────────────────────────────────

    def reset_velocity(self) -> None:
        """
        Zero out the velocity and acceleration estimates and widen the
        covariance so the filter adapts quickly to the new trajectory.
        Call this when the ball has been stationary (in setter's hands)
        and is about to be launched in an unknown direction.
        """
        self.x[2] = 0.0   # vx
        self.x[3] = 0.0   # vy
        self.x[4] = 0.0   # ax
        self.x[5] = 0.0   # ay
        # Widen uncertainty on velocity/acceleration states
        for i in (2, 3, 4, 5):
            self.P[i, i] = max(self.P[i, i], 800.0)

    # ── helpers ───────────────────────────────────────────────────────────────

    @property
    def position(self) -> tuple[float, float]:
        return float(self.x[0]), float(self.x[1])

    @property
    def velocity(self) -> tuple[float, float]:
        return float(self.x[2]), float(self.x[3])

    @property
    def speed(self) -> float:
        return float(np.hypot(self.x[2], self.x[3]))
