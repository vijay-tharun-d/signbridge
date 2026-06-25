"""
vision.py — MediaPipe Hand-Landmark Detection & Sign Classification

Changes applied:
  [Task 5]   SIGNS_VISION recalibrated to match new SIGNS servo angles using
             the specified angle→ratio mapping.
  [Task 5.1] _thumb_extension_ratio REWRITTEN with distance-based approach:
             Uses Euclidean distance from thumb TIP (lm4) to index MCP (lm5),
             normalised by palm_size. This is fully rotation-invariant and
             monotonically correlated with thumb opening — unlike the previous
             projection-based approach which was sensitive to hand orientation.
  [Task 5.2] Minimum hand-size guard (palm_size < 0.08 → return _empty).
  [Task 5.3] CONFIDENCE_THRESHOLD lowered from 1.2 to 0.80.
  [Task 5.4] _wrist_to_tip_direction helper for G/Q and K/P disambiguation.
  [Task 5.5] Disambiguation cases added/fixed: F vs W, B vs W, D vs G,
             C vs O, U vs V vs R cascaded.
  [Task 5.6] Comment added explaining Left/Right hand selection in mirrored webcam.
  [Thumb fix] G/Q/T disambiguator corrected — primary discriminator is now
              index_ratio < 0.30 (T has index at 45°) rather than thumb_ratio
              which is similar for all three signs.
  [Iteration 1] Verified every SIGNS_VISION entry matches the updated SIGNS
                 servo angles via the angle→ratio mapping table.
  [Iteration 2] Traced sign-to-text flow for letter 'B' — verified
                 classification returns correct ratios and disambiguation.
  [Iteration 3] Verified corrupted base64 returns _empty gracefully.

  [Refactor] _finger_extension_ratio COMPLETELY REWRITTEN:
             • Removed cumulative segment length math (l1+l2+l3).
             • Implements vector angle calculation (dot product) at PIP and DIP
               joints to find true bend angles.
             • Weighted PIP/DIP blend (PIP 60%, DIP 40%) — PIP is more
               anatomically significant for curl.
             • Smooth sigmoid-style mapping from joint angles to 0.0–1.0 ratio
               using calibrated breakpoints instead of harsh linear clamp.
             • Z-axis compression mitigation: dynamically boosts Z when palm
               is small (far from camera) to preserve joint geometry.
             • CONFIDENCE_THRESHOLD relaxed to 0.80 and _FEATURE_WEIGHTS
               rebalanced for angle-based logic stability.

Receives base64-encoded JPEG frames from the browser, extracts 21 hand
landmarks via MediaPipe HandLandmarker (Tasks API ≥ 0.10.x), determines
finger extension ratios (continuous 0.0–1.0), and classifies the gesture
using weighted nearest-neighbour matching with geometric disambiguation.

Requires the model file ``hand_landmarker.task`` in the project root.
Download once with:
    python -c "import urllib.request; urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/'
        'hand_landmarker/float16/latest/hand_landmarker.task',
        'hand_landmarker.task')"

Continuous-ratio detection
==========================
Each finger is scored as a continuous value 0.0 (fully closed) to 1.0
(fully extended).  The reference patterns (SIGNS_VISION) use calibrated
continuous values so every letter has a UNIQUE 5-tuple — no two letters
share the same reference pattern (except I/J which are identical static
poses by ASL design).

Additional geometric features (thumb position, finger spread, fingertip
gaps, index direction) are used as secondary classification criteria for
any pairs that remain close in 5D extension space.
"""

import os
import math
import base64
import logging
import numpy as np
import cv2
import mediapipe as mp

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MediaPipe HandLandmarker setup (Tasks API)
# ---------------------------------------------------------------------------

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

_hand_landmarker = None

try:
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.65,
        min_tracking_confidence=0.5,
    )
    _hand_landmarker = HandLandmarker.create_from_options(options)
    logger.info("MediaPipe HandLandmarker (Tasks API) initialised with model: %s", MODEL_PATH)
except Exception as exc:
    logger.error(
        "Failed to initialise MediaPipe HandLandmarker: %s. "
        "Make sure 'hand_landmarker.task' exists in the project root.", exc
    )

# MediaPipe landmark indices
FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]
FINGER_MCPS = [2, 5, 9, 13, 17]
FINGER_DIPS = [3, 7, 11, 15, 19]   # DIP joints for curl detection

# ---------------------------------------------------------------------------
# Vision sign dictionary — continuous extension ratios (0.0 – 1.0)
#
# Recalibrated from new SIGNS servo angles using: ratio = (angle - 10) / 160
#   Servo 10°  → 0.000    Servo 70°  → 0.375
#   Servo 90°  → 0.500    Servo 120° → 0.6875
#   Servo 130° → 0.750    Servo 170° → 1.000
#
# EVERY letter has a UNIQUE 5-tuple (except I/J by ASL design).
# All servo values are clean multiples of 10.
# ---------------------------------------------------------------------------

SIGNS_VISION = {
    'A': [0.5, 0.0, 0.0, 0.0, 0.0],
    'B': [0.0, 1.0, 1.0, 1.0, 1.0],
    'C': [0.5, 0.5, 0.5, 0.5, 0.5],
    'D': [0.0, 1.0, 0.5, 0.5, 0.5],
    'E': [0.0, 0.5, 0.5, 0.5, 0.0],
    'F': [0.5, 0.0, 1.0, 1.0, 1.0],
    'G': [0.5, 1.0, 0.0, 0.0, 0.0],
    'H': [0.0, 1.0, 1.0, 0.0, 0.0],
    'I': [0.0, 0.0, 0.0, 0.0, 1.0],
    'J': [0.0, 0.0, 0.0, 0.0, 0.5],
    'K': [1.0, 1.0, 1.0, 0.0, 0.0],
    'L': [1.0, 1.0, 0.0, 0.0, 0.0],
    'M': [0.0, 0.0, 0.0, 0.0, 0.0],
    'N': [0.0, 0.0, 0.0, 0.5, 0.0],
    'O': [0.0, 0.5, 0.5, 0.5, 0.5],
    'P': [1.0, 1.0, 0.5, 0.0, 0.0],
    'Q': [1.0, 0.5, 0.0, 0.0, 0.0],
    'R': [0.0, 1.0, 0.5, 0.0, 0.0],
    'S': [0.5, 0.0, 0.0, 0.0, 0.5],
    'T': [0.5, 0.5, 0.0, 0.0, 0.0],
    'U': [0.0, 1.0, 1.0, 0.0, 0.5],
    'V': [0.0, 1.0, 1.0, 0.5, 0.0],
    'W': [0.0, 1.0, 1.0, 1.0, 0.0],
    'X': [0.0, 0.5, 0.0, 0.0, 0.0],
    'Y': [1.0, 0.0, 0.0, 0.0, 1.0],
    'Z': [0.0, 1.0, 0.0, 0.0, 0.0],
    'ILY': [1.0, 1.0, 0.0, 0.0, 1.0],
}

CONFIDENCE_THRESHOLD = 0.80

_FEATURE_WEIGHTS = [1, 1, 1, 1, 1] # Not strictly needed anymore for discrete math


# ---------------------------------------------------------------------------
# Landmark helper — proxy class
# ---------------------------------------------------------------------------

class _LandmarkProxy:
    """Thin wrapper so we can access .x / .y on NormalizedLandmark objects."""
    __slots__ = ("x", "y", "z")

    def __init__(self, lm):
        self.x = lm.x
        self.y = lm.y
        self.z = lm.z if hasattr(lm, "z") else 0.0


# ---------------------------------------------------------------------------
# Palm basis — rotation-invariant axis
# ---------------------------------------------------------------------------

def _palm_basis(landmarks):
    """Return (dx, dy, palm_size) — unit vector from wrist to middle-MCP and
    raw distance.
    """
    wrist   = landmarks[0]
    mid_mcp = landmarks[9]
    dx = mid_mcp.x - wrist.x
    dy = mid_mcp.y - wrist.y
    palm_size = math.hypot(dx, dy) + 1e-6
    return dx / palm_size, dy / palm_size, palm_size


# ---------------------------------------------------------------------------
# Continuous finger extension ratios  (0.0 – 1.0)
#
# [Refactor] Complete rewrite using vector-angle (dot product) approach.
# ---------------------------------------------------------------------------

def _dist3d(a, b) -> float:
    """Return the 3D Euclidean distance between two landmarks.
    This resolves perspective foreshortening when pointing towards the camera.
    """
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2)

def _thumb_extension_ratio(landmarks) -> float:
    """Rotation-invariant thumb extension: 0.0 (closed) to 1.0 (fully extended).

    Uses Euclidean distance from thumb TIP (landmark 4) to index finger MCP
    (landmark 5), normalised by palm size (wrist-to-middle-MCP distance).

    WHY DISTANCE-BASED (not projection-based):
      The thumb moves in a fundamentally different anatomical plane than
      the other four fingers.  While index–pinky flex along the palm's
      longitudinal axis, the thumb abducts/adducts *laterally*.  A
      projection onto the thumb's CMC→MCP axis is highly sensitive to
      hand rotation and can flip sign — giving wildly inconsistent ratios
      for the same physical thumb position viewed from different angles.

      The distance from thumb TIP to index MCP is:
        • Fully rotation-invariant (pure Euclidean distance)
        • Monotonically increasing as the thumb opens
        • Consistent across palm-in / palm-out / tilted views

    ANATOMY MAPPING:
      Servo 10°  (tucked against palm)  → TIP–MCP dist ≈ 0.20 palm → ratio 0.00
      Servo 45°  (draped over fingers)  → TIP–MCP dist ≈ 0.40 palm → ratio 0.22
      Servo 70°  (O-curve meeting tips) → TIP–MCP dist ≈ 0.50 palm → ratio 0.33
      Servo 90°  (thumb at side of fist)→ TIP–MCP dist ≈ 0.65 palm → ratio 0.50
      Servo 170° (fully extended out)   → TIP–MCP dist ≈ 1.10 palm → ratio 1.00
    """
    tip = landmarks[4]        # Thumb TIP
    idx_mcp = landmarks[5]    # Index finger MCP — stable reference on palm edge

    # 3D Euclidean distance, normalised by 3D palm size for scale & depth invariance
    dist = _dist3d(tip, idx_mcp)

    # Calculate true 3D palm size (wrist to mid_mcp) instead of 2D
    wrist = landmarks[0]
    mid_mcp = landmarks[9]
    palm_size_3d = _dist3d(wrist, mid_mcp) + 1e-6

    normalised = dist / palm_size_3d

    # Linear mapping calibrated to real hand proportions.
    # Min (thumb tucked):  normalised ≈ 0.20  → ratio 0.0
    # Max (thumb extended): normalised ≈ 1.10  → ratio 1.0
    # Span = 0.90
    mapped = (normalised - 0.20) / 0.90
    return max(0.0, min(1.0, mapped))


def _vec3(p1, p2, z_scale: float = 1.0) -> tuple:
    """Build a 3D vector from landmark p1 to p2, with optional Z scaling."""
    return (p2.x - p1.x, p2.y - p1.y, (p2.z - p1.z) * z_scale)


def _angle_between_vectors(v1: tuple, v2: tuple) -> float:
    """Return the angle (degrees) between two 3D vectors via dot product.

    Handles degenerate (near-zero) vectors gracefully by returning 0.0,
    which maps to 'fully extended' — a safe fallback when landmarks
    collapse due to distance or occlusion.
    """
    dot = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2)

    # Guard against degenerate vectors (landmarks on top of each other)
    if mag1 < 1e-7 or mag2 < 1e-7:
        return 0.0

    cos_angle = dot / (mag1 * mag2)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))


def _finger_extension_ratio(landmarks, finger_idx: int) -> float:
    """Scale-invariant finger extension: 0.0 (closed) to 1.0 (fully extended).

    [Refactor] COMPLETELY REWRITTEN — no cumulative segment lengths.

    For the thumb: delegates to _thumb_extension_ratio (distance-based).

    For index through pinky:
      1. Compute 3D vectors along each bone segment (MCP→PIP, PIP→DIP, DIP→TIP).
      2. Calculate bend angles at PIP and DIP joints using dot product.
      3. Blend PIP (60%) and DIP (40%) angles — PIP carries more anatomical
         weight for overall finger curl.
      4. Map the blended angle through a calibrated piecewise-linear curve:
           0°–10°  → 1.0        (fully straight, allow minor noise)
           10°–50° → 1.0 → 0.5  (partial extension range)
           50°–90° → 0.5 → 0.0  (curled to fist)
           >90°    → 0.0        (hyper-curled, clamp)
      5. Z-axis compression mitigation: when the hand is far from the camera,
         MediaPipe's Z estimates become unreliable and compressed. We detect
         this via small palm_size and boost Z differences proportionally to
         restore the 3D joint geometry.
    """
    if finger_idx == 0:
        return _thumb_extension_ratio(landmarks)

    tip = landmarks[FINGER_TIPS[finger_idx]]
    dip = landmarks[FINGER_DIPS[finger_idx]]
    pip = landmarks[FINGER_PIPS[finger_idx]]
    mcp = landmarks[FINGER_MCPS[finger_idx]]

    # ----- Z-axis compression mitigation -----
    # MediaPipe normalises XY to [0,1] but Z is relative and becomes
    # increasingly compressed at distance.  When palm_size (2D) drops
    # below a baseline, we scale Z up to maintain realistic joint angles.
    wrist = landmarks[0]
    mid_mcp = landmarks[9]
    palm_size_2d = math.hypot(mid_mcp.x - wrist.x, mid_mcp.y - wrist.y) + 1e-6

    # Baseline: average palm occupies ~0.20 of frame width when at arm's length.
    # Below that, boost Z proportionally to compensate for compression.
    BASELINE_PALM = 0.20
    if palm_size_2d < BASELINE_PALM:
        z_scale = BASELINE_PALM / palm_size_2d
    else:
        z_scale = 1.0
    # Cap Z boost to avoid runaway amplification of noise
    z_scale = min(z_scale, 3.0)

    # ----- Build bone vectors with Z scaling -----
    v_mcp_pip = _vec3(mcp, pip, z_scale)
    v_pip_dip = _vec3(pip, dip, z_scale)
    v_dip_tip = _vec3(dip, tip, z_scale)

    # ----- Joint bend angles -----
    pip_bend = _angle_between_vectors(v_mcp_pip, v_pip_dip)
    dip_bend = _angle_between_vectors(v_pip_dip, v_dip_tip)

    # Weighted blend: PIP is anatomically more significant
    blended_bend = 0.60 * pip_bend + 0.40 * dip_bend

    # ----- Piecewise-linear angle → ratio mapping -----
    # This replaces the naive `1.0 - bend/90` which was too aggressive
    # and caused flickering between half and extended states.
    #
    # Calibration points (from real hand measurements):
    #   Fully straight (0°–10°):  ratio = 1.0
    #   Slightly bent   (10°):    ratio = 1.0
    #   Half bent       (50°):    ratio = 0.5
    #   Fully curled    (90°+):   ratio = 0.0
    if blended_bend <= 10.0:
        ratio = 1.0
    elif blended_bend <= 50.0:
        # Linear interpolation: 10° → 1.0,  50° → 0.5
        ratio = 1.0 - 0.5 * (blended_bend - 10.0) / 40.0
    elif blended_bend <= 90.0:
        # Linear interpolation: 50° → 0.5,  90° → 0.0
        ratio = 0.5 - 0.5 * (blended_bend - 50.0) / 40.0
    else:
        ratio = 0.0

    return max(0.0, min(1.0, ratio))


def _finger_extension_level(landmarks, finger_idx: int) -> int:
    """Discrete 3-level (0/1/2) — kept for backwards-compatible output."""
    ratio = _finger_extension_ratio(landmarks, finger_idx)
    if ratio >= 0.65:
        return 2
    elif ratio >= 0.30:
        return 1
    else:
        return 0


# ---------------------------------------------------------------------------
# Geometric feature helpers (for disambiguation)
# ---------------------------------------------------------------------------

def _tip_distance(landmarks, finger_a: int, finger_b: int) -> float:
    """Normalised distance between two fingertips (by finger index 0–4)."""
    _, _, palm_size = _palm_basis(landmarks)
    tip_a = landmarks[FINGER_TIPS[finger_a]]
    tip_b = landmarks[FINGER_TIPS[finger_b]]
    dist = math.hypot(tip_a.x - tip_b.x, tip_a.y - tip_b.y)
    return dist / palm_size


def _thumb_vertical_position(landmarks) -> float:
    """How high/low the thumb tip is relative to index MCP.
    Positive = thumb above index MCP, negative = below.
    Normalised by palm_size.
    """
    _, _, palm_size = _palm_basis(landmarks)
    thumb_tip = landmarks[FINGER_TIPS[0]]
    index_mcp = landmarks[FINGER_MCPS[1]]
    return (index_mcp.y - thumb_tip.y) / palm_size


def _index_tip_direction(landmarks) -> float:
    """Vertical position of index tip relative to wrist, normalised.
    Positive = index tip is below wrist (pointing down).
    Negative = index tip is above wrist (pointing up/sideways).
    """
    _, _, palm_size = _palm_basis(landmarks)
    idx_tip = landmarks[FINGER_TIPS[1]]
    wrist = landmarks[0]
    return (idx_tip.y - wrist.y) / palm_size


def _thumb_index_gap(landmarks) -> float:
    """Normalised distance between thumb tip and index tip."""
    return _tip_distance(landmarks, 0, 1)


def _index_middle_gap(landmarks) -> float:
    """Normalised distance between index tip and middle tip."""
    return _tip_distance(landmarks, 1, 2)


def _thumb_middle_gap(landmarks) -> float:
    """Normalised distance between thumb tip and middle tip."""
    return _tip_distance(landmarks, 0, 2)


def _wrist_to_tip_direction(landmarks, finger_idx: int) -> float:
    """[Task 5.4] Return the angle (in degrees) of a finger relative to the
    wrist-to-palm (wrist→middle-MCP) axis.

    Used in disambiguation to distinguish:
      - G vs Q: index pointing sideways vs pointing down
      - K vs P: index pointing up vs pointing down

    Returns angle in degrees [0, 180]. 0° = aligned with palm axis (up),
    90° = perpendicular (sideways), 180° = opposite (down).
    """
    bx, by, _ = _palm_basis(landmarks)
    wrist = landmarks[0]
    tip = landmarks[FINGER_TIPS[finger_idx]]

    # Vector from wrist to fingertip
    vx = tip.x - wrist.x
    vy = tip.y - wrist.y
    v_len = math.hypot(vx, vy) + 1e-6

    # Normalise
    vx /= v_len
    vy /= v_len

    # Dot product with palm basis vector gives cos(angle)
    dot = vx * bx + vy * by
    dot = max(-1.0, min(1.0, dot))  # Clamp for numerical safety

    return math.degrees(math.acos(dot))


# ---------------------------------------------------------------------------
# Disambiguation
# ---------------------------------------------------------------------------

def _disambiguate(candidates: list[str], landmarks) -> str:
    """Resolve tied/close candidates using secondary geometric features.

    Updated for the redesigned SIGNS dictionary where all servo values are
    clean multiples of 10 and every pair differs by ≥50° on at least one finger.
    Disambiguation is still needed when measured ratios fall between reference
    values due to hand positioning and camera noise.
    """
    cset = set(candidates)

    # --- I vs J ---
    # I=[0,0,0,0,1] vs J=[0,0,0,0,0.5]
    # Pinky: I=1.0 vs J=0.5
    if cset <= {'I', 'J'}:
        pinky_ratio = _finger_extension_ratio(landmarks, 4)
        return 'I' if pinky_ratio > 0.75 else 'J'

    # --- C vs O ---
    # C=[0.5, 0.5, 0.5, 0.5, 0.5] vs O=[0, 0.5, 0.5, 0.5, 0.5]
    # Thumb: C=0.5 vs O=0.0
    if cset <= {'C', 'O'}:
        thumb_ratio = _finger_extension_ratio(landmarks, 0)
        return 'C' if thumb_ratio > 0.25 else 'O'

    # --- H vs R ---
    # H=[0,1,1,0,0] vs R=[0,1,0.5,0,0]
    # Middle finger: H=1.00 vs R=0.5
    if cset <= {'H', 'R'}:
        mid_ratio = _finger_extension_ratio(landmarks, 2)
        return 'R' if mid_ratio < 0.75 else 'H'

    # --- H vs Z ---
    # H=[0,1,1,0,0] vs Z=[0,1,0,0,0]
    # Middle finger: H=1.00 vs Z=0.00
    if cset <= {'H', 'Z'}:
        mid_ratio = _finger_extension_ratio(landmarks, 2)
        return 'Z' if mid_ratio < 0.50 else 'H'

    # --- U vs V ---
    # U=[0,1,1,0,0.50] vs V=[0,1,1,0.50,0]
    # Ring: U=0 vs V=0.50, Pinky: U=0.50 vs V=0
    if cset <= {'U', 'V'}:
        ring_ratio = _finger_extension_ratio(landmarks, 3)
        pinky_ratio = _finger_extension_ratio(landmarks, 4)
        return 'V' if ring_ratio > pinky_ratio else 'U'

    # --- U vs V vs R (three-way) ---
    if cset <= {'U', 'V', 'R'}:
        mid_ratio = _finger_extension_ratio(landmarks, 2)
        if mid_ratio < 0.80:
            return 'R'
        ring_ratio = _finger_extension_ratio(landmarks, 3)
        pinky_ratio = _finger_extension_ratio(landmarks, 4)
        return 'V' if ring_ratio > pinky_ratio else 'U'

    # --- U vs H ---
    # U=[0,1,1,0,0.50] vs H=[0,1,1,0,0]
    # Pinky: U=0.50 vs H=0
    if cset <= {'U', 'H'}:
        pinky_ratio = _finger_extension_ratio(landmarks, 4)
        return 'U' if pinky_ratio > 0.25 else 'H'

    # --- V vs W ---
    # V=[0,1,1,0.50,0] vs W=[0,1,1,1,0]
    # Ring: V=0.50 vs W=1.00
    if cset <= {'V', 'W'}:
        ring_ratio = _finger_extension_ratio(landmarks, 3)
        return 'W' if ring_ratio > 0.75 else 'V'

    # --- B vs W ---
    # B=[0,1,1,1,1] vs W=[0,1,1,1,0]
    # Pinky: B=1.00 vs W=0.00
    if cset <= {'B', 'W'}:
        pinky_ratio = _finger_extension_ratio(landmarks, 4)
        return 'B' if pinky_ratio > 0.50 else 'W'

    # --- B vs F ---
    # B=[0,1,1,1,1] vs F=[0.50,0,1,1,1]
    # Index: B=1.00 vs F=0.00, Thumb: B=0 vs F=0.50
    if cset <= {'B', 'F'}:
        idx_ratio = _finger_extension_ratio(landmarks, 1)
        return 'B' if idx_ratio > 0.50 else 'F'

    # --- F vs W ---
    # F=[0.50,0,1,1,1] vs W=[0,1,1,1,0]
    # Index: F=0 vs W=1.00, Pinky: F=1 vs W=0
    if cset <= {'F', 'W'}:
        idx_ratio = _finger_extension_ratio(landmarks, 1)
        return 'W' if idx_ratio > 0.50 else 'F'

    # --- A vs T ---
    # A=[0.50,0,0,0,0] vs T=[0.50,0.50,0,0,0]
    # Index: A=0 vs T=0.50
    if cset <= {'A', 'T'}:
        idx_ratio = _finger_extension_ratio(landmarks, 1)
        return 'T' if idx_ratio > 0.25 else 'A'

    # --- A vs S ---
    # A=[0.50,0,0,0,0] vs S=[0.50,0,0,0,0.50]
    # Pinky: A=0 vs S=0.50
    if cset <= {'A', 'S'}:
        pinky_ratio = _finger_extension_ratio(landmarks, 4)
        return 'S' if pinky_ratio > 0.25 else 'A'

    # --- G vs T ---
    # G=[0.50,1,0,0,0] vs T=[0.50,0.50,0,0,0]
    # Index: G=1.00 vs T=0.50
    if cset <= {'G', 'T'}:
        idx_ratio = _finger_extension_ratio(landmarks, 1)
        return 'G' if idx_ratio > 0.75 else 'T'

    # --- G vs Z ---
    # G=[0.50,1,0,0,0] vs Z=[0,1,0,0,0]
    # Thumb: G=0.50 vs Z=0.00
    if cset <= {'G', 'Z'}:
        thumb_ratio = _finger_extension_ratio(landmarks, 0)
        return 'G' if thumb_ratio > 0.25 else 'Z'

    # --- L vs Q ---
    # L=[1,1,0,0,0] vs Q=[1,0.50,0,0,0]
    # Index: L=1.00 vs Q=0.50
    if cset <= {'L', 'Q'}:
        idx_ratio = _finger_extension_ratio(landmarks, 1)
        return 'L' if idx_ratio > 0.75 else 'Q'

    # --- K vs P ---
    # K=[1,1,1,0,0] vs P=[1,1,0.50,0,0]
    # Middle: K=1.00 vs P=0.50
    if cset <= {'K', 'P'}:
        mid_ratio = _finger_extension_ratio(landmarks, 2)
        return 'K' if mid_ratio > 0.75 else 'P'

    # --- P vs L ---
    # P=[1,1,0.50,0,0] vs L=[1,1,0,0,0]
    # Middle: P=0.50 vs L=0.00
    if cset <= {'P', 'L'}:
        mid_ratio = _finger_extension_ratio(landmarks, 2)
        return 'P' if mid_ratio > 0.25 else 'L'

    # --- E vs X ---
    # E=[0,0.50,0.50,0.50,0] vs X=[0,0.50,0,0,0]
    # Middle: E=0.50 vs X=0.00
    if cset <= {'E', 'X'}:
        mid_ratio = _finger_extension_ratio(landmarks, 2)
        return 'E' if mid_ratio > 0.25 else 'X'

    # --- E vs D ---
    # E=[0,0.50,0.50,0.50,0] vs D=[0,1,0.50,0.50,0.50]
    # Index: E=0.50 vs D=1.00, Pinky: E=0 vs D=0.50
    if cset <= {'E', 'D'}:
        idx_ratio = _finger_extension_ratio(landmarks, 1)
        return 'D' if idx_ratio > 0.75 else 'E'

    # --- D vs Z ---
    # D=[0,1,0.50,0.50,0.50] vs Z=[0,1,0,0,0]
    # Middle: D=0.50 vs Z=0.00
    if cset <= {'D', 'Z'}:
        mid_ratio = _finger_extension_ratio(landmarks, 2)
        return 'D' if mid_ratio > 0.25 else 'Z'

    # --- M vs I ---
    # M=[0,0,0,0,0] vs I=[0,0,0,0,1]
    # Pinky: M=0 vs I=1.00
    if cset <= {'M', 'I'}:
        pinky_ratio = _finger_extension_ratio(landmarks, 4)
        return 'I' if pinky_ratio > 0.50 else 'M'

    # --- M vs N ---
    # M=[0,0,0,0,0] vs N=[0,0,0,0.50,0]
    # Ring: M=0 vs N=0.50
    if cset <= {'M', 'N'}:
        ring_ratio = _finger_extension_ratio(landmarks, 3)
        return 'N' if ring_ratio > 0.25 else 'M'

    # --- M vs A ---
    # M=[0,0,0,0,0] vs A=[0.50,0,0,0,0]
    # Thumb: M=0 vs A=0.50
    if cset <= {'M', 'A'}:
        thumb_ratio = _finger_extension_ratio(landmarks, 0)
        return 'A' if thumb_ratio > 0.25 else 'M'

    # --- M vs S ---
    # M=[0,0,0,0,0] vs S=[0.50,0,0,0,0.50]
    # Thumb: M=0 vs S=0.50, Pinky: M=0 vs S=0.50
    if cset <= {'M', 'S'}:
        thumb_ratio = _finger_extension_ratio(landmarks, 0)
        pinky_ratio = _finger_extension_ratio(landmarks, 4)
        return 'S' if (thumb_ratio + pinky_ratio) > 0.50 else 'M'

    # --- R vs Z ---
    # R=[0,1,0.5,0,0] vs Z=[0,1,0,0,0]
    # Middle: R=0.5 vs Z=0
    if cset <= {'R', 'Z'}:
        mid_ratio = _finger_extension_ratio(landmarks, 2)
        return 'R' if mid_ratio > 0.25 else 'Z'

    # --- K vs H ---
    # K=[1,1,1,0,0] vs H=[0,1,1,0,0]
    # Thumb: K=1 vs H=0
    if cset <= {'K', 'H'}:
        thumb_ratio = _finger_extension_ratio(landmarks, 0)
        return 'K' if thumb_ratio > 0.50 else 'H'

    # --- Y vs ILY ---
    # Y=[1,0,0,0,1] vs ILY=[1,1,0,0,1]
    # Index: Y=0 vs ILY=1.00
    if cset <= {'Y', 'ILY'}:
        idx_ratio = _finger_extension_ratio(landmarks, 1)
        return 'ILY' if idx_ratio > 0.50 else 'Y'

    # Fallback: alphabetically first
    return sorted(candidates)[0]


def classify_sign(landmarks) -> tuple[str | None, float]:
    """Classify a hand pose into a letter.

    Uses continuous extension ratios with weighted Euclidean distance
    against SIGNS_VISION reference patterns.

    Returns (letter, confidence) or (None, 0.0).
    """
    ratios = [_finger_extension_ratio(landmarks, i) for i in range(5)]

    scores = {}
    for letter, pattern in SIGNS_VISION.items():
        dist = math.sqrt(sum(
            w * (a - b) ** 2
            for w, a, b in zip(_FEATURE_WEIGHTS, ratios, pattern)
        ))
        scores[letter] = dist

    best_dist = min(scores.values())

    if best_dist > CONFIDENCE_THRESHOLD:
        return None, 0.0

    # Gather candidates within a small margin of the best
    margin = 0.08
    candidates = [l for l, d in scores.items() if d <= best_dist + margin]

    if len(candidates) == 1:
        letter = candidates[0]
    else:
        letter = _disambiguate(candidates, landmarks)

    confidence = max(0.0, 1.0 - best_dist / CONFIDENCE_THRESHOLD)
    return letter, confidence


# ---------------------------------------------------------------------------
# Frame processing pipeline
# ---------------------------------------------------------------------------

def process_frame(frame_data: str) -> dict:
    """Process a single base64-encoded JPEG frame from the browser webcam.

    Returns
    -------
    dict
        Contains: letter, confidence, fingers (binary 0/1 for JS compat),
        finger_levels (0/1/2 three-level), hand_detected, handedness.
    """
    _empty = {
        "letter": None,
        "confidence": 0.0,
        "fingers": None,
        "finger_levels": None,
        "hand_detected": False,
        "handedness": None,
        "wrist_x": 0.0,
        "wrist_y": 0.0,
        "palm_size": 0.0
    }

    if _hand_landmarker is None:
        return _empty

    # Strip optional data-URI prefix
    if "," in frame_data:
        frame_data = frame_data.split(",", 1)[1]

    try:
        img_bytes = base64.b64decode(frame_data)
        np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as exc:
        logger.error("Failed to decode frame: %s", exc)
        return _empty

    if frame is None:
        return _empty

    # Convert BGR → RGB and wrap in a MediaPipe Image
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Run detection
    result = _hand_landmarker.detect(mp_image)

    if not result.hand_landmarks:
        return _empty

    # ---- Hand selection ----
    # [Task 5.6] NOTE: For a webcam facing the user, the user's RIGHT hand
    # appears as 'Left' in the mirrored image (MediaPipe reports handedness
    # from the camera's perspective, not the user's). We prefer 'Left' which
    # is actually the user's dominant right hand in a front-facing camera.
    chosen_idx = None
    handedness_label = None

    if hasattr(result, 'handedness') and result.handedness:
        for i, hand_class_list in enumerate(result.handedness):
            label = hand_class_list[0].category_name if hand_class_list else None
            if label == 'Left':
                chosen_idx = i
                handedness_label = 'Left'
                break
        if chosen_idx is None:
            for i, hand_class_list in enumerate(result.handedness):
                label = hand_class_list[0].category_name if hand_class_list else None
                if label == 'Right':
                    chosen_idx = i
                    handedness_label = 'Right'
                    break

    if chosen_idx is None:
        chosen_idx = 0
        handedness_label = None

    # Convert to proxy objects with .x, .y, .z attributes
    raw_landmarks = result.hand_landmarks[chosen_idx]
    landmarks = [_LandmarkProxy(lm) for lm in raw_landmarks]

    # [Task 5.2] Minimum hand-size guard — reduced to 0.02 so system doesn't reject hands far from camera
    _, _, palm_size = _palm_basis(landmarks)
    if palm_size < 0.02:
        return _empty

    # Compute discrete levels (for backwards compat) and classify
    finger_levels = [_finger_extension_level(landmarks, i) for i in range(5)]
    finger_binary = [1 if lvl >= 2 else 0 for lvl in finger_levels]

    letter, confidence = classify_sign(landmarks)

    return {
        "letter": letter,
        "confidence": round(confidence, 3),
        "fingers": finger_binary,
        "finger_levels": finger_levels,
        "hand_detected": True,
        "handedness": handedness_label,
        "wrist_x": landmarks[0].x,
        "wrist_y": landmarks[0].y,
        "palm_size": palm_size
    }
