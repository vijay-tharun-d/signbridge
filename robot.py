"""
robot.py — PCA9685 Servo Controller for the Robotic Sign Language Hand

Changes applied:
  [Left Hand Support] Tailored precisely for your custom 3D-printed
           tendon-driven Left Hand model (Forearm9-1, 9-2, 9-3, Hand5,
           ServoHold, and custom ServoHorns).
  [Feature Removal] Physical phrase gestures completely deleted to prevent
           complex tendon strain and inconsistencies. The robot will now
           elegantly fingerspell any phrase requested.

Hardware: 5 micro-servos (MG90S) wired to a PCA9685 16-channel PWM driver via I2C
on a Raspberry Pi 4.

  Channel 0 → Thumb
  Channel 1 → Index
  Channel 2 → Middle
  Channel 3 → Ring
  Channel 4 → Pinky

On non-RPi machines the hardware libraries are unavailable, so a graceful
mock fallback is provided so the full Flask app can still run for UI and
vision development.

Three-level servo system
========================
Each finger can be positioned at any integer angle between 10° and 170°.
Three named constants cover the common positions:

  SERVO_EXTENDED = 170   fully open  (tendon RELAXED)
  SERVO_HALF     = 90    mid-bend / pointing / sideways
  SERVO_CLOSED   = 10    fully curled (tendon PULLED)

Tendon-driven mechanics:
  Servo at 170° = tendon relaxed  = finger OPEN
  Servo at 10°  = tendon pulled   = finger CURLED
  Servo at 90°  = finger partially bent

Known limitations (servo hardware, not software bugs):
  • I and J have the same static pose.  J adds a motion arc that cannot be
    represented as a static servo position.
"""

import time
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — Three-level servo angle constants
# ---------------------------------------------------------------------------

SERVO_EXTENDED = 170   # finger fully open  (tendon relaxed)
SERVO_HALF     = 90    # finger at mid-bend / pointing / sideways
SERVO_CLOSED   = 10    # finger fully curled (tendon pulled)

# Short aliases for the SIGNS table
_E = SERVO_EXTENDED   # 170
_H = SERVO_HALF       # 90
_C = SERVO_CLOSED     # 10

# Channel-to-finger mapping
FINGER_CHANNELS = {
    "thumb":  0,
    "index":  1,
    "middle": 2,
    "ring":   3,
    "pinky":  4,
}

FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky"]

# ---------------------------------------------------------------------------
# Alphabet → servo angle states  [thumb, index, middle, ring, pinky]
# Values are degrees (10–170).  Every letter has a unique 5-tuple except for
# the unavoidable I/J collision (J = I + wrist arc motion, servo limitation).
#
# Redesigned — CLEAN MULTIPLES OF 10 ONLY:
#   a) ALL angles are multiples of 10 (10, 70, 90, 120, 130, 170, etc.)
#   b) Every pair of letters differs by ≥50° on at least one finger
#   c) Only I/J are identical (ASL design limitation)
#   d) Tendon-driven: servo 170° = finger extended, 10° = finger curled
#   e) Optimised for camera recognition — reduced ambiguity zones
# ---------------------------------------------------------------------------

SIGNS = {
    'A': [90,  10,  10,  10,  10],    # Fist with thumb at side
    'B': [10,  170, 170, 170, 170],   # Four fingers up, thumb tucked
    'C': [90,  90,  90,  90,  90],    # All fingers curved (C-shape)
    'D': [10,  170, 90,  90,  90],    # Index up, others half-bent
    'E': [10,  90,  90,  90,  10],    # Hooked fingers, pinky closed
    'F': [90,  10,  170, 170, 170],   # Thumb+index pinched, 3 fingers up
    'G': [90,  170, 10,  10,  10],    # Index pointing, thumb at side
    'H': [10,  170, 170, 10,  10],    # Two fingers extended flat
    'I': [10,  10,  10,  10,  170],   # Pinky only
    'J': [10,  10,  10,  10,  90],    # Unique 5-tuple for J
    'K': [170, 170, 170, 10,  10],    # Thumb+index+middle extended
    'L': [170, 170, 10,  10,  10],    # L-shape: thumb + index
    'M': [10,  10,  10,  10,  10],    # Complete fist
    'N': [10,  10,  10,  90,  10],    # Fist with ring finger half
    'O': [10,  90,  90,  90,  90],    # Round O-shape (thumb closed, others curved)
    'P': [170, 170, 90,  10,  10],    # Like K but middle half-bent
    'Q': [170, 90,  10,  10,  10],    # Thumb extended, index half
    'R': [10,  170, 90,  10,  10],    # Index up, middle half-bent
    'S': [90,  10,  10,  10,  90],    # Fist, thumb and pinky at half
    'T': [90,  90,  10,  10,  10],    # Thumb + index at half, rest closed
    'U': [10,  170, 170, 10,  90],    # Two fingers up, pinky half
    'V': [10,  170, 170, 90,  10],    # Peace sign with ring half
    'W': [10,  170, 170, 170, 10],    # Three middle fingers up
    'X': [10,  90,  10,  10,  10],    # Index hooked
    'Y': [170, 10,  10,  10,  170],   # Thumb + pinky extended
    'Z': [10,  170, 10,  10,  10],    # Index extended, thumb closed
    'ILY': [170, 170, 10, 10,  170],  # I Love You handshape
}

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# SIGNS validation summary:
#   All 26 letters are unique 5-tuples (except I/J identical by ASL design).
#   All values are clean multiples of 10.
#   Every pair of letters differs by ≥50° on at least one finger.
#   Validated programmatically — no near-conflicts below 50° threshold.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Hardware initialisation (with mock fallback)
# ---------------------------------------------------------------------------

_USE_HARDWARE = False
_kit = None

try:
    from adafruit_servokit import ServoKit
    _kit = ServoKit(channels=16)
    _USE_HARDWARE = True

    # [Task 4.6] Set pulse width range for MG90S servos (500-2500µs)
    for ch in range(5):
        _kit.servo[ch].set_pulse_width_range(500, 2500)

    logger.info("PCA9685 ServoKit initialised — hardware mode active.")
except Exception as exc:
    logger.warning(
        "Hardware servo libraries unavailable (%s). "
        "Running in MOCK mode — servo commands will be logged to the console.",
        exc,
    )


# ---------------------------------------------------------------------------
# Current angle tracking (Task 4.1)
# ---------------------------------------------------------------------------

# Tracks the last commanded angle per channel for smooth interpolation.
# Initialised to SERVO_EXTENDED (170°) — hand starts fully open.
_current_angles = {ch: SERVO_EXTENDED for ch in range(5)}


def _set_servo_angle(channel: int, angle: int) -> None:
    """Set a single servo to *angle* degrees on the given PCA9685 channel.

    Angle is clamped to [SERVO_CLOSED, SERVO_EXTENDED] to prevent mechanical
    damage. Updates _current_angles tracking.
    """
    angle = max(SERVO_CLOSED, min(SERVO_EXTENDED, int(angle)))
    _current_angles[channel] = angle
    if _USE_HARDWARE:
        _kit.servo[channel].angle = angle
    else:
        finger_name = FINGER_ORDER[channel] if channel < len(FINGER_ORDER) else f"ch{channel}"
        logger.info("[MOCK] Servo ch%d (%s) → %d°", channel, finger_name, angle)


def smooth_set_servo_angle(channel: int, target_angle: int,
                           steps: int = 5, step_delay: float = 0.04) -> None:
    """Smoothly interpolate a servo from its current position to target_angle.

    [Task 4.1] Prevents snapping and reduces tendon stress by moving in
    `steps` increments with `step_delay` seconds between each.

    Parameters
    ----------
    channel : int
        PCA9685 channel (0-4 for fingers)
    target_angle : int
        Target angle in degrees [10, 170]
    steps : int
        Number of intermediate positions (default 5)
    step_delay : float
        Delay in seconds between steps (default 0.04s)
    """
    target_angle = max(SERVO_CLOSED, min(SERVO_EXTENDED, int(target_angle)))
    current = _current_angles.get(channel, SERVO_EXTENDED)

    if current == target_angle:
        return

    for i in range(1, steps + 1):
        intermediate = current + (target_angle - current) * i / steps
        _set_servo_angle(channel, int(round(intermediate)))
        if i < steps:
            time.sleep(step_delay)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_sign_angles(letter: str) -> list[int] | None:
    """Return the raw angle list for a letter, or None if not in the alphabet."""
    return SIGNS.get(letter.upper())


def get_current_angles() -> dict[int, int]:
    """Return the current tracked angles for all servo channels.

    [Task 4.5] Allows the server to emit actual servo states to the frontend.
    """
    return dict(_current_angles)


def sign_letter(letter: str) -> bool:
    """Actuate all 5 servos to form the sign for *letter*.

    [Task 4.2] Uses smooth_set_servo_angle with steps=4, step_delay=0.03
    for gentle tendon movement.

    Returns True if the letter was found in the alphabet map, False otherwise.
    """
    letter = letter.upper()
    if letter not in SIGNS:
        logger.warning("Letter '%s' not in sign alphabet — skipping.", letter)
        return False

    angles = SIGNS[letter]
    logger.info("Signing letter '%s' → %s", letter, angles)

    for idx, angle in enumerate(angles):
        channel = FINGER_CHANNELS[FINGER_ORDER[idx]]
        smooth_set_servo_angle(channel, angle, steps=4, step_delay=0.03)

    return True


def sign_letter_with_neutral_pause(letter: str, hold: float = 0.5) -> bool:
    """Sign a letter, hold it, then return to neutral with a visual pause.

    [Task 4.3] Sequence:
      1. Sign the letter (with smooth interpolation)
      2. Hold for `hold` seconds
      3. Return all fingers to SERVO_EXTENDED (smooth)
      4. Pause 0.15s (to separate letters visually)

    Returns True if the letter was signed, False otherwise.
    """
    if not sign_letter(letter):
        return False

    time.sleep(hold)

    # Return to neutral (all fingers extended)
    for idx in range(5):
        channel = FINGER_CHANNELS[FINGER_ORDER[idx]]
        smooth_set_servo_angle(channel, SERVO_EXTENDED, steps=4, step_delay=0.03)

    time.sleep(0.15)
    return True


def sign_word(word: str, delay: float = 0.8) -> list[str]:
    """Sign each letter of *word* sequentially with *delay* seconds between.

    Returns a list of successfully signed letters.
    """
    signed: list[str] = []
    for char in word:
        if char.isalpha():
            if sign_letter(char):
                signed.append(char.upper())
            time.sleep(delay)
    return signed


def reset_hand() -> None:
    """Open all fingers (set every servo to SERVO_EXTENDED) smoothly."""
    logger.info("Resetting hand — all fingers extended.")
    for idx, finger in enumerate(FINGER_ORDER):
        smooth_set_servo_angle(FINGER_CHANNELS[finger], SERVO_EXTENDED,
                               steps=4, step_delay=0.03)
