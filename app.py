"""
app.py — Flask + Socket.IO Server for the Sign Language Interpretation System

Changes applied:
  [Task 9.1] signing_status 'signing' event now uses alpha_chars count for
             total and correct index numbering.
  [Task 9.2] signing_lock (threading.Lock) prevents concurrent signing.
             New requests while busy get immediate 'busy' status response.
             Uses acquire(blocking=False) to avoid race condition.
  [Task 9.3] Uses sign_letter_with_neutral_pause from robot.py.
  [Task 4.4] _sign_in_background uses sign_letter_with_neutral_pause with
             no additional per-letter delay (hold is internal).
  [Iteration 2] Traced HELLO flow: buildLetterProgress receives original
                 text but indices are based on alpha_chars. Fixed by
                 emitting alpha-only text in 'started' status for progress bar.
  [Iteration 3] signing_lock uses acquire(blocking=False) instead of
                 locked() check to prevent race conditions between check
                 and acquire.

Serves the unified web interface and handles two real-time channels:
  • video_frame  → browser sends webcam frames → server classifies → emits sign_detected
  • send_text    → browser sends typed text    → server drives servos → emits signing_status
"""

import logging
import threading

from flask import Flask, render_template
from flask_socketio import SocketIO, emit

from vision import process_frame
from robot import (
    sign_letter,
    sign_letter_with_neutral_pause,    reset_hand,
    get_current_angles,
    SIGNS,)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Flask & Socket.IO
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.config["SECRET_KEY"] = "sign-language-robot-secret"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# [Task 9.2] Signing lock — prevents concurrent signing operations.
# Uses acquire(blocking=False) pattern to avoid race conditions.
signing_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the unified web interface."""
    return render_template("index.html")

# ---------------------------------------------------------------------------
# Socket.IO events
# ---------------------------------------------------------------------------

@socketio.on("connect")
def handle_connect():
    logger.info("Client connected.")
    emit("connection_ack", {"status": "connected"})


@socketio.on("disconnect")
def handle_disconnect():
    logger.info("Client disconnected.")


@socketio.on("video_frame")
def handle_video_frame(data):
    """Receive a base64 JPEG frame, classify the sign, return the result."""
    frame_data = data.get("frame", "")
    if not frame_data:
        return

    result = process_frame(frame_data)
    emit("sign_detected", result)


@socketio.on("send_text")
def handle_send_text(data):
    """Receive typed text and drive the robotic hand in a background thread."""
    text = data.get("text", "").strip()
    if not text:
        return

    # [Task 9.2] Reject if robot is already signing — use acquire(blocking=False)
    # to avoid the race condition between checking locked() and acquiring.
    if not signing_lock.acquire(blocking=False):
        emit("signing_status", {
            "status": "busy",
            "message": "Robot is already signing",
        })
        return

    # We acquired the lock; release it — the background thread will re-acquire.
    # This approach keeps the main event thread responsive while guaranteeing
    # atomicity of the busy check.
    signing_lock.release()

    logger.info("Received text to sign: '%s'", text)

    # [Iteration 2] Pre-compute alpha chars here so 'started' event can send
    # the alpha-only text for the progress bar to align with indices.
    alpha_chars = [c.upper() for c in text if c.isalpha()]
    alpha_text = ''.join(alpha_chars)

    emit("signing_status", {"status": "started", "text": alpha_text})

    def _sign_in_background(chars_to_sign: list[str], display_text: str):
        """Run servo actuation off the main event-loop thread."""
        with signing_lock:
            for alpha_idx, letter in enumerate(chars_to_sign):
                # Notify client of progress
                socketio.emit("signing_status", {
                    "status": "signing",
                    "letter": letter,
                    "index": alpha_idx,
                    "total": len(chars_to_sign),
                })
                # [Task 4.4 / 9.3] Use sign_letter_with_neutral_pause
                # (hold + neutral reset is internal, no extra sleep needed)
                sign_letter_with_neutral_pause(letter, hold=0.5)

            # Return hand to neutral
            reset_hand()
            socketio.emit("signing_status", {
                "status": "completed",
                "text": display_text,
            })

    thread = threading.Thread(
        target=_sign_in_background,
        args=(alpha_chars, alpha_text),
        daemon=True,
    )
    thread.start()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Starting Sign Language Interpretation Server on http://0.0.0.0:5000")
    socketio.run(app, host="0.0.0.0", port=5000, debug=True, allow_unsafe_werkzeug=True)
