/**
 * main.js — Socket.IO Client Logic & DOM Manipulation
 *
 * Changes applied:
 *   [Task 1.1] Removed duplicate signing_status handler from voice IIFE.
 *              Mic enable/disable now handled in the single phrase-buttons handler.
 *   [Task 1.2] Voice recognition auto-submits on FINAL result. Interim results
 *              update placeholder text only.
 *   [Task 1.3] stopListening() called after auto-submit to prevent further input.
 *   [Task 6]   Transcript state machine (TranscriptStateMachine) replaces simple
 *              repeat-count approach. States: IDLE → CONFIRMING → CONFIRMED →
 *              COOLDOWN. Space insertion after 8 no-hand frames.
 *   [Task 7]   SIGNS_MAP regenerated from new SIGNS dict using mapping:
 *              ≤30° → 0, 31-130° → 1, 131-170° → 2
 *   [Iteration 2] Traced HELLO flow: progress bar now correctly aligns with
 *                  alpha-only text emitted by server.
 *   [Iteration 3] Voice recognition guards against empty string submission.
 *                  Transcript state machine handles null letter gracefully.
 *
 *   [Refactor] PhraseRecognizer REWRITTEN for distance-agnostic recognition:
 *              • ALL absolute screen thresholds replaced with palm_size-normalised
 *                thresholds (dx / avg_palm_size > threshold).
 *              • Movement, expansion, and direction changes are all computed
 *                relative to the hand's current size so phrases are recognised
 *                accurately regardless of user distance from camera.
 *              • State machine correctly flushes after phrase detection —
 *                history cleared, cooldown applied, TranscriptStateMachine reset
 *                to prevent accidental letter triggers post-phrase.
 *
 * Handles:
 *   1. Camera capture → sends base64 JPEG frames at ~5 FPS
 *   2. Listens for sign_detected → updates transcript via state machine
 *   3. Sends typed text → listens for signing_status → updates UI
 *
 * Three-level finger visualiser:
 *   0 = closed (red), 1 = half (amber), 2 = extended (green)
 */

(() => {
    "use strict";

    // ─── DOM References ──────────────────────────────────────
    const $  = (sel) => document.querySelector(sel);
    const $$ = (sel) => document.querySelectorAll(sel);

    const webcam           = $("#webcam");
    const canvas           = $("#capture-canvas");
    const ctx              = canvas.getContext("2d");
    const btnCameraToggle  = $("#btn-camera-toggle");
    const btnClearTranscript = $("#btn-clear-transcript");
    const detectedLetter   = $("#detected-letter");
    const handStatus       = $("#hand-status");
    const transcriptContent = $("#transcript-content");

    const chatForm         = $("#chat-form");
    const chatInput        = $("#chat-input");
    const chatMessages     = $("#chat-messages");
    const btnSend          = $("#btn-send");
    const btnMic           = $("#btn-mic");

    const robotStatusIcon  = $("#robot-status-icon");
    const robotStatusText  = $("#robot-status-text");
    const letterProgress   = $("#letter-progress");
    const robotStatusBar   = $("#robot-status-bar");

    const connectionBadge  = $("#connection-badge");

    const fingerEls = {
        thumb:  $("#vis-thumb"),
        index:  $("#vis-index"),
        middle: $("#vis-middle"),
        ring:   $("#vis-ring"),
        pinky:  $("#vis-pinky"),
    };

    // ─── State ───────────────────────────────────────────────
    let cameraActive   = false;
    let captureInterval = null;
    let isSigning = false;

    // ─── Socket.IO ───────────────────────────────────────────
    const socket = io();

    socket.on("connect", () => {
        console.log("✅ Socket connected");
        connectionBadge.classList.add("connected");
        connectionBadge.querySelector(".label").textContent = "Connected";
    });

    socket.on("disconnect", () => {
        console.log("❌ Socket disconnected");
        connectionBadge.classList.remove("connected");
        connectionBadge.querySelector(".label").textContent = "Disconnected";
    });

    socket.on("connection_ack", (data) => {
        console.log("Server acknowledged:", data);
    });

    // ─── Camera ──────────────────────────────────────────────
    async function startCamera() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480, facingMode: "user" },
                audio: false,
            });
            webcam.srcObject = stream;
            cameraActive = true;
            btnCameraToggle.innerHTML = '<span class="btn-icon">⏹️</span> Stop Camera';
            btnCameraToggle.classList.add("active");
            startCapture();
        } catch (err) {
            console.error("Camera access denied:", err);
            alert("Camera access is required for Sign-to-Text.\nPlease allow camera permissions and try again.");
        }
    }

    function stopCamera() {
        if (webcam.srcObject) {
            webcam.srcObject.getTracks().forEach((t) => t.stop());
            webcam.srcObject = null;
        }
        cameraActive = false;
        btnCameraToggle.innerHTML = '<span class="btn-icon">🎥</span> Start Camera';
        btnCameraToggle.classList.remove("active");
        stopCapture();
        detectedLetter.textContent = "—";
        updateHandStatus(false);
    }

    btnCameraToggle.addEventListener("click", () => {
        cameraActive ? stopCamera() : startCamera();
    });

    // ─── Frame Capture (~5 FPS) ──────────────────────────────
    function startCapture() {
        stopCapture();
        captureInterval = setInterval(() => {
            if (!cameraActive) return;
            canvas.width  = webcam.videoWidth  || 640;
            canvas.height = webcam.videoHeight || 480;
            ctx.drawImage(webcam, 0, 0, canvas.width, canvas.height);
            const dataURL = canvas.toDataURL("image/jpeg", 0.6);
            socket.emit("video_frame", { frame: dataURL });
        }, 200);   // 200 ms ≈ 5 FPS
    }

    function stopCapture() {
        if (captureInterval) {
            clearInterval(captureInterval);
            captureInterval = null;
        }
    }

    // ─── Transcript State Machine (Task 6) ───────────────────
    //
    // States: IDLE → CONFIRMING → CONFIRMED → COOLDOWN
    //   CONFIRMING: new letter detected, counting consecutive frames
    //   CONFIRMED:  letter appended to transcript (once), moves to COOLDOWN
    //   COOLDOWN:   ignores this letter for COOLDOWN_FRAMES frames
    //   IDLE:       waiting for next letter
    //
    // Space insertion: 8+ consecutive no-hand frames → append space (once)

    const CONFIRM_FRAMES  = 7;    // require sign to be held longer to confirm accuracy and eliminate jitter
    const COOLDOWN_FRAMES = 10;   // frames to ignore after confirming (~2s at 5fps)
    const SPACE_FRAMES    = 8;    // no-hand frames before space insertion

    const TranscriptStateMachine = {
        state: 'IDLE',         // IDLE | CONFIRMING | COOLDOWN
        currentLetter: null,
        frameCount: 0,
        glitchCount: 0,
        cooldownCount: 0,
        noHandCount: 0,
        lastAppendedSpace: false,

        reset() {
            this.state = 'IDLE';
            this.currentLetter = null;
            this.frameCount = 0;
            this.glitchCount = 0;
            this.cooldownCount = 0;
            this.noHandCount = 0;
            this.lastAppendedSpace = false;
        },

        onFrame(letter) {
            // letter is a string (detected letter) or null (no hand / no match)

            if (letter === null || letter === undefined) {
                // No hand detected
                this.noHandCount++;

                if (this.noHandCount >= SPACE_FRAMES && !this.lastAppendedSpace) {
                    appendToTranscript(' ');
                    this.lastAppendedSpace = true;
                }

                // If we were CONFIRMING, the hand disappeared — tolerate up to 2 frames
                if (this.state === 'CONFIRMING') {
                    this.glitchCount++;
                    if (this.glitchCount > 2) {
                        this.state = 'IDLE';
                        this.currentLetter = null;
                        this.frameCount = 0;
                        this.glitchCount = 0;
                    }
                }

                // If COOLDOWN, count down towards IDLE
                if (this.state === 'COOLDOWN') {
                    this.cooldownCount++;
                    if (this.cooldownCount >= COOLDOWN_FRAMES) {
                        this.state = 'IDLE';
                        this.currentLetter = null;
                        this.frameCount = 0;
                        this.cooldownCount = 0;
                    }
                }

                return;
            }

            // A letter was detected — reset no-hand counter
            this.noHandCount = 0;
            this.lastAppendedSpace = false;

            switch (this.state) {
                case 'IDLE':
                    // Start confirming a new letter
                    this.state = 'CONFIRMING';
                    this.currentLetter = letter;
                    this.frameCount = 1;
                    this.glitchCount = 0;
                    break;

                case 'CONFIRMING':
                    if (letter === this.currentLetter) {
                        this.frameCount++;
                        this.glitchCount = 0; // reset glitch count on match
                        if (this.frameCount >= CONFIRM_FRAMES) {
                            // Confirmed! Append once, move to COOLDOWN.
                            this.state = 'COOLDOWN';
                            this.cooldownCount = 0;
                            appendToTranscript(letter);
                        }
                    } else {
                        // Different letter — tolerate up to 2 frames of glitch
                        this.glitchCount++;
                        if (this.glitchCount > 2) {
                            // Too many glitches, restart confirmation
                            this.currentLetter = letter;
                            this.frameCount = 1;
                            this.glitchCount = 0;
                        }
                    }
                    break;

                case 'COOLDOWN':
                    if (letter === this.currentLetter) {
                        // Same letter during cooldown — ignore, count down
                        this.cooldownCount++;
                        if (this.cooldownCount >= COOLDOWN_FRAMES) {
                            this.state = 'IDLE';
                            this.currentLetter = null;
                            this.frameCount = 0;
                            this.cooldownCount = 0;
                        }
                    } else {
                        // Different letter — start confirming the new one
                        this.state = 'CONFIRMING';
                        this.currentLetter = letter;
                        this.frameCount = 1;
                        this.glitchCount = 0;
                        this.cooldownCount = 0;
                    }
                    break;
            }
        }
    };

    // ─── Phrase Recognizer (Distance-Agnostic Temporal Gesture Tracker) ────
    //
    // [Refactor] COMPLETELY REWRITTEN for distance-agnostic recognition.
    //
    // All motion thresholds are normalised by avg_palm_size so the same
    // physical gesture is recognised whether the user is 30cm or 2m from
    // the camera.  The palm_size field (wrist-to-middle-MCP distance in
    // normalised coordinates) serves as the scale reference.
    //
    // After a phrase is detected the recogniser:
    //   1. Clears its entire frame history
    //   2. Enters a cooldown period (skip N frames)
    //   3. Resets the TranscriptStateMachine to prevent the last held
    //      letter from bleeding into the transcript as an individual sign.

    const PhraseRecognizer = {
        history: [],
        MAX_FRAMES: 30, // ~6 seconds sliding window at 5 FPS
        cooldown: 0,

        onFrame(data) {
            // --- Cooldown tick ---
            if (this.cooldown > 0) {
                this.cooldown--;
                // During cooldown, still collect frames but don't detect
            }

            // --- Collect frame data ---
            if (!data || !data.hand_detected || !data.letter) {
                this.history.push(null);
            } else {
                this.history.push({
                    letter: data.letter,
                    x: data.wrist_x,
                    y: data.wrist_y,
                    size: data.palm_size || 0.1  // fallback to prevent div-by-zero
                });
            }

            // --- Trim sliding window ---
            if (this.history.length > this.MAX_FRAMES) {
                this.history.shift();
            }

            // --- Attempt phrase detection (only outside cooldown) ---
            if (this.cooldown === 0) {
                const phrase = this._detectPhrase();
                if (phrase) {
                    appendToTranscript(` [${phrase}] `);
                    // ── Flush state to prevent letter bleed ──
                    this.history = [];
                    this.cooldown = 15;          // ~3s at 5 FPS before next phrase
                    TranscriptStateMachine.reset(); // kill any in-progress letter
                    return true;
                }
            }
            return false;
        },

        /**
         * _detectPhrase — Distance-agnostic phrase detection.
         *
         * Every motion metric is divided by avg_palm_size so the thresholds
         * are in "palm-widths" rather than screen-fractions:
         *
         *   norm_dx = (max_x - min_x) / avg_palm_size
         *     → measures horizontal sweep in palm-widths
         *
         *   norm_dy = (max_y - min_y) / avg_palm_size
         *     → measures vertical sweep in palm-widths
         *
         *   size_change = (last_size - first_size) / avg_palm_size
         *     → measures push/pull (towards/away from camera)
         *
         *   y_change = (last_y - first_y) / avg_palm_size
         *     → measures net downward movement
         *
         * Thresholds are calibrated so that a wave of ~1.5 palm-widths
         * horizontally triggers HELLO, regardless of camera distance.
         */
        _detectPhrase() {
            let validCount = 0;
            let lastValidLetter = null;
            
            // Motion tracking bounds
            let minX = Infinity, maxX = -Infinity;
            let minY = Infinity, maxY = -Infinity;
            let firstSize = 0, lastSize = 0;
            let firstY = 0, lastY = 0;
            let sumSize = 0;
            
            // Frequency map for dominant handshape
            const counts = {};
            let domLetter = null;
            let maxCount = 0;
            
            // --- Single O(N) pass for efficiency ---
            for (let i = 0; i < this.history.length; i++) {
                const f = this.history[i];
                if (f !== null) {
                    validCount++;
                    lastValidLetter = f.letter;
                    
                    if (validCount === 1) {
                        firstSize = f.size;
                        firstY = f.y;
                    }
                    lastSize = f.size;
                    lastY = f.y;
                    
                    if (f.x < minX) minX = f.x;
                    if (f.x > maxX) maxX = f.x;
                    if (f.y < minY) minY = f.y;
                    if (f.y > maxY) maxY = f.y;
                    
                    sumSize += f.size;
                    
                    counts[f.letter] = (counts[f.letter] || 0) + 1;
                    if (counts[f.letter] > maxCount) {
                        maxCount = counts[f.letter];
                        domLetter = f.letter;
                    }
                }
            }
            
            if (validCount < 10) return null; // need sufficient motion data
            
            // --- Short-circuit: ILY handshape → instant phrase ---
            if (lastValidLetter === 'ILY') {
                return "I LOVE YOU";
            }
            
            // Handshape must be consistent
            if (maxCount / validCount < 0.6) return null;

            // --- Compute palm-size-normalised motion metrics ---
            const avg_palm_size = sumSize / validCount;
            // Safety floor to prevent division explosion
            const safe_palm = Math.max(avg_palm_size, 0.03);

            const dx = maxX - minX;
            const dy = maxY - minY;

            // All metrics in palm-widths (scale-invariant)
            const norm_dx      = dx / safe_palm;
            const norm_dy      = dy / safe_palm;
            const size_change  = (lastSize - firstSize) / safe_palm;
            const y_change     = (lastY - firstY) / safe_palm;

            // --- Motion classification ---
            const isHorizontalWave = norm_dx > 1.2 && norm_dy < 1.0;
            const isVerticalNod    = norm_dy > 0.8 && norm_dx < 0.8;
            const isCircular       = norm_dx > 0.9 && norm_dy > 0.9;
            const isExpanding      = size_change > 0.3; // Push towards camera
            const isMovingDown     = y_change > 0.8;

            // --- Smart Phrase Matching (Optimized) ---
            if (domLetter === 'B') {
                if (isHorizontalWave) return "HELLO";
                if (isExpanding)      return "THANK YOU";
                if (isMovingDown)     return "GOOD";
            }
            if (domLetter === 'C' || domLetter === 'O') {
                if (isExpanding)      return "HOW ARE YOU";
            }
            if (domLetter === 'W') {
                if (isHorizontalWave) return "BYE";
            }
            // Existing mappings
            if (domLetter === 'A' || domLetter === 'S') {
                if (isVerticalNod)    return "YES";
                if (isCircular)       return "SORRY";
            }
            if (domLetter === 'H' || domLetter === 'U' || domLetter === 'V') {
                if (isHorizontalWave) return "NO";
            }

            return null;
        }
    };



    // ─── Sign Detection Listener ─────────────────────────────
    socket.on("sign_detected", (data) => {
        updateHandStatus(data.hand_detected);

        if (!data.hand_detected || !data.letter) {
            detectedLetter.textContent = "—";
            updateFingerVisualiser(null, [0,0,0,0,0]);
            // Feed null to state machine (no hand / no letter)
            TranscriptStateMachine.onFrame(null);
            return;
        }

        detectedLetter.textContent = data.letter;

        if (data.finger_levels) {
            updateFingerVisualiser(data.letter, data.finger_levels);
        }

        // Feed data to Phrase Recognizer first
        const phraseDetected = PhraseRecognizer.onFrame(data);
        if (PhraseRecognizer.cooldown > 0 || phraseDetected) {
            // Skip letter recognition if we just detected a phrase or are in cooldown
            TranscriptStateMachine.onFrame(null);
        } else {
            // If no phrase detected, drop down to letter recognition
            if (data.letter === 'ILY') {
                // ILY is handled instantly by phrase recognizer, block from letters
                TranscriptStateMachine.onFrame(null);
            } else {
                TranscriptStateMachine.onFrame(data.letter);
            }
        }
    });

    function updateHandStatus(detected) {
        const dot = handStatus.querySelector(".dot");
        if (detected) {
            dot.className = "dot green";
            handStatus.innerHTML = "";
            handStatus.appendChild(dot);
            handStatus.append(" Hand detected");
        } else {
            dot.className = "dot red";
            handStatus.innerHTML = "";
            handStatus.appendChild(dot);
            handStatus.append(" No hand detected");
        }
    }

    // ─── Transcript ──────────────────────────────────────────
    function appendToTranscript(charOrSpace) {
        const placeholder = transcriptContent.querySelector(".placeholder");
        if (placeholder) placeholder.remove();

        if (charOrSpace === ' ') {
            // Only append space if transcript doesn't already end with one
            const current = transcriptContent.textContent;
            if (current.length > 0 && !current.endsWith(' ')) {
                transcriptContent.textContent += ' ';
            }
        } else {
            transcriptContent.textContent += charOrSpace;
        }
        transcriptContent.scrollTop = transcriptContent.scrollHeight;
    }

    btnClearTranscript.addEventListener("click", () => {
        transcriptContent.innerHTML = '<span class="placeholder">Detected signs will appear here…</span>';
        TranscriptStateMachine.reset();
    });

    // ─── Text-to-Sign (chat) ─────────────────────────────────

    function submitChatText() {
        const text = chatInput.value.trim();
        if (!text || isSigning) return;

        addChatMessage(text, "user");
        socket.emit("send_text", { text });
        chatInput.value = "";
    }

    chatForm.addEventListener("submit", (e) => {
        e.preventDefault();
        submitChatText();
    });

    function addChatMessage(content, type) {
        const div = document.createElement("div");
        div.className = type === "user" ? "user-msg" : "robot-msg";
        div.textContent = content;
        chatMessages.appendChild(div);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // ─── Signing Status Listener (single handler) ────────────
    // [Task 1.1] Single handler covers phrase buttons, mic button, and main UI.
    // No duplicate handler in voice IIFE.
    socket.on("signing_status", (data) => {
        switch (data.status) {
            case "started":
                isSigning = true;
                btnSend.disabled = true;
                btnMic.disabled = true;  // [Task 1.1] Disable mic during signing
                robotStatusIcon.textContent = "⏳";
                robotStatusText.textContent = `Signing: "${data.text}"`;
                robotStatusBar.classList.add("signing-active");
                buildLetterProgress(data.text);
                addChatMessage(`🤖 Signing "${data.text}"…`, "robot");
                // Disable phrase buttons
                _disablePhraseButtons();
                // Stop voice if listening
                if (_voiceIsListening) _voiceStopListening();
                break;

            case "signing":
                robotStatusIcon.textContent = "✋";
                robotStatusText.textContent = `Signing letter: ${data.letter}`;
                highlightProgressLetter(data.index);
                updateFingerVisualiser(data.letter, null);
                break;

            case "completed":
                isSigning = false;
                btnSend.disabled = false;
                btnMic.disabled = false;  // [Task 1.1] Re-enable mic
                robotStatusIcon.textContent = "🤖";
                robotStatusText.textContent = "Idle — ready to sign";
                robotStatusBar.classList.remove("signing-active");
                addChatMessage(`✅ Finished signing "${data.text}"`, "robot");
                resetFingerVisualiser();
                // Enable phrase buttons
                _enablePhraseButtons();
                break;

            case "busy":
                // [Task 9.2] Robot is already signing
                addChatMessage(`⚠️ ${data.message || "Robot is busy"}`, "robot");
                break;

            case "error":
                isSigning = false;
                btnSend.disabled = false;
                btnMic.disabled = false;
                robotStatusIcon.textContent = "🤖";
                robotStatusText.textContent = "Idle — ready to sign";
                robotStatusBar.classList.remove("signing-active");
                if (data.message) {
                    addChatMessage(`⚠️ ${data.message}`, "robot");
                }
                _enablePhraseButtons();
                break;
        }
    });

    // ─── Letter Progress Display ─────────────────────────────
    function buildLetterProgress(text) {
        letterProgress.innerHTML = "";
        for (let i = 0; i < text.length; i++) {
            const span = document.createElement("span");
            span.className = "lp-char";
            span.textContent = text[i];
            span.dataset.index = i;
            letterProgress.appendChild(span);
        }
    }

    function highlightProgressLetter(index) {
        const chars = letterProgress.querySelectorAll(".lp-char");
        chars.forEach((el) => {
            const idx = parseInt(el.dataset.index, 10);
            if (idx < index)       el.className = "lp-char done";
            else if (idx === index) el.className = "lp-char active";
            else                    el.className = "lp-char";
        });
    }

    // ─── Finger Visualiser (three-level) ─────────────────────
    // [Task 7] SIGNS_MAP — finger visualiser levels from new SIGNS dict:
    //   ≤ 30°   → level 0 (closed)
    //   31–130° → level 1 (half)
    //   131–170° → level 2 (extended)
    //
    // New SIGNS values (all multiples of 10):
    //   A: [90,10,10,10,10]       → [1,0,0,0,0]
    //   B: [10,170,170,170,170]   → [0,2,2,2,2]
    //   C: [90,90,90,90,90]       → [1,1,1,1,1]
    //   D: [10,170,90,90,90]      → [0,2,1,1,1]
    //   E: [10,90,90,90,10]       → [0,1,1,1,0]
    //   F: [90,10,170,170,170]    → [1,0,2,2,2]
    //   G: [90,170,10,10,10]      → [1,2,0,0,0]
    //   H: [10,170,170,10,10]     → [0,2,2,0,0]
    //   I: [10,10,10,10,170]      → [0,0,0,0,2]
    //   J: [10,10,10,10,90]       → [0,0,0,0,1]
    //   K: [170,170,170,10,10]    → [2,2,2,0,0]
    //   L: [170,170,10,10,10]     → [2,2,0,0,0]
    //   M: [10,10,10,10,10]       → [0,0,0,0,0]
    //   N: [10,10,10,90,10]       → [0,0,0,1,0]
    //   O: [10,90,90,90,90]       → [0,1,1,1,1]
    //   P: [170,170,90,10,10]     → [2,2,1,0,0]
    //   Q: [170,90,10,10,10]      → [2,1,0,0,0]
    //   R: [10,170,90,10,10]      → [0,2,1,0,0]
    //   S: [90,10,10,10,90]       → [1,0,0,0,1]
    //   T: [90,90,10,10,10]       → [1,1,0,0,0]
    //   U: [10,170,170,10,90]     → [0,2,2,0,1]
    //   V: [10,170,170,90,10]     → [0,2,2,1,0]
    //   W: [10,170,170,170,10]    → [0,2,2,2,0]
    //   X: [10,90,10,10,10]       → [0,1,0,0,0]
    //   Y: [170,10,10,10,170]     → [2,0,0,0,2]
    //   Z: [10,170,10,10,10]      → [0,2,0,0,0]

    const SIGNS_MAP = {
        A: [1,0,0,0,0], B: [0,2,2,2,2], C: [1,1,1,1,1], D: [0,2,1,1,1],
        E: [0,1,1,1,0], F: [1,0,2,2,2], G: [1,2,0,0,0], H: [0,2,2,0,0],
        I: [0,0,0,0,2], J: [0,0,0,0,1], K: [2,2,2,0,0], L: [2,2,0,0,0],
        M: [0,0,0,0,0], N: [0,0,0,1,0], O: [0,1,1,1,1], P: [2,2,1,0,0],
        Q: [2,1,0,0,0], R: [0,2,1,0,0], S: [1,0,0,0,1], T: [1,1,0,0,0],
        U: [0,2,2,0,1], V: [0,2,2,1,0], W: [0,2,2,2,0], X: [0,1,0,0,0],
        Y: [2,0,0,0,2], Z: [0,2,0,0,0],
    };

    const FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"];

    function updateFingerVisualiser(letter, fingerLevels) {
        const levels = fingerLevels || SIGNS_MAP[letter ? letter.toUpperCase() : ''] || [0,0,0,0,0];
        FINGER_NAMES.forEach((name, i) => {
            const el = fingerEls[name];
            const lvl = levels[i];
            el.className = 'finger ' + (lvl >= 2 ? 'extended' : lvl === 1 ? 'half' : 'closed');
        });
    }

    function resetFingerVisualiser() {
        FINGER_NAMES.forEach((name) => {
            fingerEls[name].className = "finger";
        });
    }

    // ─── Phrase Buttons ──────────────────────────────────────
    const phraseButtons = document.querySelectorAll('.btn-phrase');
    let _activeBtn = null;

    phraseButtons.forEach((btn) => {
        btn.addEventListener('click', () => {
            if (isSigning) return;
            const phrase = btn.dataset.phrase;
            if (!phrase) return;

            addChatMessage(`🗣️ "${btn.textContent}"`, "user");
            // [Feature Change] Phrase motions removed from robot to avoid mechanical stress.
            // Delegate the phrase to fingerspelling.
            socket.emit('send_text', { text: phrase });
            _activeBtn = btn;
            btn.classList.add('active');
        });
    });

    function _disablePhraseButtons() {
        phraseButtons.forEach((b) => { b.disabled = true; });
        if (_activeBtn) _activeBtn.classList.add('active');
    }

    function _enablePhraseButtons() {
        phraseButtons.forEach((b) => {
            b.disabled = false;
            b.classList.remove('active');
        });
        _activeBtn = null;
    }

    // ─── Voice Input Module (Task 1) ─────────────────────────
    // [Task 1.1] No duplicate signing_status handler here — the single
    // handler above covers mic enable/disable.
    // [Task 1.2] Final results auto-submit; interim results show in placeholder.
    // [Task 1.3] stopListening() called after auto-submit.

    let _voiceIsListening = false;

    function _voiceStopListening() {
        // Exposed so the signing_status handler can call it
        if (_voiceRecognition) {
            _voiceRecognition.abort();
        }
        _voiceIsListening = false;
        btnMic.classList.remove('listening');
        chatInput.placeholder = "Type or speak a word to sign…";
    }

    let _voiceRecognition = null;

    (() => {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

        if (!SpeechRecognition) {
            btnMic.disabled = true;
            btnMic.title = "Voice input not supported in this browser";
            return;
        }

        const recognition = new SpeechRecognition();
        recognition.lang = 'en-US';
        recognition.interimResults = true;
        recognition.continuous = false;
        _voiceRecognition = recognition;

        function startListening() {
            if (isSigning) return;
            try {
                recognition.start();
            } catch (_) { /* already started */ }
        }

        btnMic.addEventListener('click', () => {
            if (_voiceIsListening) {
                _voiceStopListening();
            } else {
                startListening();
            }
        });

        recognition.addEventListener('start', () => {
            _voiceIsListening = true;
            btnMic.classList.add('listening');
            chatInput.placeholder = "Listening…";
        });

        recognition.addEventListener('end', () => {
            _voiceIsListening = false;
            btnMic.classList.remove('listening');
            chatInput.placeholder = "Type or speak a word to sign…";
        });

        recognition.addEventListener('result', (e) => {
            let finalTranscript = '';
            let interimTranscript = '';

            for (let i = 0; i < e.results.length; i++) {
                const result = e.results[i];
                if (result.isFinal) {
                    finalTranscript += result[0].transcript;
                } else {
                    interimTranscript += result[0].transcript;
                }
            }

            if (finalTranscript) {
                // [Task 1.2] Final result: populate input and auto-submit
                chatInput.value = finalTranscript;
                // [Task 1.3] Stop listening before submitting
                _voiceStopListening();
                // Auto-submit if non-empty
                if (chatInput.value.trim()) {
                    submitChatText();
                }
            } else if (interimTranscript) {
                // [Task 1.2] Interim: show in placeholder for visual feedback
                chatInput.placeholder = `🎤 ${interimTranscript}`;
                // Don't put it in chatInput.value to avoid accidental submission
            }
        });

        recognition.addEventListener('error', (e) => {
            if (e.error === 'not-allowed' || e.error === 'permission-denied') {
                alert("Microphone access was denied. Please allow microphone permissions and try again.");
            }
            _voiceIsListening = false;
            btnMic.classList.remove('listening');
            chatInput.placeholder = "Type or speak a word to sign…";
        });
    })();
})();
