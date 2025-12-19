import cv2
import numpy as np
import tensorflow as tf
import time
import mido

# =====================
# CONFIG
# =====================
MODEL_PATH = r"D:/Project/updatedmodel.keras"
IMG_SIZE = 160

CONF_THRESHOLD = 0.4
COOLDOWN_SEC = 2.0

CLASS_NAMES = ["closed", "nothing", "open hand"]

# MIDI CONFIG
MIDI_PORT_NAME = "AI_Control 1"
OPEN_NOTE = 60     # C4
CLOSED_NOTE = 62   # D4
VELOCITY = 100
MIDI_CHANNEL = 0

# =====================
# LOAD MODEL
# =====================
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded")

# =====================
# MIDI OUTPUT (loopMIDI)
# =====================
print("Available MIDI ports:")
print(mido.get_output_names())

if MIDI_PORT_NAME not in mido.get_output_names():
    print(f"❌ loopMIDI port '{MIDI_PORT_NAME}' not found")
    exit()

midi_out = mido.open_output(MIDI_PORT_NAME)
print(f"🎹 Connected to loopMIDI port: {MIDI_PORT_NAME}")

# =====================
# WEBCAM
# =====================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

last_midi_time = 0.0
print("📷 Press 'q' to quit")

# =====================
# MAIN LOOP
# =====================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # -----------------
    # PREPROCESS
    # -----------------
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    # -----------------
    # PREDICTION
    # -----------------
    preds = model.predict(img, verbose=0)[0]

    closed_prob = preds[CLASS_NAMES.index("closed")]
    open_prob = preds[CLASS_NAMES.index("open hand")]

    current_time = time.time()

    label = "NOTHING"
    confidence = max(preds)
    color = (255, 255, 0)

    # -----------------
    # MIDI LOGIC
    # -----------------
    if open_prob >= CONF_THRESHOLD:
        label = "OPEN"
        confidence = open_prob
        color = (0, 255, 0)

        if current_time - last_midi_time >= COOLDOWN_SEC:
            midi_out.send(mido.Message(
                "note_on",
                note=OPEN_NOTE,
                velocity=VELOCITY,
                channel=MIDI_CHANNEL
            ))
            midi_out.send(mido.Message(
                "note_off",
                note=OPEN_NOTE,
                velocity=0,
                channel=MIDI_CHANNEL
            ))
            last_midi_time = current_time

    elif closed_prob >= CONF_THRESHOLD:
        label = "CLOSED"
        confidence = closed_prob
        color = (0, 0, 255)

        if current_time - last_midi_time >= COOLDOWN_SEC:
            midi_out.send(mido.Message(
                "note_on",
                note=CLOSED_NOTE,
                velocity=VELOCITY,
                channel=MIDI_CHANNEL
            ))
            midi_out.send(mido.Message(
                "note_off",
                note=CLOSED_NOTE,
                velocity=0,
                channel=MIDI_CHANNEL
            ))
            last_midi_time = current_time

    # -----------------
    # DISPLAY
    # -----------------
    cv2.putText(
        frame,
        f"{label} ({confidence:.2f})",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color,
        2
    )

    cv2.imshow("Gesture → Cubase MIDI", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# =====================
# CLEANUP
# =====================
cap.release()
cv2.destroyAllWindows()
midi_out.close()
