import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("deepfake_model.h5")

# Label mapping
labels = ["Real", "Fake"]

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Resize frame
    img = cv2.resize(frame, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    pred = model.predict(img, verbose=0)
    confidence = np.max(pred)
    label = labels[np.argmax(pred)]

    # Display result
    text = f"{label} ({confidence:.2f})"

    color = (0, 255, 0) if label == "Real" else (0, 0, 255)

    cv2.putText(
        frame,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.imshow("Deepfake Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
