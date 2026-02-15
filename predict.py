import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = 128

model = load_model("model/asl_model.h5")
labels = sorted(os.listdir("data"))

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    roi = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    roi = preprocess_input(roi)
    roi = np.expand_dims(roi, axis=0)

    prediction = model.predict(roi, verbose=0)
    confidence = np.max(prediction)
    letter = labels[np.argmax(prediction)]

    if confidence > 0.75:
        cv2.putText(frame, f"{letter} ({confidence:.2f})",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 255, 0),
                    4)

    cv2.imshow("ASL Translator", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
