import cv2
import numpy as np

class FrameQualityChecker:
    def __init__(self, blur_threshold=100):
        self.blur_threshold = blur_threshold

    def is_frame_valid(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Laplacian variance for blur detection
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

        if blur_score < self.blur_threshold:
            return False, blur_score

        return True, blur_score


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    checker = FrameQualityChecker()

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        valid, score = checker.is_frame_valid(frame)

        text = f"Sharpness: {score:.2f}"

        if valid:
            text += " | Valid Frame"
        else:
            text += " | Blurry Frame"

        cv2.putText(
            frame,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Frame Quality Check", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
