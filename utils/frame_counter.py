import cv2
import os


def count_video_frames(video_path):
    """
    Count the total number of frames in a video.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    return total_frames


if __name__ == "__main__":
    video_path = input("Enter video path: ")

    try:
        frames = count_video_frames(video_path)
        print(f"Total Frames: {frames}")
    except Exception as e:
        print(f"Error: {e}")
