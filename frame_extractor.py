import cv2
import os

def extract_frames(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening file: {video_path}")
        return

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    frame_id = 1
    while True:
        ret, frame = cap.read()
        if ret:
            frame_filename = os.path.join(output_path, f"{frame_id}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_id += 1
        else:
            break

    cap.release()
    print(f"Frames extracted to {output_path}")
