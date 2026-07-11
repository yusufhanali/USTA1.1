import numpy as np
import cv2 as cv

def visualize_npy_video(npy_path, wait=33, window_name="RGB Video"):
    """
    Play a NumPy video file (saved as (N, H, W, 3) array or object array of frames).

    Args:
        npy_path (str): Path to the .npy file containing RGB frames
        wait (int): Delay between frames in milliseconds (approx. 33 ms for 30 FPS).
        window_name (str): Name of the OpenCV window.
    """
    frames = np.load(npy_path, allow_pickle=True)

    # Case 1: Standard video (N, H, W, 3)
    if isinstance(frames, np.ndarray) and frames.ndim == 4 and frames.shape[-1] == 3:
        print(f"Loaded full video with {len(frames)} frames.")
    # Case 2: Ragged face list (dtype=object)
    elif isinstance(frames, np.ndarray) and frames.dtype == object:
        print(f"Loaded {len(frames)} frames.")
    else:
        print("Invalid frame format:", frames.shape, frames.dtype)
        return

    for i, frame in enumerate(frames):
        if frame is None or not isinstance(frame, np.ndarray):
            continue
        cv.imshow(window_name, frame)
        key = cv.waitKey(wait)
        if key == ord('q'):
            break

    cv.destroyAllWindows()
    

def load_video_frames_from_mp4(video_path):
    """Load frames from an MP4 file into a list of RGB NumPy arrays."""
    cap = cv.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frames.append(rgb_frame)
    cap.release()
    return frames