import pyrealsense2 as rs
import cv2
import numpy as np
import time
import datetime
import os

def record_realsense_videos(output_rgb_path="rgb_video.mp4", output_depth_path="depth_video.mp4", duration_seconds=10):
    """
    Records RGB and depth videos from Intel RealSense D435i camera.
    
    Args:
        output_rgb_path: Path to save RGB video
        output_depth_path: Path to save depth video
        duration_seconds: Recording duration in seconds
    """
    # Configure the pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    
    record_rgb = len(output_rgb_path) > 0
    record_depth = len(output_depth_path) > 0
    print(f"Recording RGB: {record_rgb}, Recording Depth: {record_depth}")
    
    # Enable RGB and depth streams
    if record_rgb:
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        print("RGB stream enabled.")
    if record_depth:
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        print("Depth stream enabled.")
    
    # Start streaming
    pipeline.start(config)
    
    # Initialize video writers
    fps = 30
    frame_size = (1280, 720)
    
    rgb_writer = None
    depth_writer = None
    if record_rgb:
        rgb_writer = cv2.VideoWriter(output_rgb_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)
    if record_depth:
        depth_writer = cv2.VideoWriter(output_depth_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)
    
    frame_count = 0
    target_frames = duration_seconds * fps
    
    try:
        rgb_frame = None
        depth_frame = None
        while frame_count < target_frames:
            frames = pipeline.wait_for_frames()
            if record_rgb:
                rgb_frame = frames.get_color_frame()
            if record_depth:
                depth_frame = frames.get_depth_frame()
            
            if not rgb_frame and not depth_frame:
                print("No frames received, skipping...")
                continue
            
            # Convert to numpy arrays
            if record_rgb:
                rgb_data = np.asarray(rgb_frame.get_data())
            if record_depth:
                depth_data = np.asarray(depth_frame.get_data())
                # Normalize and convert depth to 8-bit for video
                depth_normalized = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            # Write frames
            if rgb_writer:
                rgb_writer.write(rgb_data)
            if depth_writer:
                depth_writer.write(cv2.cvtColor(depth_normalized, cv2.COLOR_GRAY2BGR))
            
            frame_count += 1
            #print(f"Recorded frame {frame_count}/{target_frames}")
    
    finally:
        print("Stopping recording...")
        pipeline.stop()
        if rgb_writer:
            rgb_writer.release()
        if depth_writer:
            depth_writer.release()
        print(f"Videos saved: {output_rgb_path}, {output_depth_path}")

def start_recording(duration_seconds=10):
    date_and_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(date_and_time, exist_ok=True)
    record_realsense_videos(output_rgb_path=f"{date_and_time}/rgb_video.mp4", output_depth_path=f"", duration_seconds=duration_seconds)

if __name__ == "__main__":
    input_duration = input("Enter recording duration in seconds (default 10): ")
    try:
        duration = int(input_duration)
    except ValueError:
        duration = 10
        print("Invalid input. Using default duration of 10 seconds.")
    start_recording(duration_seconds=duration)