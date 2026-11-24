from datetime import datetime
from pathlib import Path

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String

from experiment.exp_config import EXPERIMENT_SIGNALS_TOPIC, START_VIDEO_RECORDING_SIGNAL, STOP_VIDEO_RECORDING_SIGNAL

class VideoRecorder(Node):
	def __init__(self):
		super().__init__("video_recorder")
		self.declare_parameter("image_topic", "/camera/image_raw")
		self.declare_parameter("camera_name", "dontforgettosetcameraname")
		self.declare_parameter("output_directory", "~/USTA1.1/data/recordings/video")
		self.declare_parameter("fps", 30.0)

		self.bridge = CvBridge()
		self.output_directory = Path(
			self.get_parameter("output_directory").value
		).expanduser()
		self.fps = float(self.get_parameter("fps").value)
		if self.fps <= 0:
			raise ValueError("fps must be greater than zero")

		self.recording = False
		self.writer = None
		self.output_path = None
		self.create_subscription(
			Image, self.get_parameter("image_topic").value, self._on_image, 10
		)
		self.create_subscription(
			String, EXPERIMENT_SIGNALS_TOPIC, self.signal_callback, 10
		)

		self.get_logger().info("Waiting for a start signal")

	def start_recording(self):
		if self.recording:
			return
		self.output_directory.mkdir(parents=True, exist_ok=True)
		name = datetime.now().strftime("recording_%Y%m%d_%H%M%S_%f.mp4")
		name = f"{self.get_parameter('camera_name').value}_{name}"
		self.output_path = self.output_directory / name
		self.recording = True
		self.get_logger().info(f"Recording started: {self.output_path}")

	def signal_callback(self, msg):
		if msg.data == START_VIDEO_RECORDING_SIGNAL:
			self.start_recording()
		elif msg.data == STOP_VIDEO_RECORDING_SIGNAL:
			self.stop_recording()

	def _on_image(self, message):
		if not self.recording:
			return
		try:
			frame = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
			if self.writer is None:
				height, width = frame.shape[:2]
				fourcc = cv2.VideoWriter_fourcc(*"mp4v")
				self.writer = cv2.VideoWriter(
					str(self.output_path), fourcc, self.fps, (width, height)
				)
				if not self.writer.isOpened():
					self.writer.release()
					self.writer = None
					self.recording = False
					self.get_logger().error(f"Unable to open {self.output_path}")
					return
			self.writer.write(frame)
		except Exception as exc:
			self.get_logger().error(f"Failed to process image: {exc}")

	def stop_recording(self):
		if not self.recording:
			return
		if self.writer is not None:
			self.writer.release()
			self.writer = None
		self.recording = False
		self.get_logger().info("Recording stopped")

	def destroy_node(self):
		self.stop_recording()
		super().destroy_node()


def main(args=None):
	rclpy.init(args=args)
	node = VideoRecorder()
	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	finally:
		node.destroy_node()
		rclpy.shutdown()


if __name__ == "__main__":
	main()
