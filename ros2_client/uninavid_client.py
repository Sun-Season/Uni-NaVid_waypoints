# coding: utf-8
"""
ROS2 client for Uni-NaVid navigation server.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import requests
import argparse
import time
import sys
import threading


class UniNaVidROS2Client(Node):
    def __init__(self, args):
        super().__init__('uninavid_ros2_client')
        self.args = args
        self.bridge = CvBridge()
        self.last_process_time = 0
        self.processing = False
        self.executing_action = False
        self.stop_execution = False
        self.session_id = None

        # 初始化 session
        if not self.create_session():
            self.get_logger().error("Failed to create session. Exiting.")
            sys.exit(1)

        self.get_logger().info(f"Connected to server at {args.server_url}")
        self.get_logger().info(f"Session ID: {self.session_id}")
        self.get_logger().info(f"Navigation goal: '{args.instruction}'")

        # 订阅图像
        self.subscription = self.create_subscription(
            Image,
            args.image_topic,
            self.image_callback,
            10)

        # 发布速度控制
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

    def create_session(self):
        """创建导航 session"""
        try:
            response = requests.post(
                f"{self.args.server_url}/api/v1/sessions",
                data={'instruction': self.args.instruction},
                timeout=10
            )
            response.raise_for_status()
            result = response.json()
            self.session_id = result['session_id']
            self.get_logger().info(f"Session created: {self.session_id}")
            return True
        except requests.exceptions.RequestException as e:
            self.get_logger().error(f"Error creating session: {e}")
            return False

    def delete_session(self):
        """删除 session"""
        if not self.session_id:
            return
        try:
            requests.delete(
                f"{self.args.server_url}/api/v1/sessions/{self.session_id}",
                timeout=5
            )
            self.get_logger().info(f"Session {self.session_id} deleted")
        except requests.exceptions.RequestException as e:
            self.get_logger().error(f"Error deleting session: {e}")

    def send_navigate_request(self, image):
        """发送导航请求"""
        # 图像压缩为 JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        result, encimg = cv2.imencode('.jpg', image, encode_param)

        if not result:
            self.get_logger().error("Image encoding failed.")
            return None

        files = {
            'image': ('image.jpg', encimg.tobytes(), 'image/jpeg')
        }
        data = {
            'session_id': self.session_id,
            'instruction': self.args.instruction
        }

        try:
            response = requests.post(
                f"{self.args.server_url}/api/v1/navigate",
                files=files,
                data=data,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.get_logger().error(f"Error sending navigate request: {e}")
            return None

    def stop_robot(self):
        """发布停止指令"""
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_publisher.publish(twist)

    def execute_action_sequence(self, actions):
        """在独立线程中执行动作序列"""
        try:
            LINEAR_SPEED = 0.2
            ANGULAR_SPEED = 0.5
            ACTION_DURATION = 1.0

            if not actions:
                return

            action = actions[0]
            self.get_logger().info(f"Executing action: {action}")

            twist = Twist()
            if action == "forward":
                twist.linear.x = LINEAR_SPEED
            elif action == "left":
                twist.angular.z = ANGULAR_SPEED
            elif action == "right":
                twist.angular.z = -ANGULAR_SPEED
            elif action in ("stop", "wait"):
                self.stop_robot()
                self.get_logger().info("Stop/wait command received.")
                time.sleep(ACTION_DURATION)
                return

            # 发布速度指令并持续一段时间
            start_time = time.time()
            while time.time() - start_time < ACTION_DURATION:
                if self.stop_execution:
                    break
                self.cmd_vel_publisher.publish(twist)
                time.sleep(0.1)

            # 动作执行完毕，停车
            self.stop_robot()

            # 等待稳定
            self.get_logger().info("Action done. Waiting for stabilization...")
            time.sleep(0.5)

        except Exception as e:
            self.get_logger().error(f"Error executing action: {e}")
            self.stop_robot()
        finally:
            self.get_logger().info("Action completed. Ready for next image.")
            self.executing_action = False

    def image_callback(self, msg):
        # 检查是否正在执行动作
        if self.executing_action:
            return

        # 频率控制
        current_time = time.time()
        if current_time - self.last_process_time < self.args.interval:
            return

        if self.processing:
            return

        self.processing = True
        try:
            # 转换 ROS Image 消息为 OpenCV 格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            self.get_logger().info("Sending image to server...")
            response = self.send_navigate_request(cv_image)

            if response:
                actions = response.get("actions", [])
                step = response.get("step", 0)
                did_inference = response.get("did_inference", False)

                self.get_logger().info(
                    f"Step {step}: actions={actions}, inference={did_inference}"
                )

                if actions:
                    self.executing_action = True
                    threading.Thread(
                        target=self.execute_action_sequence,
                        args=(actions,)
                    ).start()

            self.last_process_time = time.time()

        except Exception as e:
            self.get_logger().error(f"Error in image_callback: {e}")
            self.executing_action = False
        finally:
            self.processing = False


def main(args=None):
    rclpy.init(args=args)

    parser = argparse.ArgumentParser(description="UniNaVid ROS2 Client")
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://localhost:8000",
        help="Server URL"
    )
    parser.add_argument(
        "--instruction",
        type=str,
        required=True,
        help="Navigation instruction/goal"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Minimum interval between requests (seconds)"
    )
    parser.add_argument(
        "--image-topic",
        type=str,
        default="/camera/camera/color/image_raw",
        help="ROS2 image topic"
    )

    parsed_args, _ = parser.parse_known_args()

    client = UniNaVidROS2Client(parsed_args)

    try:
        rclpy.spin(client)
    except KeyboardInterrupt:
        pass
    finally:
        client.stop_execution = True
        client.stop_robot()
        client.delete_session()
        client.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
