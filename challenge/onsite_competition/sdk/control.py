#!/usr/bin/env python3
import math

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion


class Turn90Degrees:
    def __init__(self):
        rospy.init_node('turn_90_degrees_node', anonymous=True)

        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        self.odom_sub = rospy.Subscriber('/ranger_base_node/odom', Odometry, self.odom_callback)

        self.current_yaw = 0.0
        self.start_yaw = None
        self.turning = False
        self.turn_angle = math.radians(90)  # angle
        self.angular_speed = -0.2  # dirention and speed
        self.rate = rospy.Rate(10)  # 10Hz

    def odom_callback(self, msg):
        # Get the yaw angle from a quaternion.
        orientation = msg.pose.pose.orientation
        quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(quaternion)

        self.current_yaw = yaw

        # initialize the start yaw
        if self.start_yaw is None and not self.turning:
            self.start_yaw = yaw
            rospy.loginfo(f"start yaw: {math.degrees(self.start_yaw):.2f}")

        # --- position (new)
        p = msg.pose.pose.position
        self.current_xy = (p.x, p.y)

    def execute_turn(self):
        if self.start_yaw is None:
            rospy.loginfo("wait for init yaw state...")
            return False

        if not self.turning:
            self.turning = True
            rospy.loginfo("start to turn")

        # compute turned angle
        current_angle = self.current_yaw - self.start_yaw

        # normalize the angle
        if current_angle > math.pi:
            current_angle -= 2 * math.pi
        elif current_angle < -math.pi:
            current_angle += 2 * math.pi

        # compute remaining angle
        remaining_angle = self.turn_angle - abs(current_angle)

        # create Twist msg
        twist = Twist()

        # if not reach the goal angle, keep turning
        if remaining_angle > 0.05:  # allow some diff (about 2.86 degree）
            twist.angular.z = self.angular_speed * min(1.0, remaining_angle * 6)
            print(f"twist.angular.z {twist.angular.z} remaining_angle {remaining_angle}")
            self.cmd_vel_pub.publish(twist)
            return False
        else:
            twist.angular.z = 0.0
            self.cmd_vel_pub.publish(twist)
            rospy.loginfo(f"finish turn, final yaw: {math.degrees(self.current_yaw):.2f}")
            return True

    def run(self):
        while not rospy.is_shutdown():
            if self.execute_turn():
                rospy.loginfo("Task finished")
                break
            self.rate.sleep()


class DiscreteRobotController(Turn90Degrees):
    """
    Extends Turn90Degree to allow discrete step-based control.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # initialize parent class

    def stand_still(self, duration=0.2):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)
        rospy.sleep(duration)  # Maintain stand still for a short duration
        rospy.loginfo("Stand still command executed.")

    def move_forward(self, distance=0.5, speed=0.2):
        twist = Twist()
        twist.linear.x = speed  # Forward speed
        twist.angular.z = 0.0
        duration = distance / twist.linear.x  # Time to move forward the specified distance

        rospy.loginfo(f"Moving forward for {duration:.2f} seconds.")
        end_time = rospy.Time.now() + rospy.Duration(duration)

        while rospy.Time.now() < end_time and not rospy.is_shutdown():
            self.cmd_vel_pub.publish(twist)
            self.rate.sleep()

        # self.stand_still()  # Stop after moving forward
        rospy.loginfo("Move forward command executed.")

    # ---- NEW: feedback-controlled forward motion


def move_feedback(self, distance=0.25, speed=0.5, tol=0.02, timeout=None):
    """
    里程计闭环直线移动：支持正/负 distance。
    - distance: 目标路程（米）。>0 前进，<0 倒车。
    - speed: 名义线速度（m/s），可给正/负，最终会按正数取绝对值。
    - tol: 终止距离容差（米）
    - timeout: 超时（秒）；默认 max(3*|distance|/speed, 3.0)
    """
    # 等待位姿
    while self.current_xy is None and not rospy.is_shutdown():
        rospy.loginfo_throttle(2.0, "Waiting for /odom...")
        self.rate.sleep()
    if rospy.is_shutdown():
        return

    # 方向与目标
    direction = 1.0 if distance >= 0.0 else -1.0
    target = abs(distance)
    speed = abs(speed) if speed is not None else 0.5

    start_xy = self.current_xy
    start_time = rospy.Time.now()

    if timeout is None:
        timeout = max(3.0, 3.0 * (target / max(speed, 0.05)))

    twist = Twist()

    # 简单 P 控制，越靠近越慢
    Kp = 1.5
    min_speed = 0.06  # 防止轮子停转

    rospy.loginfo(
        f"move_linear: target={target:.3f} m, dir={'forward' if direction>0 else 'backward'}, "
        f"speed≈{speed:.2f} m/s, tol={tol:.3f} m"
    )

    try:
        while not rospy.is_shutdown():
            # 超时
            if (rospy.Time.now() - start_time).to_sec() > timeout:
                rospy.logwarn("move_linear timeout reached; stopping.")
                break

            # 走过的距离（欧式距离，不区分前后，目标用 abs）
            cx, cy = self.current_xy
            sx, sy = start_xy
            traveled = math.hypot(cx - sx, cy - sy)
            remaining = target - traveled

            # 达标退出
            if remaining <= tol:
                rospy.loginfo(f"Reached distance: traveled={traveled:.3f} m (tol {tol} m)")
                break

            # 速度控制（带方向）
            v = Kp * remaining
            v = max(min(v, speed), min_speed)  # [min_speed, speed]
            twist.linear.x = direction * v  # 关键：按方向加符号
            twist.angular.z = 0.0
            self.cmd_vel_pub.publish(twist)

            rospy.loginfo_throttle(
                1.0, f"traveled={traveled:.3f} m, remaining={remaining:.3f} m, v={twist.linear.x:.2f} m/s"
            )
            self.rate.sleep()

    finally:
        # 停车
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)

    rospy.loginfo("Move linear command executed.")

    def turn(self, angle=15, speed=0.5):
        self.turn_angle = math.radians(angle)  # update angle
        self.angular_speed = speed  # Set positive angular speed for left turn
        self.start_yaw = None  # Reset start yaw to current position
        self.turning = False  # Reset turning flag
        self.run()
        rospy.loginfo("Turn left command executed.")


if __name__ == '__main__':
    try:
        control = DiscreteRobotController()
        control.turn(15, 0.5)  # left
        control.turn(15, -0.5)  # right
        control.move_feedback(0.25, 0.5)
        control.move_feedback(-0.25, 0.5)

    except rospy.ROSInterruptException:
        pass
