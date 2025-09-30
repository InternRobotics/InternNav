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
        self.turn_angle = math.radians(90)  # 角度调这里
        self.angular_speed = -0.2  # 方向和速度调这里
        self.rate = rospy.Rate(10)  # 10Hz

    def odom_callback(self, msg):
        # 从四元数获取偏航角
        orientation = msg.pose.pose.orientation
        quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(quaternion)

        self.current_yaw = yaw

        # 初始化起始偏航角
        if self.start_yaw is None and not self.turning:
            self.start_yaw = yaw
            rospy.loginfo(f"起始偏航角: {math.degrees(self.start_yaw):.2f}度")

    def execute_turn(self):
        if self.start_yaw is None:
            rospy.loginfo("等待初始位姿数据...")
            return False

        if not self.turning:
            self.turning = True
            rospy.loginfo("开始原地转弯90度")

        # 计算已转过的角度
        current_angle = self.current_yaw - self.start_yaw

        # 处理角度超过π或小于-π的情况（角度归一化）
        if current_angle > math.pi:
            current_angle -= 2 * math.pi
        elif current_angle < -math.pi:
            current_angle += 2 * math.pi

        # 计算还需转过的角度
        remaining_angle = self.turn_angle - abs(current_angle)

        # 创建 Twist 消息
        twist = Twist()

        # 如果还没达到目标角度，继续旋转
        if remaining_angle > 0.05:  # 留一点余量（约2.86度）
            twist.angular.z = self.angular_speed * min(1.0, remaining_angle * 6)
            print(f"twist.angular.z {twist.angular.z} remaining_angle {remaining_angle}")
            self.cmd_vel_pub.publish(twist)
            return False
        else:
            twist.angular.z = 0.0
            self.cmd_vel_pub.publish(twist)
            rospy.loginfo(f"完成转弯，最终偏航角: {math.degrees(self.current_yaw):.2f}度")
            return True

    def run(self):
        while not rospy.is_shutdown():
            if self.execute_turn():
                rospy.loginfo("任务完成，退出节点")
                break
            self.rate.sleep()


class DiscreteRobotController(Turn90Degrees):
    """
    Extends Turn90Degree to allow discrete step-based control.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # initialize parent class

    def stand_still(self, duration=0.5):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)
        rospy.sleep(duration)  # Maintain stand still for a short duration
        rospy.loginfo("Stand still command executed.")

    def move_forward(self, distance=0.25):
        twist = Twist()
        twist.linear.x = 0.2  # Forward speed
        twist.angular.z = 0.0
        duration = distance / twist.linear.x  # Time to move forward the specified distance

        rospy.loginfo(f"Moving forward for {duration:.2f} seconds.")
        end_time = rospy.Time.now() + rospy.Duration(duration)

        while rospy.Time.now() < end_time and not rospy.is_shutdown():
            self.cmd_vel_pub.publish(twist)
            self.rate.sleep()

        self.stand_still()  # Stop after moving forward
        rospy.loginfo("Move forward command executed.")

    def turn_left(self, angle=30, speed=0.2):
        self.turn_angle = math.radians(angle)  # 角度调这里
        self.angular_speed = speed  # Set positive angular speed for left turn
        self.start_yaw = None  # Reset start yaw to current position
        self.turning = False  # Reset turning flag
        self.run()
        self.stand_still()  # Stop after moving forward
        rospy.loginfo("Turn left command executed.")

    def turn_right(self, angle=30, speed=-0.2):
        self.turn_angle = math.radians(angle)  # 角度调这里
        self.angular_speed = speed  # Set positive angular speed for left turn
        self.start_yaw = None  # Reset start yaw to current position
        self.turning = False  # Reset turning flag
        self.run()
        self.stand_still()  # Stop after moving forward
        rospy.loginfo("Turn right command executed.")


if __name__ == '__main__':
    try:
        control = DiscreteRobotController()
        control.turn_left(10)
        control.move_forward(0.1)
        control.turn_right(10)

    except rospy.ROSInterruptException:
        pass
