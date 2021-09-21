#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
# NumPy
import numpy as np

# ROS Python API
import rospy

# Import the messages we're interested in sending and receiving, and having and sharing
from geometry_msgs.msg import PoseStamped, TwistStamped, Twist, Pose2D, Vector3
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import UInt8
from heron_msgs.msg import Drive

# Libraries for data manipulation
# from cluster.lib.tools import saturate
# from cluster.adapters import generic as adapter
from tf.transformations import quaternion_from_euler, euler_from_quaternion

INIT_AVERAGING_SAMPLES = 10
MSG_QUEUE_MAXLEN = 10

#Boat Constants
BOAT_WIDTH      = 0.7366 #m, ~15inches
MAX_FWD_THRUST  = 45.0  #Newtons
MAX_BCK_THRUST  = 25.0  #Newtons
MAX_FWD_VEL     = 2     #m/s
MAX_BCK_VEL     = 0.5   #m/s
MAX_OUTPUT      = 1
MAX_YAW_RATE    = 0.5   #rad/s

class RobotAdapterNode(object):
    """
    """
    def __init__(self):
        self.rate = 10.0
        asv_odom = rospy.get_param('~asv_odom', 'odometry/filtered')

        asmc_vel = rospy.get_param('~asmc_control_input', 'vectornav/ins_2d/local_vel')
        asmc_pose= rospy.get_param('~asmc_control_input', 'vectornav/ins_2d/NED_pose')

        asmc_ctrl_input = rospy.get_param('~asmc_control_input', 'usv_control/controller/control_input')

        # Tomo una primera posicion
        rospy.loginfo("[ADAPTER] Waiting for position initialization..")

        # Registro el publisher
        self.flag_pub = rospy.Publisher(
            "/arduino_br/ardumotors/flag",
            UInt8,
            queue_size=MSG_QUEUE_MAXLEN)
        rospy.loginfo('[ADAPTER] Will publish ardumotors flag = 1')

        # Registro el publisher
        self.ardu_pub = rospy.Publisher(
            'arduino',
            UInt8,
            queue_size=MSG_QUEUE_MAXLEN)
        rospy.loginfo('[ADAPTER] Will publish arduino flag = 1')

        # Registro el publisher
        self.cmd_pub = rospy.Publisher(
            "cmd_drive",
            Drive,
            queue_size=MSG_QUEUE_MAXLEN)
        rospy.loginfo('[ADAPTER] Will publish cmd Drive to topic: %s', "cmd_drive")

        # Registro el publisher
        self.asmc_pose_pub = rospy.Publisher(
            asmc_pose,
            Pose2D,
            queue_size=MSG_QUEUE_MAXLEN)
        rospy.loginfo('[ADAPTER] Will publish vehicle pose (NED) to topic: %s', asmc_pose)

        # Registro el publisher
        self.asmc_vel_pub = rospy.Publisher(
            asmc_vel,
            Vector3,
            queue_size=MSG_QUEUE_MAXLEN)
        rospy.loginfo('[ADAPTER] Will publish local vehicle velocity to topic: %s', asmc_vel)

        # Me subscribo al tópico de la odometría del ASV
        self.asv_odom_subs = rospy.Subscriber(
            asv_odom,
            Odometry,
            self.handle_odometry)
        rospy.loginfo('[ADAPTER] Subscribing to Heron Odometry topic: %s', asv_odom)

        # Me suscribo al tópico de la salida del controlador ASMC
        self.asmc_ctrl_input_subs = rospy.Subscriber(
            asmc_ctrl_input,
            Pose2D,
            self.handle_ctrl_output)
        rospy.loginfo('[ADAPTER] Subscribing to ASMC controller output topic: %s', asmc_ctrl_input)


        # Creo 1 timer para enviar los datos a la tasa pedida
        self.update_timer = rospy.Timer(
            rospy.Duration(1/self.rate),
            self.send_flag,
            oneshot=False)
        rospy.logdebug('[ASV ADAPTER] flag update rate: %f Hz', self.rate)

        rospy.loginfo("[ASV ADAPTER] Started Robot Adapter Node (check my topics)...")

    def handle_cmd(self, cmd_world_msg):
        """Transforms individual vehicle commands from World frame to Body"""
        rospy.logdebug('[ASV ADAPTER] New C World -> Body transform update call at %s', rospy.get_time())

        self.cmd_vel = saturate(adapter.process_cmd_msg(cmd_world_msg, True, self.prev_pose[3]), [-0.75, 0.75])

    def send_flag(self, event):
        """Sends the flags to asmc Alejandro's code"""
        msg=UInt8()
        msg.data = 1
        self.flag_pub.publish(msg)
        self.ardu_pub.publish(msg)
#         rospy.logdebug('[ASV ADAPTER] Command enqueued: %s', cmd_body_msg)

    def handle_ctrl_output(self, msg):
        fx = msg.x
        tauz = msg.theta

        max_tauz = MAX_BCK_THRUST*2*BOAT_WIDTH
        tauz = np.clip(tauz, -max_tauz, max_tauz)

        left_thrust = -tauz/(2*BOAT_WIDTH)
        right_thrust = tauz/(2*BOAT_WIDTH)

        #Provide maximum allowable thrust after yaw torque is guaranteed
        max_fx = 0
        if (tauz >= 0): 
            if (fx >= 0):   #forward thrust on the left thruster will be limiting factor
                max_fx = (MAX_FWD_THRUST - left_thrust) * 2
                fx = np.minimum(max_fx,fx)
            else :          #backward thrust on the right thruster will be limiting factor
                max_fx = (-MAX_BCK_THRUST - right_thrust) * 2
                fx = np.maximum(max_fx,fx)
        else :
            if (fx >= 0 ) :
                max_fx = (MAX_FWD_THRUST - right_thrust) * 2
                fx = np.minimum(max_fx,fx)
            else :
                max_fx = (-MAX_BCK_THRUST - left_thrust) * 2
                fx = np.maximum(max_fx,fx)

        left_thrust += fx/2.0
        right_thrust += fx/2.0

        left_thrust = self.saturate_thrusters(left_thrust)
        right_thrust = self.saturate_thrusters(right_thrust)

        cmd_output = Drive()
        cmd_output.left = self.calculate_motor_setting (left_thrust)
        cmd_output.right = self.calculate_motor_setting (right_thrust)
        self.cmd_pub.publish(cmd_output)

    def handle_odometry(self, msg):
        """Computes the vehicle (x, y, z, yaw) from odometry data"""

        # # Obtengo los parámetros del mensaje
        position = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
        orientation = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]

        roll, pitch, yaw = euler_from_quaternion(orientation)
        # orientation = quaternion_from_euler(0, 0, xyz_yaw[3])

        # Creo el mensaje con los datos necesarios
        message = Pose2D()
        # message.header.stamp = msg.header.stamp
        message.x = position[0]
        message.y = position[1]
        message.theta = yaw

        # Publico el mensaje
        self.asmc_pose_pub.publish(message)
        # rospy.loginfo('[ASV ADAPTER] Vehicle position enqueued: %s', message)

        # Creo el mensaje con los datos necesarios
        message = Vector3()
        # message.header.stamp = msg.header.stamp
        message.x = msg.twist.twist.linear.x
        message.y = msg.twist.twist.linear.y
        message.z = msg.twist.twist.angular.z

        # Publico el mensaje
        self.asmc_vel_pub.publish(message)
        # rospy.logdebug('[ASV ADAPTER] Vehicle position (relative) enqueued: %s', message)

    def saturate_thrusters (self, thrust):
        thrust = np.minimum(MAX_FWD_THRUST,thrust)
        thrust = np.maximum(-1*MAX_BCK_THRUST,thrust)
        return thrust

    def calculate_motor_setting (self, thrust):
        output = 0.0
        if (thrust > 0):
            output = thrust * (MAX_OUTPUT/MAX_FWD_THRUST)
        else:
            if (thrust < 0):
                output = thrust * (MAX_OUTPUT/MAX_BCK_THRUST)
        return output

    def shutdown(self):
        """Unregisters publishers and subscribers and shutdowns timers"""
        self.ardu_pub.unregister()
        self.flag_pub.unregister()
        self.asmc_pose_pub.unregister()
        self.asmc_vel_pub.unregister()
        self.asv_odom_subs.unregister()
        self.asmc_ctrl_input_subs.unregister()
        rospy.loginfo("[ADAPTER] Sayonara Adapter. Nos vemo' en Disney.")


def main():
    """Entrypoint del nodo"""
    rospy.init_node('heron_adapter', anonymous=True, log_level=rospy.INFO)
    node = RobotAdapterNode()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("[ADAPTER] Received Keyboard Interrupt (^C). Shutting down.")

    node.shutdown()

if __name__ == '__main__':
    main()
