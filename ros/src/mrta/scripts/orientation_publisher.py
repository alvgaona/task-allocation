#!/usr/bin/python3

import random
import rospy
from mrta.msg import Orientation


TOPIC_NAME = 'orientation'
NODE_NAME = 'orientation_publisher'


def publish_orientation():
    pub = rospy.Publisher(TOPIC_NAME, Orientation, queue_size=10)
    rospy.init_node(NODE_NAME, anonymous=True)
    rate = rospy.Rate(1)  # In Hz

    while not rospy.is_shutdown():
        x = 0
        y = random.randint(0, 255)
        z = 0
        orientation = Orientation(x, y, z)

        pub.publish(orientation)
        rate.sleep()


if __name__ == '__main__':
    try:
        publish_orientation()
    except rospy.ROSInterruptException:
        pass
