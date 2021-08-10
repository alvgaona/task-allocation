#!/usr/bin/python3

import rospy

from tcp_endpoint.RosTCPServer import TCPServer
from tcp_endpoint.RosSubscriber import RosSubscriber

from mrta.msg import Orientation


def main():
    ros_node_name = rospy.get_param("/TCP_NODE_NAME", 'TCPServer')
    buffer_size = rospy.get_param("/TCP_BUFFER_SIZE", 1024)
    connections = rospy.get_param("/TCP_CONNECTIONS", 10)
    tcp_server = TCPServer(ros_node_name, buffer_size, connections)

    tcp_server.source_destination_dict = {
        'orientation': RosSubscriber('orientation', Orientation, tcp_server),
    }

    rospy.init_node(ros_node_name, anonymous=True)
    tcp_server.start()
    rospy.spin()


if __name__ == "__main__":
    main()
