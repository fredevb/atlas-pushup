"""fencedemo.py

   Create a visual structure (objects) in RVIZ using a visualization
   marker array.

   For the types of markers see: http://wiki.ros.org/rviz/DisplayTypes/Marker

   Node:        /fencedemo
   Publish:     /visualization_marker_array     visualization_msgs/MarkerArray

"""

import rclpy
import numpy as np

from rclpy.node             import Node
from rclpy.qos              import QoSProfile, DurabilityPolicy
from geometry_msgs.msg      import Point
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray


def box(x, y, z, lx, ly, lz):
    # Create the cube marker.
    marker = Marker()
    marker.type               = Marker.CUBE

    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0
    marker.pose.position.x    = x
    marker.pose.position.y    = y
    marker.pose.position.z    = z

    marker.scale.x            = lx
    marker.scale.y            = ly
    marker.scale.z            = lz

    marker.color.r            = 1.0
    marker.color.g            = 0.0
    marker.color.b            = 0.0
    marker.color.a            = 1.0     # Transparency

    return marker


def fence():
    # Start with an empty marker list.
    markers = []

    x = 1.32155
    y = 0.2256
    z = 0.2
    # Append the bars.
    markers.append(box(x, y, z/2 , 0.3, 0.3, z))

    # Return the list
    return markers


#
#   Demo Node Class
#
class DemoNode(Node):
    # Initialization.
    def __init__(self, name):
        # Initialize the node, naming it as specified
        super().__init__(name)

        # Prepare the publisher (latching for new subscribers).
        quality = QoSProfile(durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             depth=1)
        self.pub = self.create_publisher(
            MarkerArray, '/visualization_marker_array', quality)

        # Wait for a connection to happen, so only have to send once.
        self.get_logger().info("Waiting for RVIZ...")
        while(not self.count_subscribers('/visualization_marker_array')):
            pass

        # Create the markers visualize.
        markers = fence()

        # Add the timestamp, frame, namespace, action, and id to each marker.
        timestamp =self.get_clock().now().to_msg()
        for (i,marker) in enumerate(markers):
            marker.header.stamp       = timestamp
            marker.header.frame_id    = 'world'
            marker.ns                 = 'fence'
            marker.action             = Marker.ADD
            marker.id                 = i

        # Create the marker array message and publish.
        arraymsg = MarkerArray()
        arraymsg.markers = markers
        self.pub.publish(arraymsg)


        
#
#  Main Code
#
def main(args=None):
    # Initialize ROS and the demo node.
    rclpy.init(args=args)
    node = DemoNode('fencedemo')

    # Run until interrupted.
    rclpy.spin(node)

    # Shutdown the node and ROS.
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
