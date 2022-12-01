"""

This launch file is intended as an example of how to spin an
ungrounded robot (humanoid).  To use, run:

   ros2 launch ungounded pirouette.launch.py

This should start
  1) RVIZ, ready to view the robot
  2) The robot_state_publisher to broadcast the robot model
  2) The GUI to determine the joints.
  3) The pirouette example code

"""

import os
import xacro

from ament_index_python.packages import get_package_share_directory as pkgdir

from launch                            import LaunchDescription
from launch.actions                    import Shutdown
from launch_ros.actions                import Node


#
# Generate the Launch Description
#
def generate_launch_description():

    ######################################################################
    # LOCATE FILES

    # Locate the RVIZ configuration file.
    rvizcfg = os.path.join(pkgdir('atlaspushup'), 'rviz/viewurdf.rviz')
    # Locate/load the robot's URDF file (XML).
    urdf = os.path.join(pkgdir('atlas_description'), 'urdf/atlas_v5.urdf')
    with open(urdf, 'r') as file:
        robot_description = file.read()



    ######################################################################
    # PREPARE THE LAUNCH ELEMENTS

    # Configure a node for the robot_state_publisher.
    node_robot_state_publisher = Node(
        name       = 'robot_state_publisher', 
        package    = 'robot_state_publisher',
        executable = 'robot_state_publisher',
        output     = 'screen',
        parameters = [{'robot_description': robot_description}])

    # Configure a node for RVIZ
    node_rviz = Node(
        name       = 'rviz', 
        package    = 'rviz2',
        executable = 'rviz2',
        output     = 'screen',
        arguments  = ['-d', rvizcfg],
        on_exit    = Shutdown())

    # Configure a node for the GUI
    # node_gui = Node(
    #     name       = 'gui', 
    #     package    = 'joint_state_publisher_gui',
    #     executable = 'joint_state_publisher_gui',
    #     output     = 'screen',
    #     on_exit    = Shutdown())

    # Configure a node for the pirouette demo.
    node_pushup = Node(
        name       = 'AsymmetricPushup',
        package    = 'atlaspushup',
        executable = 'AsymmetricPushup',
        output     = 'screen',
        on_exit    = Shutdown())

    node_demo = Node(
        name       = 'asymmetricpushupstand',
        package    = 'atlaspushup',
        executable = 'asymmetricpushupstand',
        output     = 'screen',
        on_exit    = Shutdown())



    ######################################################################
    # COMBINE THE ELEMENTS INTO ONE LIST
    
    # Return the description, built as a python list.
    return LaunchDescription([

        # Start the robot_state_publisher, RVIZ, the GUI, and the demo.
        node_robot_state_publisher,
        node_rviz,
        node_demo,
        node_pushup,
    ])
