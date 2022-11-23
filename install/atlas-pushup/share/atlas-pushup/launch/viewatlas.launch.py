import os

from ament_index_python.packages import get_package_share_directory
import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.conditions import IfCondition, UnlessCondition

def generate_launch_description():
    #urdf_file_name = LaunchConfiguration('urdf_arg')
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    #urdf_arg = DeclareLaunchArgument('urdf_arg', default_value='urdf/atlas.urdf')

    # Parameters
    pkg_name = 'atlas_description'
    urdf_file_name = 'urdf/atlas_v5.urdf'
    rviz_file_name = 'rviz/viewurdf.rviz'
    
    urdf = os.path.join(
        get_package_share_directory(pkg_name),
        urdf_file_name)
    with open(urdf, 'r') as infp:
        robot_desc = infp.read()
        
    rvizconfig = os.path.join(get_package_share_directory(pkg_name),
        rviz_file_name)							

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time, 'robot_description': robot_desc}],
        arguments=[urdf]
    )
    
    joint_state_publisher_gui_node = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui'
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rvizconfig],
        on_exit=launch.actions.Shutdown(),
    )

    return LaunchDescription([
        robot_state_publisher_node,
        rviz_node,
        joint_state_publisher_gui_node
    ])
