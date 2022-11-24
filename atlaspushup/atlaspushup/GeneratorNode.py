'''GeneratorNode.py

   This creates a trajectory generator node

   generator = GeneratorNote(name, rate, TrajectoryClass)

      Initialize the node, under the specified name and rate.  This
      also requires a trajectory class which must implement:

         trajectory = TrajectoryClass(node)
         jointnames = trajectory.jointnames()
         (q, qdot)  = trajectory.evaluate(t, dt)

      where jointnames, q, qdot are all python lists of the joint's
      name, position, and velocity respectively.  The dt is the time
      since the last evaluation, to be used for integration.

   HW#6: Updated to use a constant time step and avoid rate jitter.

   Node:        /generator
   Publish:     /joint_states           sensor_msgs/JointState

'''

import rclpy
import numpy as np

from asyncio            import Future
from rclpy.node         import Node
from sensor_msgs.msg    import JointState
from tf2_ros            import TransformBroadcaster
from geometry_msgs.msg  import Vector3, Quaternion, Transform
from geometry_msgs.msg  import TransformStamped
from atlaspushup.TransformHelpers import *

#
#   Trajectory Generator Node Class
#
class GeneratorNode(Node):
    # Initialization.
    def __init__(self, name, rate, Trajectory):
        # Initialize the node, naming it as specified
        super().__init__(name)

        # Set up a trajectory.
        self.trajectory = Trajectory(self)
        self.jointnames = self.trajectory.jointnames()

        self.demoNode = DemoNode(name, rate, self.trajectory)

        # Add a publisher to send the joint commands.
        self.pub = self.create_publisher(JointState, '/joint_states', 10)

        # Wait for a connection to happen.  This isn't necessary, but
        # means we don't start until the rest of the system is ready.
        self.get_logger().info("Waiting for a /joint_states subscriber...")
        while(not self.count_subscribers('/joint_states')):
            pass

        # Create a future object to signal when the trajectory ends,
        # i.e. no longer returns useful data.
        self.future = Future()

        # Create a timer to keep calculating/sending commands.
        self.starttime = self.get_clock().now()
        self.servotime = self.starttime
        self.timer     = self.create_timer(1/float(rate), self.update)

        self.t  = 0.0
        self.dt = self.timer.timer_period_ns * 1e-9
        self.get_logger().info("Running with dt of %f seconds (%fHz)" %
                               (self.dt, rate))

    # Shutdown
    def shutdown(self):
        # Destroy the timer, then shut down the node.
        self.timer.destroy()
        self.destroy_node()

    # Spin
    def spin(self):
        # Keep running (taking care of the timer callbacks and message
        # passing), until interrupted or the trajectory is complete
        # (as signaled by the future object).
        rclpy.spin_until_future_complete(self, self.future)

        # Report the reason for shutting down.
        if self.future.done():
            self.get_logger().info("Stopping: " + self.future.result())
        else:
            self.get_logger().info("Stopping: Interrupted")


    # Update - send a new joint command every time step.
    def update(self):
        # transform is the pelvis transform
        # Grab the current time.
        now = self.get_clock().now()
        t   = (now - self.starttime).nanoseconds * 1e-9
        dt  = (now - self.servotime).nanoseconds * 1e-9
        self.servotime = now

        # To avoid the time jitter introduced by an inconsistent timer,
        # just enforce a constant time step and ignore the above.
        self.t += self.dt
        t  = self.t
        dt = self.dt

        self.demoNode.update()

        # Compute the desired joint positions and velocities for this time.
        desired = self.trajectory.evaluateJoints(t, dt)
        if desired is None:
            self.future.set_result("Trajectory has ended")
            return
        (q, qdot) = desired

        # Build up a command message and publish.
        cmdmsg = JointState()
        cmdmsg.header.stamp = now.to_msg()      # Current time
        cmdmsg.name         = self.jointnames   # List of joint names
        cmdmsg.position     = q                 # List of joint positions
        cmdmsg.velocity     = qdot              # List of joint velocities
        self.pub.publish(cmdmsg)


#
#   Demo Node Class
#
class DemoNode(Node):
    # Initialization.
    def __init__(self, name, rate, trajectory):
        # Initialize the node, naming it as specified
        super().__init__(name)
        self.trajectory = trajectory
        # In addition to any publishers, create a broadcaster to
        # define the pelvis transform w.r.t. the world.

        # Initialize the transform broadcaster
        self.broadcaster = TransformBroadcaster(self)

        # Create a timer to keep calculating/sending commands.
        self.starttime = self.get_clock().now()
        self.servotime = self.starttime
        self.timer     = self.create_timer(1/float(rate), self.update)

        self.t  = 0.0
        self.dt = self.timer.timer_period_ns * 1e-9
        self.get_logger().info("Running with dt of %f seconds (%fHz)" %
                               (self.dt, rate))

    # Shutdown
    def shutdown(self):
        # Destroy the timer, then shut down the node.
        self.timer.destroy()
        self.destroy_node()

    # Update - send a new joint command every time step.
    def update(self):
        # Grab the current time.  But to avoid the time jitter in
        # timing, enforce a constant time step.
        now     = self.get_clock().now()
        self.t += self.dt
        t       = self.t
        dt      = self.dt

        # Compute position/orientation of the pelvis (w.r.t. world).
        Tpelvis = self.trajectory.evaluatePelvis(t, dt)
        
        # Build up and send the Pelvis w.r.t. World Transform!
        trans = TransformStamped()
        trans.header.stamp    = self.get_clock().now().to_msg()
        trans.header.frame_id = 'world'
        trans.child_frame_id  = 'pelvis'
        trans.transform       = transform_from_T(Tpelvis)
        self.broadcaster.sendTransform(trans)
