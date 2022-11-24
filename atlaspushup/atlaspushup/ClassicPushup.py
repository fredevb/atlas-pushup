'''ClassicPushup

   This is the code for the final project

   Implements a classic pushup

   Node:        /generator
   Publish:     /joint_states           sensor_msgs/JointState

'''

from readline import add_history
import rclpy
import numpy as np
import math
from atlaspushup.GeneratorNode     import GeneratorNode, DemoNode
from atlaspushup.KinematicChain    import KinematicChain
from atlaspushup.TransformHelpers  import *


#
#   Spline Helper
#
#   We could also the Segments module.  But this seemed quicker and easier?
#
def spline(t, T, p0, pf):
    p = p0 + (pf-p0) * (3*t**2/T**2 - 2*t**3/T**3)
    v =      (pf-p0) * (6*t   /T**2 - 6*t**2/T**3)
    return (p, v)


#
#   Trajectory Class
#
class Trajectory():
    # Initialization.
    def __init__(self, node):

        # initialize one kinematic chain list two feet two hands
        # initialize starting position of qoints and pelvis
        # initialize final position of pelvis and joints?
        self.Tpelvis = None

        # initialize atlas dimensions
        self.shoulderHeight = 10
        self.upperArmLength = 10
        self.lowerArmLength = 10

        raise NotImplementedError


    # Declare the joint names.
    def jointnames(self):
        return [
            'back_bkx', 'back_bkx', 'back_bky',
            'back_bkz', 'l_arm_elx', 'l_arm_ely',
            'l_arm_shx', 'l_arm_shz', 'l_arm_wrx',
            'l_arm_wry', 'l_arm_wry2', 'l_leg_akx',
            'l_leg_aky', 'l_leg_hpx', 'l_leg_hpy',
            'l_leg_hpz', 'l_leg_kny', 'neck_ry',
            'r_arm_elx', 'r_arm_ely', 'r_arm_shx',
            'r_arm_shz', 'r_arm_wrx', 'r_arm_wry',
            'r_arm_wry2', 'r_leg_arollkx', 'r_leg_aky',
            'r_leg_hpx', 'r_leg_hpy', 'r_leg_hpz',
            'r_leg_kny'
            ]

    # Evaluate at the given time.  This was last called (dt) ago.
    def evaluateJoints(self, t, dt):
        # have pelvis trajectory as seperate method to evaluate so one method evaluateJoints, one evaluatePelvis
        # compute desired x from pelvis and do intervse kinematics

        shoulderq, shoulderqdot, elbowq, elbowqdot = self.armJoints(self.pelvisRotationAngle(t)) # compute h

        raise NotImplementedError

        # Grab the last joint value and task error.
        q   = self.q
        err = self.err

        # Compute the inverse kinematics
        J    = np.vstack((self.chain.Jv(),self.chain.Jw()))
        xdot = np.vstack((vd, wd))
        qdot = np.linalg.inv(J) @ (xdot + self.lam * err)

        # Integrate the joint position and update the kin chain data.
        q = q + dt * qdot
        self.chain.setjoints(q)

        # Compute the resulting task error (to be used next cycle).
        err  = np.vstack((ep(pd, self.chain.ptip()), eR(Rd, self.chain.Rtip())))

        # Save the joint value and task error for the next cycle.
        self.q   = q
        self.err = err

        # Return the position and velocity as python lists.
        return (q.flatten().tolist(), qdot.flatten().tolist())

    def evaluatePelvis(self, t, dt):
        # update to find pxyz and rotation at given time
        a, adot = spline(t%3, 3, np.pi/3, np.pi/2)
        ppelvis = pxyz(np.sin(a), 0, 0.95 - np.sin(a) + 0.1) # 0.95 should be height of pelvis, 0.1 should be height of foot
        Rpelvis = Roty(a)
        Tpelvis = T_from_Rp(Rpelvis, ppelvis)
        self.Tpelvis = Tpelvis
        return Tpelvis

    def pelvisRotationAngle(self, t, dt):
        return np.sin(t)

    def armJoints(self, theta, h = 0):
        d = math.sin(theta)*self.shoulderHeight - h
        shoulderq = math.pi/2 - theta
        shoulderqdot = -1
        elbowq = math.acos((self.upperArmLength**2 + self.lowerArmLength**2 - d**2)/(2*self.upperArmLength*self.lowerArmLength))
        elbowqdot = None
        return (shoulderq, shoulderqdot, elbowq, elbowqdot)

#
#  Main Code
#
def main(args=None):
    # Initialize ROS and the generator node (100Hz) for the Trajectory.
    rclpy.init(args=args)
    generator = GeneratorNode('generator', 100, Trajectory)
    # use DemoNode for pelvis

    # Spin, until interrupted or the trajectory ends.
    generator.spin()

    # Shutdown the node and ROS.
    generator.shutdown()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
