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

# does not prevent feet from sliding...
def line(t, T, p0, pf):
    p = p0 + (t * (pf - p0) / T)
    v = (pf - p0) / T
    return (p, v)

#
#   Trajectory Class
#
class Trajectory():
    # Initialization.
    def __init__(self, node):

        # initialize atlas dimensions
        self.legLength = 0.95
        self.footLength = 0.17

        # initialize pushup data
        self.pushupDuration = 8

        # initialize kinematic chain - neck joint currently not in any chain
        self.larmjoints = ['back_bkz', 'back_bky', 'back_bkx', 'l_arm_shz', 'l_arm_shx', 'l_arm_ely', 'l_arm_elx', 'l_arm_wry', 'l_arm_wrx', 'l_arm_wry2']
        self.rarmjoints = ['back_bkz', 'back_bky', 'back_bkx', 'r_arm_shz', 'r_arm_shx', 'r_arm_ely', 'r_arm_elx', 'r_arm_wry', 'r_arm_wrx', 'r_arm_wry2']
        self.llegjoints = ['l_leg_hpz', 'l_leg_hpx', 'l_leg_hpy', 'l_leg_kny', 'l_leg_aky', 'l_leg_akx']
        self.rlegjoints = ['r_leg_hpz', 'r_leg_hpx', 'r_leg_hpy', 'r_leg_kny', 'r_leg_aky', 'r_leg_akx']

        # fixed joints at 0
        self.armNonContributingJoints = ['back_bkz', 'back_bky', 'back_bkx']
        self.legNonContributingJoints = []
        # ['l_leg_hpz', 'l_leg_hpx', 'l_leg_kny', 'r_leg_hpz', 'r_leg_hpx', 'r_leg_kny']

        self.rarmchain = KinematicChain(node, 'pelvis', 'r_hand', self.rarmjoints)
        self.larmchain = KinematicChain(node, 'pelvis', 'l_hand', self.larmjoints)
        self.rlegchain = KinematicChain(node, 'pelvis', 'r_foot', self.rlegjoints)
        self.llegchain = KinematicChain(node, 'pelvis', 'l_foot', self.llegjoints)

        # initialize pelvis data
        self.pelvisStartAngle = np.radians(58.8)
        self.pelvisEndAngle = np.radians(75)
        Rpelvis, ppelvis = self.getPelvisData(0)
        self.Tpelvis = T_from_Rp(Rpelvis, ppelvis)


        # initialize qs and xs
        # initial x (6x1 - 3x1 for left hand, 3x1 for right hand) with respect to world frame
        handWidth = 1.4
        self.rHandx = pxyz(1.32155,-0.2256*handWidth,0.115332)
        self.lHandx = pxyz(1.32155,0.2256*handWidth,0.115332)

        legWidth = 1
        self.rFootx = pxyz(0.07, -0.1*legWidth, self.footLength)
        self.lFootx = pxyz(0.07, 0.1*legWidth, self.footLength)

        # initial joints 30x1 for starting pushup
        
        self.q0 = np.array([0.0, 0.0, 0.0, -1.0665207119388567, -1.582323088245135, 
                -0.5705856587004811, -0.9068233383949136, 0.515869865694755, -0.15915481819248703, 
                0.48879079149892446, 0.015488554035449355, 0.2645835607263747, -0.013248255893721639, 
                -0.22017916590252892, 0.00802388863784277, 0.5001861909675166, 0.0, 1.066520711938855, 
                -1.5823230882452168, 0.5705856587005895, 0.9068233383949171, -0.5158698656947643, 
                -0.15915481819249414, 3.6303834450888752, -0.015488554035449355, 0.2645835607263747, 
                0.013248255893721639, -0.22017916590252892, -0.00802388863784277, 0.5001861909675166] 
            ).reshape((-1,1))
        
        # change to use q0 once q0 is known to be correct
        self.q = self.q0

        # set joints
        self.larmchain.setjoints(self.getSpecificJoints(self.q, self.larmjoints).reshape((-1,1)))
        self.rarmchain.setjoints(self.getSpecificJoints(self.q, self.rarmjoints).reshape((-1,1)))
        self.llegchain.setjoints(self.getSpecificJoints(self.q, self.llegjoints).reshape((-1,1)))
        self.rlegchain.setjoints(self.getSpecificJoints(self.q, self.rlegjoints).reshape((-1,1)))

    # Declare the joint names.
    def jointnames(self):
        return [
            'back_bkx', 'back_bky', 'back_bkz',
            'l_arm_elx', 'l_arm_ely', 'l_arm_shx',
            'l_arm_shz', 'l_arm_wrx', 'l_arm_wry',
            'l_arm_wry2', 'l_leg_akx', 'l_leg_aky',
            'l_leg_hpx', 'l_leg_hpy', 'l_leg_hpz',
            'l_leg_kny', 'neck_ry', 'r_arm_elx',
            'r_arm_ely', 'r_arm_shx', 'r_arm_shz',
            'r_arm_wrx', 'r_arm_wry', 'r_arm_wry2',
            'r_leg_akx', 'r_leg_aky', 'r_leg_hpx',
            'r_leg_hpy', 'r_leg_hpz', 'r_leg_kny',
            ]

    # extracts joint values of the joints in desiredJointLabels from a 30x1 list of all joints
    def getSpecificJoints(self, allJoints, desiredJointLabels):
        joints = []
        for jointLabel in desiredJointLabels:
            idx = self.jointnames().index(jointLabel)
            joints.append(allJoints[idx])
        return np.array(joints)

    # input should be list of (Jacobian, related_joint_labels) - returns single Jacobian of size nx30 where n is the number of tasks across all jaobians
    def stackJacobians(self, jacobians, zeroContributionJoints = []):
        totalJacobians = []
        for jacobian, jointLabels in jacobians:
            J = np.zeros((len(jacobian), len(self.jointnames())))
            jacobian = np.array(jacobian)
            for i in range(len(jointLabels)):
                jointLabel = jointLabels[i]
                col = jacobian[:,i]
                idx = self.jointnames().index(jointLabel)
                J[:, idx] = col
            totalJacobians.append(J)
        J = np.vstack(totalJacobians)

        for jointLabel in zeroContributionJoints:
            idx = self.jointnames().index(jointLabel)
            J[:,idx] = np.zeros(len(J))
        return J

    # method to take multiple lists of joint values of form (joint_values, related_joint_labels) and return single 30x1 list of joint values
    def combineIntoAllJoints(self, jointLists, defaultValue = [0]):
        joints  = []
        for jointLabel in self.jointnames():
            found = False
            for values, jointLabels in jointLists:
                if jointLabel in jointLabels:
                    idx = jointLabels.index(jointLabel)
                    value = values[idx]
                    joints.append(value)
                    found = True
                if found:
                    break
            if not found:
                joints.append(defaultValue)
        return np.array(joints)


    def getInverseKinematicsData(self, chain, joints, nonContributingJoints, wxd, isLeftSide, isArms, dt):
        q  = self.q

        # Get jacobian for chain
        Jv = self.stackJacobians([(chain.Jv(), joints)], nonContributingJoints)
        Jw = self.stackJacobians([(chain.Jw(), joints)], nonContributingJoints)
        J = np.vstack((Jv, Jw))

        # compute desired relative to pelvis
        xd = np.linalg.inv(R_from_T(self.Tpelvis)) @ (wxd - p_from_T(self.Tpelvis))
        fixedArmRotation = Rotx(-np.pi/2) if isLeftSide else Rotx(-np.pi/2) @ Rotz(np.pi)
        fixedLegRotation = Roty(np.pi/2)
        totalFixedRotation = fixedArmRotation if isArms else fixedLegRotation

        Rd = np.linalg.inv(R_from_T(self.Tpelvis)) @ totalFixedRotation

        # get current relative to pelvis
        x = p_from_T(chain.Ttip())
        R = R_from_T(chain.Ttip())

        # compute xddot and return values
        xddot = np.vstack((ep(xd, x), eR(Rd, R))) * 1/dt
        return (xddot, J)

    def limitJointVelocities(self, qdot, limit = 2):
        for i in range(len(qdot)):
            if np.abs(qdot[i][0]) > limit:
                qdot[i][0] = limit if qdot[i][0] > limit else -limit
        return qdot

    def inverse(self, J, weight = 0.005):
        return np.linalg.inv(J.T @ J + weight**2 * np.eye(len(J.T))) @ J.T

    def  getSecondaryTaskGoals(self):
        return [
            ('l_arm_shx', -np.pi/3), ('r_arm_shx', np.pi/3), # flaring out
            ('l_arm_ely', -np.pi/2), ('r_arm_ely', -np.pi/2),
        ]

    # Evaluate at the given time. This was last called (dt) ago.
    def evaluateJoints(self, t, dt):
        # Grab last qoint value 
        q = self.q
        
        (LAxddot, LAJac) = self.getInverseKinematicsData(self.larmchain, self.larmjoints, self.armNonContributingJoints, self.lHandx, True, True, dt)
        (RAxddot, RAJac) = self.getInverseKinematicsData(self.rarmchain, self.rarmjoints, self.armNonContributingJoints, self.rHandx, False, True, dt)
        (LLxddot, LLJac) = self.getInverseKinematicsData(self.llegchain, self.llegjoints, self.legNonContributingJoints, self.lFootx, True, False, dt)
        (RLxddot, RLJac) = self.getInverseKinematicsData(self.rlegchain, self.rlegjoints, self.legNonContributingJoints, self.rFootx, False, False, dt)

        #compute J
        J = np.vstack((LAJac, RAJac, LLJac, RLJac))

        #compute xddot
        xddot = np.vstack((LAxddot, RAxddot, LLxddot, RLxddot))

        #compute qdot
        qdotprimary = self.limitJointVelocities(self.inverse(J) @ xddot, 2)

        # establish secondary task
        qdotsecondary = np.zeros((30,1))
        secondaryTaskJoints = self.getSecondaryTaskGoals()
        for jointLabel, value in secondaryTaskJoints:
            idx = self.jointnames().index(jointLabel)
            qdotsecondary[idx][0] = (value - q[idx][0]) * 1/dt
        qdotsecondary = self.limitJointVelocities(qdotsecondary, 2)
        nullspace = np.eye(len(J.T)) - self.inverse(J) @ J
        qdot = qdotprimary + nullspace @ qdotsecondary
        qdot = self.limitJointVelocities(qdot, 2)

        # left arm shx must be negative, right arm shx must be positive
        # integrate for q
        q = q + dt * qdot
        self.q = q

        # update chain joint values
        self.larmchain.setjoints(self.getSpecificJoints(self.q, self.larmjoints).reshape((-1,1)))
        self.rarmchain.setjoints(self.getSpecificJoints(self.q, self.rarmjoints).reshape((-1,1)))

        self.llegchain.setjoints(self.getSpecificJoints(self.q, self.llegjoints).reshape((-1,1)))
        self.rlegchain.setjoints(self.getSpecificJoints(self.q, self.rlegjoints).reshape((-1,1)))

        return (q.flatten().tolist(), qdot.flatten().tolist())

    def evaluatePelvis(self, t, dt):
        # update to find pxyz and rotation at given time
        pushupTime = t % self.pushupDuration
        Rpelvis, ppelvis = self.getPelvisData(pushupTime)
        Tpelvis = T_from_Rp(Rpelvis, ppelvis)
        self.Tpelvis = Tpelvis
        return Tpelvis

    def getPelvisData(self, t):
        # compute pelvis theta
        halfTime = self.pushupDuration / 2
        if t < halfTime:
            a, adot = spline(t, halfTime, self.pelvisStartAngle, self.pelvisEndAngle)
        else:
            t1 = t - 4
            a, adot = spline(t1, halfTime, self.pelvisStartAngle, self.pelvisEndAngle)

            a = self.pelvisStartAngle + self.pelvisEndAngle - a

        Rpelvis = Roty(a)

        angle_from_horizontal = np.pi/2 - a
        # compute pelvis pxyz
        ppelvis = pxyz(np.cos(angle_from_horizontal)*(self.legLength) , 0.0, np.sin(angle_from_horizontal)* (self.legLength + self.footLength))

        return Rpelvis, ppelvis

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
