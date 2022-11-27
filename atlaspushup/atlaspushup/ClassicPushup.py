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
#from Segments import Hold, Stay, GotoCubic, SplineCubic


#class ArmTrajectory():
#    def __init__(self, node, arm):
#        if arm == 'left':
#            self.joints = ['back_bkz', 'back_bky', 'back_bkx', 'l_arm_shz', 'l_arm_shx', 'l_arm_ely', 'l_arm_elx', 'l_arm_wry', 'l_arm_wrx', 'l_arm_wry2']
#            self.wxd = pxyz(1.32155,-0.2256,0.115332)
#        elif arm == 'right':
#            self.joints = ['back_bkz', 'back_bky', 'back_bkx', 'r_arm_shz', 'r_arm_shx', 'r_arm_ely', 'r_arm_elx', 'r_arm_wry', 'r_arm_wrx', 'r_arm_wry2']
#        KinematicChain(node, 'pelvis', 'l_hand', self.joints)

#
#   Spline Helper
#
#   We could also the Segments module.  But this seemed quicker and easier?
#
def spline(t, T, p0, pf):
    p = p0 + (pf-p0) * (3*t**2/T**2 - 2*t**3/T**3)
    v =      (pf-p0) * (6*t   /T**2 - 6*t**2/T**3)
    return (p, v)

#def JacwInv(J, gamma):
#    Jt = np.transpose(J)
#    return np.linalg.inv(Jt @ J + (gamma**2)*np.eye(3)) @ Jt

#def JacstarInv(J, gamma):
#    (U, s, Vt) = np.linalg.svd(J)
#    diag = np.diag(s)
#    diaginv = np.eye(3)
#    for i in range(0,3):
#        for j in range(0,3):
#            si = diag[i][j]
#            if abs(si) >= gamma:
#                diaginv[i][j] = 1/si
#            else:
#                diaginv[i][j] = si/(gamma**2)
#                
#    return np.transpose(Vt)@diaginv@np.transpose(U)

#
#   Trajectory Class
#
# IDEA: IMPLEMENT INVERSE KINEMATICS OF HANDS FIRST, THEN ADD FEET
class Trajectory():
    # Initialization.
    def __init__(self, node):

        # initialize atlas dimensions
        self.legLength = 0.941
        self.footLength = 0.17

        # initialize pushup data
        self.pushupDuration = 8

        # initialize kinematic chain - neck joint currently not in any chain
        self.larmjoints = ['back_bkz', 'back_bky', 'back_bkx', 'l_arm_shz', 'l_arm_shx', 'l_arm_ely', 'l_arm_elx', 'l_arm_wry', 'l_arm_wrx', 'l_arm_wry2']
        self.rarmjoints = ['back_bkz', 'back_bky', 'back_bkx', 'r_arm_shz', 'r_arm_shx', 'r_arm_ely', 'r_arm_elx', 'r_arm_wry', 'r_arm_wrx', 'r_arm_wry2']
        self.nonContributingJoints = ['back_bkz', 'back_bky', 'back_bkx']
        # lleg = ['l_leg_hpz', 'l_leg_hpx', 'l_leg_hpy', 'l_leg_kny', 'l_leg_aky', 'l_leg_akx']
        # rleg = ['r_leg_hpz', 'r_leg_hpx', 'r_leg_hpy', 'r_leg_kny', 'r_leg_aky', 'r_leg_akx']

        self.rarmchain = KinematicChain(node, 'pelvis', 'r_hand', self.rarmjoints)
        self.larmchain = KinematicChain(node, 'pelvis', 'l_hand', self.larmjoints)
        # self.rlegchain = KinematicChain(node, 'pelvis', 'r_foot', rleg)
        # self.llegchain = KinematicChain(node, 'pelvis', 'l_foot', lleg)

        # initialize pelvis data
        self.pelvisStartAngle = np.radians(60)#np.radians(57.2)
        self.pelvisEndAngle = np.radians(80)
        Rpelvis, ppelvis = self.getPelvisData(0)
        self.Tpelvis = T_from_Rp(Rpelvis, ppelvis)


        # initialize qs and xs
        # initial x (6x1 - 3x1 for left hand, 3x1 for right hand) with respect to world frame
        width = 1
        self.rHandx = pxyz(1.32155,-0.2256*width,0.115332)
        self.lHandx = pxyz(1.32155,0.2256*width,0.115332)
        self.wxd = np.vstack((self.lHandx)) #, rHandx))
        # initial joints 30x1 for starting pushup position relative to ???
        self.q0 = np.array([0,0,0,0,0,-0.5,-np.pi/2,0,0,0,0,0,0,0,0,0,0,0,0,0.5,np.pi/2,0,0,0,0,0,0,0,0,0]).reshape((-1,1))

        # change to use q0 once q0 is known to be correct
        self.q = self.q0
        self.err = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]).reshape((-1,1))

        # set joints
        self.larmchain.setjoints(self.getSpecificJoints(self.q, self.larmjoints).reshape((-1,1)))
        # self.rarmchain.setjoints(self.getSpecificJoints(self.q, self.rarmjoints).reshape((-1,1)))

        self.lmbda = 40
        # self.llegchain.setjoints(np.array([0.0 for i in range(len(lleg))]).reshape((-1,1)))
        # self.rlegchain.setjoints(np.array([0.0 for i in range(len(rleg))]).reshape((-1,1)))


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


    def getInverseKinematicsData(self, chain, joints, wxd, leftHand, dt):
        q   = self.q

        # Get jacobian for chain
        Jv = self.stackJacobians([(chain.Jv(), joints)], self.nonContributingJoints)
        Jw = self.stackJacobians([(chain.Jw(), joints)], self.nonContributingJoints)
        J = np.vstack((Jv, Jw))

        # compute desired relative to pelvis
        xd = np.linalg.inv(R_from_T(self.Tpelvis)) @ (wxd - p_from_T(self.Tpelvis))
        fixedRotation = Rotx(-np.pi/2)@Roty(np.pi/2) if leftHand else Rotx(-np.pi/2)@Roty(np.pi/2)@Rotz(np.pi)
        # Rotx(np.pi/2) -> Rotx(-np.pi/2)@Roty(np.pi/2)
        Rd = np.linalg.inv(R_from_T(self.Tpelvis)) @ fixedRotation

        # get current relative to pelvis
        x = p_from_T(chain.Ttip()) 
        R = R_from_T(chain.Ttip())

        # compute xddot and return values
        xddot = np.vstack((ep(xd, x), eR(Rd, R))) * 1/dt
        return (xddot, J)


    # Evaluate at the given time. This was last called (dt) ago.
    def evaluateJoints(self, t, dt):
        # Grab last qoint value Rotx(np.pi/2)
        q = self.q

        # Grab the desired xdot and jacobians
        (lxddot, leftArmJacobian) = self.getInverseKinematicsData(self.larmchain, self.larmjoints, self.lHandx, True, dt)
        (rxddot, rightArmJacobian) = self.getInverseKinematicsData(self.rarmchain, self.rarmjoints, self.rHandx, False, dt) #Rotx(-np.pi/2) @ Rotz(np.pi)
        J = np.vstack((leftArmJacobian, rightArmJacobian))

        # compute qdot
        xddot = np.vstack((lxddot, rxddot))
        qdot = np.linalg.pinv(J) @ xddot

        # integrate for q
        q = q + dt * qdot
        self.q = q

        # update chain joint values
        self.larmchain.setjoints(self.getSpecificJoints(self.q, self.larmjoints).reshape((-1,1)))
        self.rarmchain.setjoints(self.getSpecificJoints(self.q, self.rarmjoints).reshape((-1,1)))

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

        # compute pelvis pxyz
        ppelvis = pxyz(np.sin(a)*self.legLength, 0.0, np.cos(a)*self.legLength + self.footLength)
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
