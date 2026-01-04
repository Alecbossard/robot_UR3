#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 14:48:07 2025

@author: taix
"""
import pybullet as p
import time
from src.pybullet import affiche

F_CONTROL = 10000
sleep_time = 1/F_CONTROL 

def update_simulation(steps, sleep_time=0.0001):
     """
     Update the simulation by stepping and waiting for a specified time.
     """
     for _ in range(steps):
         p.stepSimulation()
         time.sleep(sleep_time)
         
# function to joint position, velocity and torque feedback
def getJointStates(robot_id,control_joints):
    joint_states = p.getJointStates(robot_id, control_joints)
    joint_positions = [state[0] for state in joint_states]
    joint_velocities = [state[1] for state in joint_states]
    joint_torques = [state[3] for state in joint_states]
    return joint_positions, joint_velocities, joint_torques


    # function for setting joint positions of robot
def setJointPosition(robot_id, control_joints, position, kp=1.0, kv=1.0):
    print('Joint position controller')
    zero_vec = [0.0] * len(control_joints)
    p.setJointMotorControlArray(robot_id,
                                control_joints,
                                p.POSITION_CONTROL,
                                targetPositions=position,
                                targetVelocities=zero_vec,
                                positionGains=[kp] * len(control_joints),
                                velocityGains=[kv] * len(control_joints))
    frame_ee = []
    for _ in range(100):
        p.stepSimulation()
        time.sleep(sleep_time)
        
    frame_ee = affiche.update_joint_frame(robot_id, 7, frame_ee)

#################################################################
    #function to do joint velcoity control
def JointVelocityControl(robot_id, control_joints, joint_velocities, sim_time=0.05, max_force=200):
    print('Joint velocity controller')
    t=0
    while t<sim_time:
        p.setJointMotorControlArray(robot_id,
                                    control_joints,
                                    p.VELOCITY_CONTROL,
                                    targetVelocities=joint_velocities,
                                    forces = [max_force] * (len(control_joints)))
        p.stepSimulation()
        time.sleep(sleep_time)
        t += sleep_time

####################################################################        
# Fonction pour un controle en couple
def JointTorqueControl(robot_id, couple,control_joints):
        """
        sends torque commands (needs an array of desired joint torques)
        """
        zeroGains = [0.0] * len(control_joints)

        p.setJointMotorControlArray(robot_id, control_joints, 
                                    p.TORQUE_CONTROL, forces=couple, 
                                    positionGains=zeroGains, velocityGains=zeroGains)

