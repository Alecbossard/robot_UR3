#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 11:24:20 2025

@author: taix
"""
import matplotlib.pyplot as plt
import pybullet as p
import numpy as np


#############################################
def afficheloi_mvt(s,ds,dds,temps,tc):
    fig, axes = plt.subplots(nrows=3,ncols=1)

    axes[0].plot(temps, s, "r-", label=" s(t) ")
    axes[0].axvline(x=tc[1])
    axes[0].axvline(x=tc[2])
    axes[0].grid()
    axes[0].set_title('Fonction s(t)')
    axes[1].plot(temps, ds, "b-", label=" ds(t)")
    axes[1].axvline(x=tc[1])
    axes[1].axvline(x=tc[2])
    axes[1].grid()
    axes[1].set_title('Fonction ds(t)')
    axes[2].plot(temps, dds, "g-", label=" dds(t)")
    axes[2].axvline(x=tc[1])
    axes[2].axvline(x=tc[2])
    axes[2].grid()
    axes[2].set_title('Fonction dds(t)')
    plt.tight_layout()
    plt.legend()
    plt.show()
######################################################
# affichage de 3 subplots
# INPUT: 
#       t   = tableau des instants de commutation de 0 à tf
#       tim =  liste des instants de calcul ti
#       fi  = valeurs de l afoncton à tracer pour chaque ti
#       refi= texte décrivant la fonction à afficher
# OUTPUT:
#       affiche les 3 courbes avec les instants de commutation
################################################################
def affichage3courbes(t, tim, f1, ref1, f2, ref2, f3, ref3):
    fig, axes = plt.subplots(nrows=3,ncols=1)

    axes[0].plot(tim, f1, "r-")
    axes[0].axvline(x=t[1])
    axes[0].axvline(x=t[2])
    axes[0].grid()
    axes[0].set_title('Fonction' + ref1)
    axes[1].plot(tim, f2, "b-")
    axes[1].axvline(x=t[1])
    axes[1].axvline(x=t[2])
    axes[1].grid()
    axes[1].set_title('Fonction' + ref2)
    axes[2].plot(tim, f3, "g-")
    axes[2].axvline(x=t[1])
    axes[2].axvline(x=t[2])
    axes[2].grid()
    axes[2].set_title('Fonction' + ref3)
    plt.tight_layout()
    
    plt.show()
    return
##########################################################################
def plot3figures(time,f,g,h) :
    fig = plt.figure()
    
    plt.subplot(3,1,1)
    plt.plot(time, f)
    plt.ylabel(r'$x$')
    plt.subplot(3,1,2)
    plt.plot(time, g)
    plt.ylabel(r'$y$')
    plt.subplot(3,1,3)
    plt.plot(time, h)
    plt.ylabel(r'$z$')
    plt.xlabel('Time [s]')
    return  
########################################################################################
def draw_coordinate_frame(pos, orn, length=1.0):
    """
    Dessine un repère 3D (X, Y, Z) à une position et orientation données.
    
    :param pos: La position [x, y, z] du centre du repère.
    :param orn: L'orientation [qx, qy, qqz, qw] (quaternion) du repère.
    :param length: La longueur de chaque axe.
    :return: Une liste des IDs des lignes de débogage.
    """
    
    # Convertir le quaternion en matrice de rotation pour obtenir les vecteurs des axes
    rot_matrix = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
    
    # Vecteurs de base du repère dans son propre référentiel
    base_x = np.array([1, 0, 0])
    base_y = np.array([0, 1, 0])
    base_z = np.array([0, 0, 1])
    
    # Calculer les vecteurs des axes dans le référentiel monde
    axis_x = pos + rot_matrix @ (base_x * length)
    axis_y = pos + rot_matrix @ (base_y * length)
    axis_z = pos + rot_matrix @ (base_z * length)
    
    # Convertir la position en tableau NumPy si ce n'est pas déjà le cas pour les opérations
    pos_np = np.array(pos)
    
    # Tracer les axes avec des couleurs différentes (R, G, B)
    line_id_x = p.addUserDebugLine(pos_np, axis_x, lineColorRGB=[1, 0, 0], lineWidth=2) # Rouge pour X
    line_id_y = p.addUserDebugLine(pos_np, axis_y, lineColorRGB=[0, 1, 0], lineWidth=2) # Vert pour Y
    line_id_z = p.addUserDebugLine(pos_np, axis_z, lineColorRGB=[0, 0, 1], lineWidth=2) # Bleu pour Z
    
    return [line_id_x, line_id_y, line_id_z]

# Exemple d'utilisation (à insérer dans votre boucle de simulation PyBullet)
# p.connect(p.GUI)
# p.setAdditionalSearchPath(pybullet_data.getDataPath())
# p.setGravity(0, 0, -10)
# planeId = p.loadURDF("plane.urdf")
# 
# # Position et orientation (par exemple, au centre du monde, sans rotation)
# my_pos = [0, 0, 0.5]
# my_orn = p.getQuaternionFromEuler([0, 0, 0])
# 
# frame_ids = draw_coordinate_frame(my_pos, my_orn, length=0.8)
# 
# # Pour supprimer le repère plus tard :
# # for line_id in frame_ids:
# # p.removeUserDebugItem(line_id)
# 
# # ... boucle de simulation p.stepSimulation()

###############################################################################
def update_joint_frame(robotId, jointIndex, line_ids, length=0.2):
    """
    Supprime le repère précédent et dessine un nouveau repère sur le joint spécifié.
    
    :param robotId: ID du robot.
    :param jointIndex: Index du joint/link cible.
    :param line_ids: Liste des IDs des lignes du repère précédent (pour la suppression).
    :param length: Longueur des axes du repère.
    :return: Nouvelle liste des IDs des lignes du repère.
    """
    # 1. Suppression des lignes précédentes
    for line_id in line_ids:
        p.removeUserDebugItem(line_id)
        
    # 2. Récupération de la pose actuelle du link
    try:
        link_state = p.getLinkState(robotId, jointIndex)
        pos = link_state[0]
        orn = link_state[1]
    except p.error:
        print(f"Erreur : Joint/Link {jointIndex} introuvable.")
        return []

    # 3. Dessin du nouveau repère
    
    # Calcul des vecteurs des axes (similaire à la fonction draw_coordinate_frame)
    rot_matrix = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
    pos_np = np.array(pos)
    
    axis_x = pos_np + rot_matrix @ (np.array([1, 0, 0]) * length)
    axis_y = pos_np + rot_matrix @ (np.array([0, 1, 0]) * length)
    axis_z = pos_np + rot_matrix @ (np.array([0, 0, 1]) * length)
    
    # Tracer les axes
    line_id_x = p.addUserDebugLine(pos_np, axis_x, lineColorRGB=[1, 0, 0], lineWidth=3, lifeTime=0)
    line_id_y = p.addUserDebugLine(pos_np, axis_y, lineColorRGB=[0, 1, 0], lineWidth=3, lifeTime=0)
    line_id_z = p.addUserDebugLine(pos_np, axis_z, lineColorRGB=[0, 0, 1], lineWidth=3, lifeTime=0)
    
    return [line_id_x, line_id_y, line_id_z]
