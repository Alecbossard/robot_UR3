import pybullet as p
import pybullet_data
import time
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Imports de vos modules ---
from const_v import dh
from matrice_tn import generate_transformation_matrices, calcul_T06_global
from modele_differentiel import Jacob_geo
from part4_generation_articulaire import traj
from utils import mgd_vers_simulation, simulation_vers_mgd

# --- Configuration ---
# Chemin vers l'URDF (A adapter selon l'emplacement exact de votre dossier ur_description)
# D'après votre notebook, c'est: "./ur_description/urdf/ur3_robot.urdf"
URDF_PATH = "ur_description/urdf/ur3_robot.urdf" 

def init_simulation(dt):
    """Initialise PyBullet, charge le sol et le robot."""
    try:
        p.connect(p.GUI)
    except:
        p.connect(p.DIRECT)
        
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(dt)
    
    # Chargement du sol
    p.loadURDF("plane.urdf")
    
    # Chargement du robot
    if not os.path.exists(URDF_PATH):
        print(f"ATTENTION: Le fichier {URDF_PATH} est introuvable.")
        print("Veuillez vérifier le chemin URDF_PATH ligne 16.")
    
    startPos = [0, 0, 0]
    startOrn = p.getQuaternionFromEuler([0, 0, 0])
    robot_id = p.loadURDF(URDF_PATH, startPos, startOrn, useFixedBase=True)
    
    # Identification des joints contrôlables (les 6 axes)
    # D'après votre notebook, indices [1, 2, 3, 4, 5, 6]
    joint_indices = [1, 2, 3, 4, 5, 6]
    
    return robot_id, joint_indices

def get_feedback(robot_id, joint_indices):
    """
    Récupère q et dq du simulateur et les convertit en convention MGD.
    """
    states = p.getJointStates(robot_id, joint_indices)
    
    # 1. Lecture brute (Convention Simu)
    q_sim = [s[0] for s in states]
    qp_sim = [s[1] for s in states]
    
    # 2. Conversion (Convention MGD)
    q_mgd = simulation_vers_mgd(q_sim)
    qp_mgd = simulation_vers_mgd(qp_sim)
    
    return q_mgd, qp_mgd

def simulation_position(robot_id, joint_indices, time_vector, q_traj, O, R):
    """
    VI.1 : Exécute la trajectoire en contrôle de POSITION.
    Compare la position réelle (calculée via MGD sur q_mesuré) avec le cercle théorique.
    """
    print("\n=== VI.1 Simulation en POSITION ===")
    
    # Reset du robot à la position initiale
    q_init_sim = mgd_vers_simulation(q_traj[0])
    for i, joint in enumerate(joint_indices):
        p.resetJointState(robot_id, joint, q_init_sim[i])
        
    # Listes pour stocker les mesures
    X_mesure = []
    X_theorique = []
    
    input("Appuyez sur Entrée pour démarrer la simulation POSITION...")
    
    for i, t in enumerate(time_vector):
        # 1. Consigne : On prend le q théorique et on le convertit pour la simu
        target_q_mgd = q_traj[i]
        target_q_sim = mgd_vers_simulation(target_q_mgd)
        
        # 2. Envoi Commande Position
        p.setJointMotorControlArray(
            robot_id,
            joint_indices,
            p.POSITION_CONTROL,
            targetPositions=target_q_sim,
            # Gains par défaut ou ajustables (kp)
            positionGains=[1.0]*6,
            velocityGains=[1.0]*6
        )
        
        # 3. Pas de simulation
        p.stepSimulation()
        time.sleep(time_vector[1] - time_vector[0]) # Temps réel
        
        # 4. Mesure & Calcul MGD (Pour voir où on est vraiment)
        q_actuel_mgd, _ = get_feedback(robot_id, joint_indices)
        
        # MGD sur la position actuelle
        mats_actuels = generate_transformation_matrices(q_actuel_mgd, dh)
        T06_actuel = calcul_T06_global(mats_actuels)
        pos_actuelle = T06_actuel[:3, 3] # [x, y, z]
        X_mesure.append(pos_actuelle)
        
        # Calcul position théorique pour comparaison (MGD sur la consigne)
        mats_theo = generate_transformation_matrices(target_q_mgd, dh)
        T06_theo = calcul_T06_global(mats_theo)
        pos_theo = T06_theo[:3, 3]
        X_theorique.append(pos_theo)

    # Affichage des résultats
    X_mesure = np.array(X_mesure)
    X_theorique = np.array(X_theorique)
    
    plt.figure(figsize=(10, 5))
    
    # Trajectoire 2D (XZ)
    plt.subplot(1, 2, 1)
    plt.plot(X_theorique[:, 0], X_theorique[:, 2], 'r--', label='Consigne (Cercle)')
    plt.plot(X_mesure[:, 0], X_mesure[:, 2], 'b', label='Réel Robot')
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.title('Suivi de Trajectoire (Position)')
    plt.legend()
    plt.axis('equal')
    
    # Erreur au cours du temps
    erreurs = np.linalg.norm(X_mesure - X_theorique, axis=1)
    plt.subplot(1, 2, 2)
    plt.plot(time_vector, erreurs)
    plt.xlabel('Temps (s)')
    plt.ylabel('Erreur (m)')
    plt.title('Erreur de position')
    plt.grid()
    
    plt.tight_layout()
    plt.show()

def simulation_vitesse(robot_id, joint_indices, time_vector, qp_traj, V_cible):
    """
    VI.2 : Exécute la trajectoire en contrôle de VITESSE.
    Compare la norme de la vitesse cartésienne atteinte avec la consigne V.
    """
    print("\n=== VI.2 Simulation en VITESSE ===")
    
    # Reset du robot (Important car le mode vitesse fait dériver la position)
    # On reprend la position de départ de la trajectoire
    # Note: Il faut recalculer le q_traj[0] ou le passer en argument, ici on le recalcule vite fait
    # Ou mieux, on reset juste à une position "connue" proche du départ
    # Pour simplifier, on suppose que le robot est resté à la fin de la simu précédente, 
    # donc on le reset brutalement.
    q_start = [0.0, np.pi/2, -np.pi/4, 0.0, -np.pi/2, 0.0] # Estimation départ
    q_start_sim = mgd_vers_simulation(q_start)
    
    for i, joint in enumerate(joint_indices):
        p.resetJointState(robot_id, joint, q_start_sim[i])
    
    V_mesure_norme = []
    
    input("Appuyez sur Entrée pour démarrer la simulation VITESSE...")
    
    for i, t in enumerate(time_vector):
        # 1. Consigne : Vitesse articulaire
        target_qp_mgd = qp_traj[i]
        target_qp_sim = mgd_vers_simulation(target_qp_mgd) # Linéaire, donc marche aussi pour vitesses
        
        # 2. Envoi Commande Vitesse
        # En mode vitesse, il faut désactiver le gain de position (sinon il essaie de rester sur place)
        p.setJointMotorControlArray(
            robot_id,
            joint_indices,
            p.VELOCITY_CONTROL,
            targetVelocities=target_qp_sim,
            forces=[500]*6 # Force max suffisante
        )
        
        # 3. Simulation
        p.stepSimulation()
        time.sleep(time_vector[1] - time_vector[0])
        
        # 4. Mesure & Calcul Vitesse Cartésienne (Jacobienne)
        q_actuel_mgd, qp_actuel_mgd = get_feedback(robot_id, joint_indices)
        
        # Calcul Jacobienne à la position actuelle
        mats = generate_transformation_matrices(q_actuel_mgd, dh)
        J = Jacob_geo(mats)
        J_v = J[:3, :] # Partie linéaire 3x6
        
        # Vitesse Cartésienne V = J * q_point
        v_cartesienne = np.dot(J_v, qp_actuel_mgd)
        v_norm = np.linalg.norm(v_cartesienne)
        
        V_mesure_norme.append(v_norm)
        
    # Affichage Résultats
    plt.figure()
    plt.plot(time_vector, V_mesure_norme, label='Vitesse Robot |OE|')
    plt.axhline(V_cible, color='r', linestyle='--', label='Consigne V')
    plt.xlabel('Temps (s)')
    plt.ylabel('Vitesse (m/s)')
    plt.title('Suivi de Vitesse opérationnelle')
    plt.legend()
    plt.grid()
    plt.show()

def main():
    # 1. Génération de trajectoire (Module V.4)
    print("Calcul de la trajectoire...")
    # Paramètres (Exemple)
    O = [0.25, -0.15, 0.5]
    R = 0.1
    V = 0.05
    
    time_vec, q_traj, qp_traj, _ = traj(O, R, V, Debug=False)
    dt = time_vec[1] - time_vec[0]
    
    # 2. Initialisation Simulateur
    robot_id, joint_indices = init_simulation(dt)
    
    # 3. Simulation Position
    simuler_position(robot_id, joint_indices, time_vec, q_traj, O, R)
    
    # 4. Simulation Vitesse
    simuler_vitesse(robot_id, joint_indices, time_vec, qp_traj, V)
    
    p.disconnect()

if __name__ == "__main__":
    main()