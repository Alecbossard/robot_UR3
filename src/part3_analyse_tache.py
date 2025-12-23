import numpy as np
import matplotlib.pyplot as plt


from src.const_v import dh
from src.matrice_tn import generate_transformation_matrices, calcul_T06_global
from src.modele_differentiel import Jacob_geo



def calcul_vitesse_OE(dX):
    """
    Calcule la vitesse scalaire (norme) du point OE au cours du temps.
    """
    return np.linalg.norm(dX, axis=1)


def afficher_tache_X_t(time, X, dX, ddX, v_norm, temps_commutation=None):
    """
    Affiche les courbes demandées pour le module V.3.
    Si temps_commutation est fourni (t1, t2, tf), on affiche les temps de
    commutation t1 et t2 sur les courbes.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Position
    axes[0, 0].plot(time, X[:, 0], label='x')
    axes[0, 0].plot(time, X[:, 1], label='y')
    axes[0, 0].plot(time, X[:, 2], label='z')
    axes[0, 0].set_title('Positions X(t)')
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    # Vitesse
    axes[1, 0].plot(time, dX[:, 0], label='vx')
    axes[1, 0].plot(time, dX[:, 1], label='vy')
    axes[1, 0].plot(time, dX[:, 2], label='vz')
    axes[1, 0].set_title('Vitesses dX(t)')
    axes[1, 0].grid(True)
    axes[1, 0].legend()

    # Accélération
    axes[0, 1].plot(time, ddX[:, 0], label='ax')
    axes[0, 1].plot(time, ddX[:, 1], label='ay')
    axes[0, 1].plot(time, ddX[:, 2], label='az')
    axes[0, 1].set_title('Accélérations ddX(t)')
    axes[0, 1].grid(True)
    axes[0, 1].legend()

    # Norme Vitesse OE
    axes[1, 1].plot(time, v_norm, 'k', lw=2, label='||V_OE||')
    axes[1, 1].set_title('Vitesse Scalaire OE')
    axes[1, 1].grid(True)
    axes[1, 1].legend()

    # Ajout des temps de commutation si fournis
    if temps_commutation is not None:
        t1, t2, tf = temps_commutation
        for axe in [axes[0, 0], axes[1, 0], axes[0, 1]]:
            axe.axvline(t1, color='k', ls='--', alpha=0.5)
            axe.axvline(t2, color='k', ls='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


def calcul_X_robot_et_erreurs(temps, X_consigne, dX_consigne, q, q_point):
    """
    À partir de la trajectoire articulaire (q, q_point), on recalcule :
      - la position de l'outil X_robot(t) via la MGD,
      - la vitesse de l'outil dX_robot(t) via la jacobienne,
    puis on renvoie les erreurs par rapport aux consignes.
    """
    nb_points = len(temps)

    X_robot = np.zeros_like(X_consigne)
    dX_robot = np.zeros_like(dX_consigne)

    for i in range(nb_points):
        q_i = q[i, :]
        q_point_i = q_point[i, :]

        # MGD : position de l'outil
        matrices = generate_transformation_matrices(q_i, dh)
        T06 = calcul_T06_global(matrices)
        X_robot[i, :] = T06[:3, 3]

        # Modèle différentiel : vitesse de l'outil
        J = Jacob_geo(matrices)
        J_v = J[:3, :]   # partie linéaire
        dX_robot[i, :] = J_v.dot(q_point_i)

    # Erreurs
    erreur_X = X_consigne - X_robot
    erreur_dX = dX_consigne - dX_robot

    return X_robot, dX_robot, erreur_X, erreur_dX


def afficher_erreurs_X(temps, erreur_X, erreur_dX):
    """
    Affiche les erreurs sur X(t) et Xdot(t) sous forme de deux sous-graphiques.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Erreurs de position
    axes[0].plot(temps, erreur_X[:, 0], label='e_x')
    axes[0].plot(temps, erreur_X[:, 1], label='e_y')
    axes[0].plot(temps, erreur_X[:, 2], label='e_z')
    axes[0].set_title('Erreurs sur la position X(t)')
    axes[0].set_ylabel('Erreur [m]')
    axes[0].grid(True)
    axes[0].legend()

    # Erreurs de vitesse
    axes[1].plot(temps, erreur_dX[:, 0], label='e_vx')
    axes[1].plot(temps, erreur_dX[:, 1], label='e_vy')
    axes[1].plot(temps, erreur_dX[:, 2], label='e_vz')
    axes[1].set_title('Erreurs sur la vitesse Xdot(t)')
    axes[1].set_ylabel('Erreur [m/s]')
    axes[1].set_xlabel('Temps (s)')
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.show()
