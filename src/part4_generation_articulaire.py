import numpy as np
import matplotlib.pyplot as plt

# Imports internes des autres modules du projet
from src.const_v import dh
from src.matrice_tn import generate_transformation_matrices
from src.modele_differentiel import Jacob_geo, MGI_numerique

# Imports des parties V.1 et V.2 (Refactoring)
from src.part1_loi_mouvement import calcul_loi_mouvement
from src.part2_trajectoire_operationnelle import calcul_trajectoire_operationnelle


def traj(O, R, V, Debug=False):
    """
    V.4 : Génération de mouvement dans l'espace articulaire.
    Combine V.1, V.2 et les modèles inverses pour sortir q(t).

    Returns:
        time, q, qp, qpp
    """
    # 1. Génération de la consigne opérationnelle (Appel aux parties V.1 et V.2)
    # Note: On récupère le tuple des temps dans '_' mais on ne l'utilise pas ici
    time, s, s_dot, s_ddot, _ = calcul_loi_mouvement(R, V)

    X_ref, dX_ref, ddX_ref = calcul_trajectoire_operationnelle(O, R, s, s_dot, s_ddot)

    N = len(time)
    dt = time[1] - time[0]

    # Tableaux de sortie
    q = np.zeros((N, 6))
    qp = np.zeros((N, 6))

    # État initial estimé pour le MGI
    q_prev = np.array([0.0, np.pi / 2, -np.pi / 4, 0.0, -np.pi / 2, 0.0])

    if Debug: print(f"Calcul de la trajectoire articulaire ({N} points)...")

    for i in range(N):
        # A. Position Articulaire (MGI)
        q_sol = MGI_numerique(X_ref[i], q_prev, dh, max_iter=20, alpha=0.8, tol=1e-5)

        if q_sol is None:
            if Debug: print(f"Warn: MGI non convergé itération {i}")
            q_sol = q_prev

        q[i, :] = q_sol
        q_prev = q_sol

        # B. Vitesse Articulaire (MDI / Jacobienne)
        mats = generate_transformation_matrices(q_sol, dh)
        J = Jacob_geo(mats)
        J_v = J[:3, :]  # Partie linéaire
        qp[i, :] = np.dot(np.linalg.pinv(J_v), dX_ref[i])

    # C. Accélération (Dérivation numérique)
    qpp = np.gradient(qp, dt, axis=0)

    return time, q, qp, qpp


def plot_resultats_articulaires(time, q, qp, qpp, temps_commutation=None):
    """ Affiche les courbes q, q_dot, q_ddot (avec temps de commutation si fournis) """

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    labels = [f'q{i + 1}' for i in range(6)]

    # Positions
    for i in range(6):
        axes[0].plot(time, q[:, i], label=labels[i])
    axes[0].set_title('Positions Articulaires q(t)')
    axes[0].set_ylabel('rad')
    axes[0].legend(loc='right', fontsize='small')
    axes[0].grid(True)

    # Vitesses
    for i in range(6):
        axes[1].plot(time, qp[:, i], label=labels[i])
    axes[1].set_title(r'Vitesses Articulaires $\dot{q}(t)$')
    axes[1].set_ylabel('rad/s')
    axes[1].grid(True)

    # Accélérations
    for i in range(6):
        axes[2].plot(time, qpp[:, i], label=labels[i])
    axes[2].set_title(r'Accélérations Articulaires $\ddot{q}(t)$')
    axes[2].set_ylabel('rad/s²')
    axes[2].set_xlabel('Temps (s)')
    axes[2].grid(True)

    # Ajout des temps de commutation t1, t2 si on les a
    if temps_commutation is not None:
        t1, t2, tf = temps_commutation

        for axe in axes:
            axe.axvline(t1, color='k', ls='--', alpha=0.5)
            axe.axvline(t2, color='k', ls='--', alpha=0.5)

    plt.tight_layout()
    plt.show()
