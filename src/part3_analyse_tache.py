import numpy as np
import matplotlib.pyplot as plt


def calcul_vitesse_OE(dX):
    """
    Calcule la vitesse scalaire (norme) du point OE au cours du temps.
    """
    return np.linalg.norm(dX, axis=1)


def afficher_tache_X_t(time, X, dX, ddX, v_norm):
    """
    Affiche les courbes demandées pour le module V.3
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Position
    axes[0, 0].plot(time, X[:, 0], label='x')
    axes[0, 0].plot(time, X[:, 1], label='y')
    axes[0, 0].plot(time, X[:, 2], label='z')
    axes[0, 0].set_title('Positions X(t)')
    axes[0, 0].grid(True);
    axes[0, 0].legend()

    # Vitesse
    axes[1, 0].plot(time, dX[:, 0], label='vx')
    axes[1, 0].plot(time, dX[:, 1], label='vy')
    axes[1, 0].plot(time, dX[:, 2], label='vz')
    axes[1, 0].set_title('Vitesses dX(t)')
    axes[1, 0].grid(True);
    axes[1, 0].legend()

    # Accélération
    axes[0, 1].plot(time, ddX[:, 0], label='ax')
    axes[0, 1].plot(time, ddX[:, 1], label='ay')
    axes[0, 1].plot(time, ddX[:, 2], label='az')
    axes[0, 1].set_title('Accélérations ddX(t)')
    axes[0, 1].grid(True);
    axes[0, 1].legend()

    # Norme Vitesse OE
    axes[1, 1].plot(time, v_norm, 'k', lw=2, label='||V_OE||')
    axes[1, 1].set_title('Vitesse Scalaire OE')
    axes[1, 1].grid(True);
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()