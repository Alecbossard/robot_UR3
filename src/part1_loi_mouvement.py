import numpy as np
import matplotlib.pyplot as plt


def calcul_loi_mouvement(R, V, dt=0.005):
    """
    V.1 : Calcule la loi de mouvement s(t), s_dot(t), s_ddot(t) pour le profil A->B->C->A.
    """
    # Phase 1: Accélération (A->B, quart de tour)
    dist_acc = (np.pi * R) / 2
    acc = V ** 2 / (2 * dist_acc)
    t1 = V / acc

    # Phase 2: Vitesse constante (B->C, quart de tour)
    dist_const = (np.pi * R) / 2
    dt_const = dist_const / V
    t2 = t1 + dt_const

    # Phase 3: Décélération (C->A, demi-tour)
    dist_dec = np.pi * R
    dec = V ** 2 / (2 * dist_dec)
    dt_dec = V / dec
    tf = t2 + dt_dec

    # Vecteur temps
    N = int(tf / dt) + 1
    time = np.linspace(0, tf, N)

    s = np.zeros_like(time)
    s_dot = np.zeros_like(time)
    s_ddot = np.zeros_like(time)

    for i, t in enumerate(time):
        if t <= t1:  # Accélération
            s_ddot[i] = acc
            s_dot[i] = acc * t
            s[i] = 0.5 * acc * t ** 2
        elif t <= t2:  # Vitesse constante
            s_t1 = dist_acc
            s_ddot[i] = 0
            s_dot[i] = V
            s[i] = s_t1 + V * (t - t1)
        else:  # Décélération
            s_t2 = dist_acc + dist_const
            t_rel = t - t2
            s_ddot[i] = -dec
            s_dot[i] = V - dec * t_rel
            s[i] = s_t2 + V * t_rel - 0.5 * dec * t_rel ** 2

            if s_dot[i] < 0: s_dot[i] = 0
            if s[i] > 2 * np.pi * R: s[i] = 2 * np.pi * R  # Saturation fin

    # Retourne les vecteurs et les temps de commutation
    return time, s, s_dot, s_ddot, (t1, t2, tf)


def afficher_courbes_loi_mouvement(time, s, s_dot, s_ddot, temps_commutation):
    t1, t2, tf = temps_commutation
    plt.figure(figsize=(10, 8))

    plt.subplot(3, 1, 1)
    plt.plot(time, s, label='s(t) [m]', color='blue')
    plt.axvline(t1, color='k', ls='--', alpha=0.5)
    plt.axvline(t2, color='k', ls='--', alpha=0.5)
    plt.legend();
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(time, s_dot, label='vitesse(t) [m/s]', color='orange')
    plt.axvline(t1, color='k', ls='--', alpha=0.5)
    plt.axvline(t2, color='k', ls='--', alpha=0.5)
    plt.legend();
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(time, s_ddot, label='accel(t) [m/s²]', color='green')
    plt.axvline(t1, color='k', ls='--', alpha=0.5)
    plt.axvline(t2, color='k', ls='--', alpha=0.5)
    plt.legend();
    plt.grid(True)

    plt.tight_layout()
    plt.show()