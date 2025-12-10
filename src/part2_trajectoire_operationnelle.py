import numpy as np
import matplotlib.pyplot as plt


def calcul_trajectoire_operationnelle(O, R, s, s_dot, s_ddot):
    """
    V.2 : Calcule X(t), dX(t), ddX(t) dans l'espace opérationnel.
    Cercle dans le plan XZ, Y constant.
    """
    Cx, Cy, Cz = O
    u = s / R  # Angle relatif

    # Positions X(t)
    x = Cx - R * np.sin(u)
    y = np.full_like(s, Cy)
    z = Cz + R * np.cos(u)
    X = np.vstack((x, y, z)).T

    # Vitesses dX(t)
    vx = -s_dot * np.cos(u)
    vy = np.zeros_like(s)
    vz = -s_dot * np.sin(u)
    dX = np.vstack((vx, vy, vz)).T

    # Accélérations ddX(t)
    ax = -s_ddot * np.cos(u) + (s_dot ** 2 / R) * np.sin(u)
    ay = np.zeros_like(s)
    az = -s_ddot * np.sin(u) - (s_dot ** 2 / R) * np.cos(u)
    ddX = np.vstack((ax, ay, az)).T

    return X, dX, ddX


def afficher_courbes_operationnelles(time, X, dX, ddX, Center, Rayon):
    """ Affiche les profils X, Z et la 3D """
    fig = plt.figure(figsize=(12, 8))

    # Position X
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(time, X[:, 0], 'b', label='x(t)')
    ax1.plot(time, X[:, 2], 'g', label='z(t)')
    ax1.set_title('Positions X et Z')
    ax1.grid(True);
    ax1.legend()

    # Trajectoire 3D
    ax3d = fig.add_subplot(2, 2, 2, projection='3d')
    ax3d.plot(X[:, 0], X[:, 1], X[:, 2], label='Trajectoire')
    ax3d.scatter(Center[0], Center[1], Center[2], c='r', marker='x', label='Centre')
    ax3d.set_xlabel('X');
    ax3d.set_ylabel('Y');
    ax3d.set_zlabel('Z')
    ax3d.legend()

    plt.tight_layout()
    plt.show()