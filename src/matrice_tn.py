import numpy as np


def matrice_Tim1_Ti(qi, ai_m1, alphai_m1, ri, theta_offset=0):
    """
    Calcule la matrice de transformation DH entre deux articulations i-1 et i.
    """
    # L'angle total est l'angle moteur (qi) + le décalage géométrique (offset)
    theta = qi + theta_offset

    c_t = np.cos(theta)
    s_t = np.sin(theta)
    c_a = np.cos(alphai_m1)
    s_a = np.sin(alphai_m1)

    # Matrice de transformation homogène 4x4
    T = np.array([
        [c_t, -s_t, 0, ai_m1],
        [s_t * c_a, c_t * c_a, -s_a, -ri * s_a],
        [s_t * s_a, c_t * s_a, c_a, ri * c_a],
        [0, 0, 0, 1]
    ])
    return T


def generate_transformation_matrices(q, dh):
    """
    Génère la liste des matrices de transformation [T01, T12, T23...]
    basée sur les paramètres DH et les angles actuels q.
    """
    matrices = []
    num_joints = len(dh["a_i_m1"])

    for i in range(num_joints):
        # Sécurité : si q est plus court que le nombre d'axes, on met 0
        angle = q[i] if i < len(q) else 0

        T = matrice_Tim1_Ti(
            qi=angle,
            ai_m1=dh["a_i_m1"][i],
            alphai_m1=dh["alpha_i_m1"][i],
            ri=dh["r_i"][i],
            theta_offset=dh["theta_offset"][i]
        )
        matrices.append(T)

    return matrices


def calcul_T06_global(matrices):
    """
    Multiplie toutes les matrices pour obtenir la transformation finale T06 (Base -> Outil).
    """
    T_global = np.eye(4)
    for T in matrices:
        T_global = np.dot(T_global, T)
    return T_global