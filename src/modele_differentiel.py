import numpy as np


def calculate_z_and_o(T):
    """
    Extrait le vecteur Z (axe de rotation) et l'origine O (position) d'une matrice 4x4.
    """
    z = T[:3, 2]  # 3ème colonne : vecteur Z
    o = T[:3, 3]  # 4ème colonne : position O
    return z, o


def Jacob_geo(matrices, Debug=False):
    """
    Calcule la Matrice Jacobienne Géométrique (6x6) pour l'UR3.
    """
    # 1. Calcul des matrices cumulées (T01, T02, ... T06)
    T_cumul = []
    current_T = np.eye(4)

    for T in matrices:
        current_T = np.dot(current_T, T)
        T_cumul.append(current_T)

    # T_cumul contient [T01, T02, T03, T04, T05, T06]

    # 2. Initialisation Base
    z0 = np.array([0, 0, 1])  # Axe Z de la base
    o0 = np.array([0, 0, 0])  # Origine de la base

    # 3. Extraction des Z et O pour chaque repère
    # Note: L'indice i correspond à la transformation T0(i+1)
    zs = [z0]
    os = [o0]

    # On récupère z et o pour les repères 1 à 5 (le repère 6 sert pour l'OT)
    for i in range(5):
        z, o = calculate_z_and_o(T_cumul[i])
        zs.append(z)
        os.append(o)

    # Position de l'organe terminal (OT) -> Origine du repère 6
    _, ot = calculate_z_and_o(T_cumul[5])

    if Debug:
        print("\n--- Debug Jacobienne ---")
        print(f"OT (Position Outil): {ot}")

    # 4. Calcul des colonnes de la Jacobienne
    # J_i = [ z_{i-1} x (OT - O_{i-1}) ]  <- Vitesse linéaire
    #       [ z_{i-1}                  ]  <- Vitesse angulaire

    J_cols = []
    for i in range(6):
        z_prev = zs[i]
        o_prev = os[i]

        # Partie Linéaire (Jv)
        Jv = np.cross(z_prev, ot - o_prev)

        # Partie Angulaire (Jw)
        Jw = z_prev

        # Concaténation pour faire une colonne de 6 éléments
        J_col = np.concatenate((Jv, Jw))
        J_cols.append(J_col)

    # Création de la matrice (Transposée car on a ajouté colonne par colonne)
    J = np.array(J_cols).T

    return J


def MDD(dq, J):
    """
    Modèle Différentiel Direct : Calcule la vitesse de l'outil (dX)
    à partir des vitesses articulaires (dq).
    dX = J * dq
    """
    return np.dot(J, dq)


def MDI(dX, J):
    """
    Modèle Différentiel Inverse : Calcule les vitesses articulaires (dq)
    à partir de la vitesse d'outil désirée (dX).
    dq = inv(J) * dX
    Utilise la pseudo-inverse pour la robustesse.
    """
    # pinv (pseudo-inverse) gère les cas où la matrice est singulière ou mal conditionnée
    return np.dot(np.linalg.pinv(J), dX)