import numpy as np
from src.matrice_tn import generate_transformation_matrices, calcul_T06_global


def calculate_z_and_o(T):
    """ Extrait le vecteur Z (axe de rotation) et l'origine O (position) d'une matrice 4x4. """
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

    # Position de l'organe terminal (OT)
    _, ot = calculate_z_and_o(T_cumul[5])

    # 2. Calcul des colonnes de la Jacobienne
    J_cols = []
    for i in range(6):
        # En DH Modifié, l'axe moteur i est porté par z_i de la matrice T0i
        zi, oi = calculate_z_and_o(T_cumul[i])

        # Partie Linéaire (Jv) = z_i ^ (OT - O_i)
        Jv = np.cross(zi, ot - oi)

        # Partie Angulaire (Jw) = z_i
        Jw = zi

        # Concaténation
        J_col = np.concatenate((Jv, Jw))
        J_cols.append(J_col)

    # Création de la matrice (Transposée)
    J = np.array(J_cols).T

    return J


def MGI_numerique(target_pos, q_init, dh_params, max_iter=100, tol=1e-4, alpha=0.5, Debug=False):
    """
    Inverse Kinematics (MGI) par méthode de Newton-Raphson amortie.
    Retrouve les angles q pour atteindre target_pos [x,y,z].
    """
    q = np.array(q_init, dtype=float)
    target_pos = np.array(target_pos, dtype=float)

    if Debug:
        print(f"\n--- Début MGI Numérique ---")
        print(f"Cible : {target_pos}")

    for i in range(max_iter):
        # 1. Calculer la position actuelle
        matrices = generate_transformation_matrices(q, dh_params)
        T06 = calcul_T06_global(matrices)
        curr_pos = T06[:3, 3]  # [x, y, z]

        # 2. Calculer l'erreur
        err = target_pos - curr_pos
        err_norm = np.linalg.norm(err)

        if Debug and i % 10 == 0:
            print(f"Iter {i}: Erreur = {err_norm:.5f} m")

        # 3. Condition d'arrêt
        if err_norm < tol:
            if Debug: print(f"Succès MGI en {i} itérations !")
            # Normalisation des angles entre -pi et pi
            return (q + np.pi) % (2 * np.pi) - np.pi

        # 4. Calculer la Jacobienne
        J = Jacob_geo(matrices)

        # 5. Inversion (On ne corrige que la position X,Y,Z -> 3 premières lignes)
        J_pos = J[:3, :]
        dq = np.dot(np.linalg.pinv(J_pos), err)

        # 6. Mise à jour avec Gain (alpha) pour la stabilité
        q = q + alpha * dq

    print(f"Echec MGI : Erreur finale {err_norm:.4f}")
    return None