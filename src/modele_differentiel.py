import numpy as np
from src.matrice_tn import generate_transformation_matrices, calcul_T06_global


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
    Utilise la convention DH Modifié (l'axe i est porté par z_i).
    """
    # 1. Calcul des matrices cumulées absolues (T01, T02, ... T06)
    T_abs = []
    T_curr = np.eye(4)

    for M in matrices:
        T_curr = np.dot(T_curr, M)
        T_abs.append(T_curr)

    # Position de l'organe terminal (OT) -> Origine du repère 6
    _, ot = calculate_z_and_o(T_abs[5])

    if Debug:
        print(f"Debug Jacobienne - OT: {ot}")

    # 2. Calcul des colonnes de la Jacobienne
    cols = []
    for i in range(6):
        # En DH Modifié, l'axe moteur i est porté par z_i de la matrice T0i
        zi, oi = calculate_z_and_o(T_abs[i])

        # Vitesse Linéaire Jv = z_i ^ (OT - O_i)
        vec_levier = ot - oi
        Jv = np.cross(zi, vec_levier)

        # Vitesse Angulaire Jw = z_i
        Jw = zi

        # Concaténation pour faire une colonne de 6 éléments
        col = np.concatenate((Jv, Jw))
        cols.append(col)

    # Création de la matrice (Transposée car on a ajouté colonne par colonne)
    J = np.array(cols).T
    return J


def MDD(dq, J):
    """ Modèle Différentiel Direct : Vitesse Articulaire -> Vitesse Cartésienne """
    return np.dot(J, dq)


def MDI(dX, J):
    """ Modèle Différentiel Inverse : Vitesse Cartésienne -> Vitesse Articulaire """
    # pinv (pseudo-inverse) gère les cas où la matrice est singulière
    return np.dot(np.linalg.pinv(J), dX)


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
        # 1. Où est-on ? (MGD)
        mats = generate_transformation_matrices(q, dh_params)
        T06 = calcul_T06_global(mats)
        curr_pos = T06[:3, 3]

        # 2. Erreur
        err = target_pos - curr_pos
        err_norm = np.linalg.norm(err)

        if Debug and i % 10 == 0:
            print(f"Iter {i}: Erreur = {err_norm:.5f} m")

        # 3. Condition d'arrêt
        if err_norm < tol:
            if Debug: print(f"Succès MGI en {i} itérations !")
            # Normalisation des angles entre -pi et pi
            return (q + np.pi) % (2 * np.pi) - np.pi

        # 4. Correction via Jacobienne Inverse
        J = Jacob_geo(mats)
        # On ne corrige que la position X,Y,Z -> on prend les 3 premières lignes
        J_pos = J[:3, :]

        dq = np.dot(np.linalg.pinv(J_pos), err)

        # 5. Mise à jour avec Gain (alpha) pour la stabilité
        q = q + alpha * dq

    if Debug:
        print(f"Echec MGI : Erreur finale {err_norm:.4f}m")
    return None