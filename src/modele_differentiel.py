import numpy as np
from src.matrice_tn import generate_transformation_matrices, calcul_T06_global
import sympy as sp


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


def Jacob_analytique(q_val=None, Debug=False):
    """
    Calcule la Jacobienne Analytique (Symbolique) pour l'UR3.
    Adapté du projet DigiTwin pour le robot UR3 (6 axes, DH Modifié).

    Args:
        q_val (list, optional): Valeurs numériques des angles [q1...q6] (en radians).
                                Si fourni, retourne la matrice numérique.
        Debug (bool): Affiche les matrices symboliques intermédiaires.

    Returns:
        sp.Matrix (si q_val=None) ou np.ndarray (si q_val fourni)
    """
    # 1. Définition des symboles
    # q1..q6 sont les variables, les autres sont les paramètres géométriques constants
    q1, q2, q3, q4, q5, q6 = sp.symbols('q1 q2 q3 q4 q5 q6')

    # Paramètres UR3 (provenant de votre const_v.py)
    r1, r2, r4, r5, r6 = 0.152, 0.120, 0.010, 0.083, 0.082
    a2, a3 = 0.244, 0.213

    # Paramètres DH Modifiés UR3 : (alpha_i-1, a_i-1, r_i, theta_offset)
    # Attention aux signes de r2 et aux offsets définis dans votre projet
    dh_params = [
        (0, 0, r1, q1 + sp.pi / 2),  # Axe 1
        (sp.pi / 2, 0, -r2, q2),  # Axe 2 (r2 négatif dans votre const_v)
        (0, a2, 0, q3),  # Axe 3
        (0, a3, r4, q4 + sp.pi / 2),  # Axe 4 (r4 ajouté)
        (sp.pi / 2, 0, r5, q5),  # Axe 5
        (-sp.pi / 2, 0, r6, q6 - sp.pi / 2)  # Axe 6
    ]

    # 2. Construction des Matrices de Transformation Symboliques
    T_abs = []  # Liste des matrices T0i
    T_curr = sp.eye(4)  # Matrice identité 4x4

    # Fonction locale pour DH Modifié (Symbolique)
    def mat_dh_sym(alpha, a, r, theta):
        c = sp.cos(theta)
        s = sp.sin(theta)
        ca = sp.cos(alpha)
        sa = sp.sin(alpha)
        return sp.Matrix([
            [c, -s, 0, a],
            [s * ca, c * ca, -sa, -r * sa],
            [s * sa, c * sa, ca, r * ca],
            [0, 0, 0, 1]
        ])

    for alpha, a, r, theta in dh_params:
        T_elem = mat_dh_sym(alpha, a, r, theta)
        T_curr = T_curr * T_elem  # Multiplication symbolique
        # On simplifie un peu à chaque étape pour éviter des expressions géantes
        # (Note: sp.simplify peut être lent, on l'utilise avec parcimonie)
        T_abs.append(T_curr)

    # T_abs contient [T01, T02, T03, T04, T05, T06]

    # 3. Extraction des vecteurs Z et O pour la Jacobienne
    # Rappel DH Modifié : l'axe du moteur i est z_i de la matrice T0i

    # Position de l'organe terminal (OT) -> Origine de T06
    OT = T_abs[5][:3, 3]

    cols = []

    for i in range(6):
        T_i = T_abs[i]

        # Axe de rotation z_i (3ème colonne)
        z_i = T_i[:3, 2]

        # Centre de l'articulation o_i (4ème colonne)
        o_i = T_i[:3, 3]

        # Partie Linéaire : Jv = z_i ^ (OT - o_i)
        Jv = z_i.cross(OT - o_i)

        # Partie Angulaire : Jw = z_i
        Jw = z_i

        # Construction de la colonne (Jv en haut, Jw en bas)
        col = Jv.col_join(Jw)
        cols.append(col)

    # Assemblage de la matrice Jacobienne complète (6x6)
    J_sym = cols[0]
    for i in range(1, 6):
        J_sym = J_sym.row_join(cols[i])

    if Debug:
        print("\n--- Debug Jacobienne Analytique (Symbolique) ---")
        print("La matrice est calculée. Affichage des termes simplifiés (peut être long)...")
        # On affiche juste la dimension pour confirmer
        print(f"Dimension J: {J_sym.shape}")
        # sp.pprint(J_sym) # Décommenter si vous voulez voir les équations (très larges !)

    # 4. Évaluation Numérique (si q_val est fourni)
    if q_val is not None:
        if len(q_val) != 6:
            raise ValueError("q_val doit contenir 6 angles.")

        # Dictionnaire de substitution : {q1: val1, q2: val2...}
        subs_dict = {
            q1: q_val[0], q2: q_val[1], q3: q_val[2],
            q4: q_val[3], q5: q_val[4], q6: q_val[5]
        }

        # Substitution et conversion en numpy float
        J_num = np.array(J_sym.evalf(subs=subs_dict)).astype(np.float64)
        return J_num

    return J_sym