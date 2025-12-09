import numpy as np
from src.const_v import dh
from src.matrice_tn import generate_transformation_matrices, calcul_T06_global
from src.modele_differentiel import Jacob_geo


def test_validation_mgd():
    """
    Teste le MGD sur des positions géométriques connues du robot UR3.
    """
    print("\n=== TEST 1 : VALIDATION DU MGD (Positions connues) ===")

    # --- Cas A : Position "Zéro" (Tout à 0) ---
    # Sur votre modèle DH Modifié UR3 :
    # Le bras est à l'horizontale le long de X (à cause de l'offset pi/2 sur q1 et -pi/2 sur q6)
    # Z devrait être juste la hauteur de la base (r1 = 0.152)
    q_zero = [0, 0, 0, 0, 0, 0]

    mats = generate_transformation_matrices(q_zero, dh)
    T = calcul_T06_global(mats)
    pos = T[:3, 3]

    print(f"\nConfiguration q = {q_zero}")
    print(f"Position calculée : X={pos[0]:.3f}, Y={pos[1]:.3f}, Z={pos[2]:.3f}")

    # Vérification théorique (UR3 à plat)
    # X devrait être -(r2) + r6 ? Non, regardons votre modèle.
    # Avec vos offsets : q1 tourne de 90°. Le bras part en Y.
    # C'est ce que nous avions validé : Y ~ 540mm, Z ~ 152mm.

    # --- Cas B : Position "Chandelle" (Bras Vertical) ---
    # q2 = pi/2 (pour lever le bras a2)
    # q4 = -pi/2 (pour aligner le poignet)
    q_vertical = [0, np.pi / 2, 0, -np.pi / 2, 0, 0]

    mats_v = generate_transformation_matrices(q_vertical, dh)
    T_v = calcul_T06_global(mats_v)
    pos_v = T_v[:3, 3]

    print(f"\nConfiguration q = [0, -pi/2, 0, -pi/2, 0, 0] (Verticale)")
    print(f"Position calculée : X={pos_v[0]:.3f}, Y={pos_v[1]:.3f}, Z={pos_v[2]:.3f}")

    # Vérification: Z doit être proche de la hauteur max (~0.77m ou 0.69m selon poignet)
    if pos_v[2] > 0.6:
        print(">>> SUCCESS : La hauteur Z semble cohérente pour une position verticale.")
    else:
        print(">>> FAILURE : Le bras ne semble pas monter.")


def test_validation_jacobienne():
    """
    Vérifie la Jacobienne par la méthode des différences finies.
    On compare la vitesse calculée par J * dq avec le déplacement réel (MGD(q+dq) - MGD(q)).
    """
    print("\n=== TEST 2 : VALIDATION JACOBIENNE (Différences Finies) ===")

    # 1. Configuration arbitraire (pas de singularité pour éviter les divisions bizarres)
    q = np.array([0.1, -0.5, 0.8, -0.2, 0.5, 0.1])

    # 2. Calcul analytique via votre fonction Jacob_geo
    mats = generate_transformation_matrices(q, dh)
    J = Jacob_geo(mats)

    # On ne teste que la partie linéaire (vitesse v), donc les 3 premières lignes de J
    J_v = J[:3, :]

    print(f"Configuration q : {q}")
    print("Vérification colonne par colonne (Axe par Axe) :")

    epsilon = 1e-6  # Une toute petite variation

    all_passed = True

    for i in range(6):
        # On crée un dq qui ne bouge que l'axe i
        dq = np.zeros(6)
        dq[i] = epsilon

        # --- Méthode A : Via Jacobienne ---
        # Vitesse = J * vitesse_angulaire (ici vitesse = 1 rad/s * epsilon)
        # Déplacement théorique = Vitesse * 1 = J_colonne_i * epsilon
        delta_pos_jacob = J_v[:, i] * epsilon

        # --- Méthode B : Via MGD (Réalité) ---
        q_perturb = q.copy()
        q_perturb[i] += epsilon

        # Position originale
        p_orig = calcul_T06_global(generate_transformation_matrices(q, dh))[:3, 3]
        # Position perturbée
        p_pert = calcul_T06_global(generate_transformation_matrices(q_perturb, dh))[:3, 3]

        delta_pos_mgd = p_pert - p_orig

        # --- Comparaison ---
        diff = np.linalg.norm(delta_pos_jacob - delta_pos_mgd)

        print(f"  Axe {i + 1} : Erreur entre Jacobienne et MGD = {diff:.2e} m", end="")

        if diff < 1e-8:
            print(" -> OK")
        else:
            print(" -> ERREUR (La Jacobienne ne correspond pas au MGD)")
            all_passed = False

    if all_passed:
        print("\n>>> SUCCESS : La Jacobienne est mathématiquement cohérente avec le MGD.")
    else:
        print("\n>>> FAILURE : Il y a une incohérence dans le calcul de la Jacobienne.")


if __name__ == "__main__":
    test_validation_mgd()
    test_validation_jacobienne()