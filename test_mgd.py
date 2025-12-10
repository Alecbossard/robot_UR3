import numpy as np
from src.const_v import dh
from src.matrice_tn import generate_transformation_matrices, calcul_T06_global


def test_mgd_numerique():
    print("==================================================")
    print("       TEST DU MGD (CALCUL NUMÉRIQUE)")
    print("==================================================\n")

    # --- Cas 1 : Configuration Zéro (Tout à 0) ---
    # Le robot est 'à plat' ou dans sa config initiale.
    q_zero = [0, 0, 0, 0, 0, 0]

    print(f"Test 1 : Configuration q = {q_zero}")

    # 1. Génération des matrices élémentaires (T01, T12, etc.)
    matrices = generate_transformation_matrices(q_zero, dh)

    # 2. Produit matriciel cumulé (Fonction numérique)
    T06 = calcul_T06_global(matrices)

    # Extraction Position
    pos = T06[:3, 3]
    print("Matrice T06 résultante :")
    print(np.round(T06, 2))
    print(f"\n=> Position Outil : X={pos[0]:.4f}, Y={pos[1]:.4f}, Z={pos[2]:.4f}")
    print("-" * 40)

    # --- Cas 2 : Configuration 'Chandelle' (Bras vertical) ---
    # Pour l'UR3, lever le bras implique souvent q2 = -pi/2 (ou +pi/2 selon sens)
    # On teste une config où le robot pointe vers le haut pour vérifier Z_max.
    q_vert = [0, np.pi / 2, 0, 0, 0, 0]

    print(f"\nTest 2 : Configuration Verticale q = [0, pi/2, 0, 0, 0, 0]")

    mats_v = generate_transformation_matrices(q_vert, dh)
    T_v = calcul_T06_global(mats_v)
    pos_v = T_v[:3, 3]

    print(f"=> Position Outil : X={pos_v[0]:.4f}, Y={pos_v[1]:.4f}, Z={pos_v[2]:.4f}")

    # Vérification simple : Z doit être la somme des longueurs verticales
    # r1 (base) + a2 (bras) + a3 (av-bras) + r5 (poignet) + r6 (main) ?
    # Selon votre modèle : r1 + a2 + a3 + r5 (si alignés)
    hauteur_estimee = dh["r_i"][0] + dh["a_i_m1"][2] + dh["a_i_m1"][3] + dh["r_i"][4]
    # Note: vérifiez les indices selon votre const_v.py
    print(f"(Hauteur théorique estimée ~ {hauteur_estimee:.4f} m)")


if __name__ == "__main__":
    test_mgd_numerique()