import numpy as np
from src.const_v import dh
from src.matrice_tn import generate_transformation_matrices, calcul_T06_global
from src.modele_differentiel import Jacob_geo, MDD


def main():
    print("=== Simulation Robot UR3 - MGD & Jacobienne ===\n")

    # 1. Configuration des angles (q)
    # Exemple : [0, 90°, 0, 0, -90°, 0]
    q_test = [0, np.pi / 2, 0, 0, -np.pi / 2, 0]
    print(f"Configuration q : {np.round(q_test, 3)}")

    # 2. Calcul des Matrices de Transformation
    matrices = generate_transformation_matrices(q_test, dh)

    # Calcul de la position finale (MGD)
    T06 = calcul_T06_global(matrices)
    pos_finale = T06[:3, 3]

    print("\n--- Modèle Géométrique Direct (MGD) ---")
    print(f"Position Outil (x, y, z) : {np.round(pos_finale, 4)}")

    # 3. Calcul de la Jacobienne
    J = Jacob_geo(matrices, Debug=True)

    print("\n--- Matrice Jacobienne (J) ---")
    # Affichage propre
    with np.printoptions(precision=3, suppress=True):
        print(J)

    # 4. Test du Modèle Différentiel Direct (MDD)
    # On applique une petite vitesse sur l'axe 1 et l'axe 6
    dq_test = [0.1, 0, 0, 0, 0, 0.1]

    vitesse_outil = MDD(dq_test, J)

    print("\n--- Test MDD ---")
    print(f"Vitesses articulaires dq : {dq_test}")
    print(f"Vitesse Outil résultante [vx vy vz wx wy wz] :")
    with np.printoptions(precision=4, suppress=True):
        print(vitesse_outil)


if __name__ == "__main__":
    main()