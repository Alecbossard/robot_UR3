import numpy as np
from src.const_v import dh
from src.matrice_tn import generate_transformation_matrices, calcul_T06_global
from src.modele_differentiel import MGI_numerique


def main():
    print("=== PROJET UR3 : Validation MGD & MGI ===\n")

    # ---------------------------------------------------------
    # ETAPE 1 : Utiliser le MGD pour créer une Cible
    # ---------------------------------------------------------
    # On choisit une configuration "Bras levé et plié" (arbitraire)
    q_cible_connue = [0, np.pi / 2, -np.pi / 4, 0, -np.pi / 2, 0]

    # On calcule où se trouve l'outil pour cette configuration
    matrices_cible = generate_transformation_matrices(q_cible_connue, dh)
    T06_cible = calcul_T06_global(matrices_cible)
    position_xyz_cible = T06_cible[:3, 3]

    print("1. DEFINITION DE LA CIBLE (via MGD) :")
    print(f"   Angles réels : {np.round(q_cible_connue, 3)}")
    print(f"   Position XYZ calculée : {np.round(position_xyz_cible, 4)}")
    print("-" * 50)

    # ---------------------------------------------------------
    # ETAPE 2 : Utiliser le MGI pour retrouver les angles
    # ---------------------------------------------------------
    # On efface la mémoire du robot : on le met dans une position neutre
    q_depart = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    print("2. RESOLUTION MGI (Le robot cherche la cible...) :")
    q_trouve = MGI_numerique(position_xyz_cible, q_depart, dh, alpha=0.5, Debug=True)

    print("-" * 50)

    # ---------------------------------------------------------
    # ETAPE 3 : Vérification Finale
    # ---------------------------------------------------------
    if q_trouve is not None:
        print("3. VERIFICATION :")
        print(f"   Angles trouvés par MGI : {np.round(q_trouve, 3)}")

        # On recalcule la position avec les angles trouvés pour voir si on est bon
        mats_verif = generate_transformation_matrices(q_trouve, dh)
        pos_verif = calcul_T06_global(mats_verif)[:3, 3]

        dist = np.linalg.norm(pos_verif - position_xyz_cible)
        print(f"   Position atteinte      : {np.round(pos_verif, 4)}")
        print(f"   Précision (Erreur)     : {dist:.6f} m")

        if dist < 1e-3:
            print("\n   >>> SUCCES : Le MGI et le MGD sont cohérents ! <<<")
        else:
            print("\n   >>> ECHEC : Précision insuffisante. <<<")
    else:
        print("Erreur critique : Le MGI n'a pas trouvé de solution.")


if __name__ == "__main__":
    main()