import numpy as np
from src.const_v import dh
from src.matrice_tn import generate_transformation_matrices, calcul_T06_global
from src.modele_differentiel import MGI_numerique


def test_mgi_validation():
    print("==================================================")
    print("       TEST DU MGI (INVERSE KINEMATICS)")
    print("==================================================\n")

    # --- ÉTAPE 1 : Créer une cible valide (via le MGD) ---
    # On choisit une configuration arbitraire mais réaliste
    # (Bras levé, coude plié, poignet tourné)
    q_cible_connue = [0.5, -0.8, 1.2, -0.5, 1.0, 0.5]

    # On calcule la position cartésienne correspondante
    mats_cible = generate_transformation_matrices(q_cible_connue, dh)
    T_cible = calcul_T06_global(mats_cible)
    pos_cible = T_cible[:3, 3]  # [x, y, z]

    print(f"1. Configuration Cible (Théorique) :")
    print(f"   Angles : {q_cible_connue}")
    print(f"   Position XYZ à atteindre : {np.round(pos_cible, 4)}")
    print("-" * 40)

    # --- ÉTAPE 2 : Lancer le MGI ---
    # On part d'une position "neutre" (pas trop loin, mais différente)
    # pour voir si l'algorithme converge vers la cible.
    q_depart = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    print(f"2. Recherche MGI (Départ : {q_depart})...")

    # Appel de votre fonction MGI
    q_trouve = MGI_numerique(
        target_pos=pos_cible,
        q_init=q_depart,
        dh_params=dh,
        max_iter=50,
        alpha=0.5,  # Gain d'apprentissage
        tol=1e-5,  # Précision demandée (10 microns)
        Debug=True  # On veut voir les itérations
    )

    print("-" * 40)

    # --- ÉTAPE 3 : Vérification ---
    if q_trouve is not None:
        print("3. RÉSULTATS :")
        print(f"   Angles trouvés : {np.round(q_trouve, 4)}")

        # Vérification : On réinjecte les angles trouvés dans le MGD
        mats_verif = generate_transformation_matrices(q_trouve, dh)
        pos_verif = calcul_T06_global(mats_verif)[:3, 3]

        # Calcul de l'erreur
        dist_erreur = np.linalg.norm(pos_verif - pos_cible)
        print(f"   Position atteinte : {np.round(pos_verif, 4)}")
        print(f"   Erreur de position : {dist_erreur:.6f} m")

        if dist_erreur < 1e-4:
            print("\n>>> SUCCÈS : Le MGI fonctionne parfaitement ! <<<")

            # Note sur l'unicité
            # On vérifie si les angles sont les mêmes que ceux de départ
            diff_angles = np.linalg.norm(np.array(q_trouve) - np.array(q_cible_connue))
            if diff_angles < 0.1:
                print("(L'algorithme a retrouvé la même configuration articulaire)")
            else:
                print("(L'algorithme a trouvé une AUTRE configuration valide pour la même position)")
        else:
            print("\n>>> ÉCHEC : La position n'est pas assez précise. <<<")
    else:
        print("\n>>> ÉCHEC : Le MGI n'a pas convergé. <<<")
        print("Essayez d'augmenter le nombre d'itérations ou de changer le point de départ.")


if __name__ == "__main__":
    test_mgi_validation()