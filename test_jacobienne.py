import numpy as np
from src.const_v import dh
from src.matrice_tn import generate_transformation_matrices, calcul_T06_global
from src.modele_differentiel import Jacob_geo


def mgd_position(q):
    """ Helper : Renvoie juste (x,y,z) pour une config q """
    mats = generate_transformation_matrices(q, dh)
    T = calcul_T06_global(mats)
    return T[:3, 3]


def test_jacobienne_validation():
    print("==================================================")
    print("       TEST DE LA JACOBIENNE (VALIDATION)")
    print("==================================================\n")

    # 1. Configuration de test (Arbitraire mais non singulière)
    # On évite les 0 parfaits pour avoir des dérivées claires partout
    q_test = np.array([0.1, -0.5, 0.8, -0.2, 0.5, 0.1])

    print(f"Configuration de test q : {q_test}")

    # 2. Calcul Analytique (Votre fonction Jacob_geo)
    mats = generate_transformation_matrices(q_test, dh)
    J_geo = Jacob_geo(mats)

    # On isole la partie Vitesse Linéaire (3 premières lignes)
    J_v = J_geo[:3, :]

    print("\n--- Comparaison : Jacobienne vs Différences Finies ---")
    print("On bouge chaque axe de epsilon et on regarde si l'outil bouge comme prévu.\n")

    epsilon = 1e-6  # Petit déplacement angulaire (1 microradian)

    all_ok = True

    for i in range(6):  # Pour chaque articulation
        # A. Prédiction via Jacobienne
        # Vitesse cartésienne = J * Vitesse articulaire
        # Si on bouge de 'eps' sur l'axe i, le déplacement cartésien est J[:, i] * eps
        deplacement_predit = J_v[:, i] * epsilon

        # B. Calcul Réel via MGD (Différences finies)
        q_plus = q_test.copy()
        q_plus[i] += epsilon

        pos_init = mgd_position(q_test)
        pos_pert = mgd_position(q_plus)

        deplacement_reel = pos_pert - pos_init

        # C. Comparaison (Erreur)
        erreur = np.linalg.norm(deplacement_predit - deplacement_reel)

        status = "OK" if erreur < 1e-8 else "ERREUR"
        if erreur >= 1e-8: all_ok = False

        print(f"Axe {i + 1} :")
        print(f"  - Prédit (J) : {deplacement_predit}")
        print(f"  - Réel (MGD) : {deplacement_reel}")
        print(f"  -> Delta     : {erreur:.2e} m  [{status}]")
        print("-" * 30)

    print("\n==================================================")
    if all_ok:
        print(">>> RÉSULTAT : LA JACOBIENNE EST VALIDE ! <<<")
        print("Elle décrit parfaitement les variations du MGD.")
    else:
        print(">>> RÉSULTAT : Il y a des erreurs dans la Jacobienne.")
        print("Vérifiez vos produits vectoriels ou la convention (DH Modifié vs Standard).")
    print("==================================================")


if __name__ == "__main__":
    test_jacobienne_validation()