import numpy as np
from src.const_v import dh
from src.matrice_tn import generate_transformation_matrices
from src.modele_differentiel import Jacob_geo, MDD, MDI


def test_mdd_mdi():
    print("==================================================")
    print("       TEST COHÉRENCE MDD / MDI")
    print("==================================================\n")

    # 1. Configuration arbitraire (Non singulière)
    # On évite les angles tout à 0 pour que la Jacobienne soit inversible
    q_test = [0.1, -0.5, 0.8, -0.2, 0.5, 0.1]

    # Calcul de la Jacobienne à cette position
    mats = generate_transformation_matrices(q_test, dh)
    J = Jacob_geo(mats)

    # On travaille sur la partie linéaire (vitesse cartésienne 3D)
    # ou complète (6D) selon votre implémentation de MDI.
    # Dans votre projet, MDI utilise souvent J_v (3x6) pour suivre une trajectoire XYZ.
    # Pour ce test, on va tester les deux cas si possible, ou se concentrer sur le cas 6x6 complet
    # pour vérifier l'inversion mathématique parfaite.

    print(f"Configuration q : {q_test}")
    print(f"Conditionnement de J : {np.linalg.cond(J):.2f}\n")

    # --- TEST 1 : MDD (Vitesse Articulaire -> Vitesse Opérationnelle) ---
    print("--- Test 1 : MDD (Calcul de vitesse de sortie) ---")

    # On impose une vitesse articulaire connue (ex: l'axe 1 tourne à 1 rad/s)
    dq_input = np.array([1.0, 0.5, -0.2, 0.0, 0.0, 0.1])

    dX_output = MDD(dq_input, J)

    print(f"Entrée dq : {dq_input}")
    print(f"Sortie dX : {np.round(dX_output, 4)}")
    # dX est un vecteur [vx, vy, vz, wx, wy, wz] (6x1)

    # --- TEST 2 : MDI (Inversion) ---
    print("\n--- Test 2 : MDI (Retrouver la vitesse d'entrée) ---")
    print("On donne dX au MDI et on regarde s'il retrouve dq.")

    # Note : Si on utilise la Jacobienne complète 6x6 et qu'elle n'est pas singulière,
    # on doit retrouver exactement dq_input.
    dq_recupere = MDI(dX_output, J)

    print(f"dq récupéré : {np.round(dq_recupere, 4)}")

    erreur = np.linalg.norm(dq_input - dq_recupere)
    print(f"Erreur de reconstruction : {erreur:.6e}")

    if erreur < 1e-10:
        print(">>> SUCCÈS : Le MDI inverse parfaitement le MDD (Jacobienne 6x6).")
    else:
        print(">>> ATTENTION : L'inversion n'est pas parfaite (Singularité ou sous-détermination).")

    # --- TEST 3 : MDI Partiel (Cas Trajectoire XYZ) ---
    # C'est le cas réel de votre fonction 'traj' : on ne donne que [vx, vy, vz]
    print("\n--- Test 3 : Cas réel Trajectoire (Juste XYZ) ---")

    J_v = J[:3, :]  # Sous-Jacobienne (3x6)
    dX_xyz = dX_output[:3]  # On ne garde que la vitesse linéaire

    print(f"Cible Vitesse XYZ : {np.round(dX_xyz, 4)}")

    # On demande au robot de faire cette vitesse XYZ
    # Comme il a 6 moteurs pour 3 libertés, il y a une infinité de solutions.
    # La pseudo-inverse va trouver la solution avec la plus petite norme (la plus économe).
    dq_sol = MDI(dX_xyz, J_v)

    print(f"Solution dq trouvée : {np.round(dq_sol, 4)}")

    # Vérification : Est-ce que cette solution donne bien la bonne vitesse XYZ ?
    dX_verif = np.dot(J_v, dq_sol)
    err_proj = np.linalg.norm(dX_verif - dX_xyz)

    print(f"Vitesse XYZ obtenue : {np.round(dX_verif, 4)}")
    print(f"Erreur sur la vitesse : {err_proj:.6e}")

    if err_proj < 1e-10:
        print(">>> SUCCÈS : Le robot respecte la consigne de vitesse XYZ.")
    else:
        print(">>> ECHEC : Le robot ne va pas à la bonne vitesse.")


if __name__ == "__main__":
    test_mdd_mdi()