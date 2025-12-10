import numpy as np
import matplotlib.pyplot as plt

# Import des modules refactorisés
from src.part1_loi_mouvement import calcul_loi_mouvement, afficher_courbes_loi_mouvement
from src.part2_trajectoire_operationnelle import calcul_trajectoire_operationnelle, afficher_courbes_operationnelles
from src.part3_analyse_tache import calcul_vitesse_OE, afficher_tache_X_t
from src.part4_generation_articulaire import traj, plot_resultats_articulaires


def main():
    print("===============================================================")
    print("PROJET UR3 : GÉNÉRATION DE TRAJECTOIRE COMPLÈTE (V.1 -> V.4)")
    print("===============================================================\n")

    # --- Paramètres de la simulation ---
    # Cercle situé devant le robot (X positif), centré en Y
    O = [0.25, -0.15, 0.5]  # Centre [Cx, Cy, Cz] en mètres
    R = 0.1  # Rayon en mètres (10 cm)
    V = 0.05  # Vitesse désirée en m/s (5 cm/s)

    print(f"PARAMÈTRES :")
    print(f" - Centre O : {O}")
    print(f" - Rayon R  : {R} m")
    print(f" - Vitesse V: {V} m/s\n")

    # =========================================================================
    # PARTIE V.1 : LOI DE MOUVEMENT TEMPORELLE s(t)
    # =========================================================================
    print("--- V.1 : Calcul de la loi de mouvement s(t) ---")
    time, s, s_dot, s_ddot, transitions = calcul_loi_mouvement(R, V)

    t1, t2, tf = transitions
    print(f" -> Durée totale : {tf:.2f} s")
    print(f" -> Phases : Accel [0-{t1:.2f}s], Const [{t1:.2f}-{t2:.2f}s], Decel [{t2:.2f}-{tf:.2f}s]")
    print(" -> Affichage des courbes s(t), s_dot(t), s_ddot(t)... (Fermez la fenêtre pour continuer)")

    afficher_courbes_loi_mouvement(time, s, s_dot, s_ddot, transitions)
    print("OK.\n")

    # =========================================================================
    # PARTIE V.2 : TRAJECTOIRE OPÉRATIONNELLE GÉOMÉTRIQUE X(s)
    # =========================================================================
    print("--- V.2 : Calcul de la trajectoire opérationnelle géométrique X(s) ---")
    # On utilise les vecteurs s(t) calculés précédemment pour générer X(t)
    X, dX, ddX = calcul_trajectoire_operationnelle(O, R, s, s_dot, s_ddot)

    print(" -> Affichage de la trajectoire 3D et des profils géométriques... (Fermez la fenêtre)")
    afficher_courbes_operationnelles(time, X, dX, ddX, O, R)
    print("OK.\n")

    # =========================================================================
    # PARTIE V.3 : ANALYSE DE LA TÂCHE X(t) ET VITESSE OE
    # =========================================================================
    print("--- V.3 : Analyse de la tâche X(t) et vérification vitesse OE ---")
    # Calcul de la norme de la vitesse cartésienne
    v_norm = calcul_vitesse_OE(dX)
    v_max_atteinte = np.max(v_norm)

    print(f" -> Vitesse max atteinte par OE : {v_max_atteinte:.4f} m/s (Cible : {V} m/s)")
    print(" -> Affichage des composantes X(t) et de la vitesse |OE|... (Fermez la fenêtre)")

    afficher_tache_X_t(time, X, dX, ddX, v_norm)
    print("OK.\n")

    # =========================================================================
    # PARTIE V.4 : GÉNÉRATION ARTICULAIRE q(t) (INVERSE KINEMATICS)
    # =========================================================================
    print("--- V.4 : Génération de mouvement articulaire q(t) ---")
    print(" -> Calcul MGI + MDI (Cela peut prendre quelques secondes)...")

    # La fonction traj réutilise la logique V1/V2 en interne et applique MGI/MDI
    # On lui passe Debug=False pour ne pas ré-afficher les courbes internes maintenant
    t_art, q, qp, qpp = traj(O, R, V, Debug=False)

    print(f" -> Trajectoire générée : {len(t_art)} points.")
    print(" -> Affichage des résultats articulaires (Positions, Vitesses, Accélérations)...")

    plot_resultats_articulaires(t_art, q, qp, qpp)
    print("\n=== SIMULATION TERMINÉE ===")


if __name__ == "__main__":
    main()