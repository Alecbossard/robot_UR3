import numpy as np


def mgd_vers_simulation(q_mgd):
    """
    Convertit la configuration articulaire du MGD (Théorique)
    vers la configuration attendue par PyBullet (Réel/Simulé).
    """
    # 1. Définition des corrections
    # Signes : Inversion de l'axe 2
    signes = np.array([1, -1, -1, 1, 1, 1])

    # Offsets : Décalages pour les axes 4 et 5
    offsets = np.array([0, 0, 0, -np.pi / 2, np.pi, 0])

    # 2. Calcul
    q_mgd = np.array(q_mgd)
    q_sim = (q_mgd * signes) + offsets

    return q_sim


def simulation_vers_mgd(q_sim):
    """
    Convertit la configuration articulaire de la Simulation (PyBullet)
    vers la configuration du MGD (Théorique).

    Inverse de mgd_vers_simulation.
    """
    # 1. Définition des corrections (Doivent être identiques à l'aller)
    signes = np.array([1, -1, -1, 1, 1, 1])
    offsets = np.array([0, 0, 0, -np.pi / 2, np.pi, 0])

    # 2. Calcul Inverse
    # Formule aller : q_sim = (q_mgd * signes) + offsets
    # Formule retour : q_mgd = (q_sim - offsets) / signes

    q_sim = np.array(q_sim)

    # Note : diviser par 'signes' ou multiplier par 'signes' revient au même ici (1/-1 = -1)
    q_mgd = (q_sim - offsets) * signes

    return q_mgd

# --- Exemple de vérification avec vos valeurs ---
if __name__ == "__main__":
    # Cas 1 : Tout à zéro
    q_test1 =  [0, -np.pi/2, np.pi/2, -np.pi/2, np.pi, 0]
    # print(f"MGD: {q_test1} -> Simu: {mgd_vers_simulation(q_test1)}")
    # Résultat attendu : [0, 0, 0, -1.57, 3.14, 0] -> OK
    q_retour = simulation_vers_mgd(q_test1)
    print(f"Simu -> MGD : {np.round(q_retour, 3)}")
    # Cas 2 : Bras Vertical
    # q_test2 = [0, np.pi / 2, 0, 0, 0, 0]
    # print(f"MGD: {q_test2} -> Simu: {np.round(mgd_vers_simulation(q_test2), 2)}")
    # Résultat attendu : [0, -1.57, 0, -1.57, 3.14, 0] -> OK