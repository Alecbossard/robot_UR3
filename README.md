# Projet UR3 – Modélisation cinématique et génération de trajectoire (Python)

Ce dépôt contient un petit projet pédagogique autour du robot **UR3** :
- Modèle géométrique direct (MGD) et inverse (MGI)
- Jacobienne et modèle différentiel
- Génération d’une **trajectoire circulaire** dans le plan (X, Z) et conversion en trajectoire articulaire

Objectif (exigences client) : le client doit pouvoir
- **entrer/modifier** le centre du cercle **O**, le rayon **R** et la vitesse **V**
- **tester** la fonction `traj(O, R, V)`
- **visualiser** toutes les courbes demandées

---

## Prérequis

- Python 3.x
- Dépendances : `numpy`, `matplotlib`, `sympy`

Installation recommandée :

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

---

## Exécution rapide

Le programme de base à lancer pour générer la trajectoire complète et afficher les figures est :

```bash
python main_traj.py
```

Le script ouvre plusieurs fenêtres matplotlib. Fermer une fenêtre permet de passer à la suivante.

---

## Modifier O, R, V (paramètres de la trajectoire)

Dans `main_traj.py`, modifier directement :

- `O = [Ox, Oy, Oz]` (m) : centre du cercle dans l’espace opérationnel
- `R` (m) : rayon du cercle
- `V` (m/s) : vitesse cible de l’outil le long de la trajectoire

Remarque : la trajectoire est dans le plan **(X, Z)**, donc **y(t) est constant** (= `Oy`).

---

## Sorties minimales attendues

L’exécution de `python main_traj.py` affiche au minimum :

1) **Loi de mouvement**
- Les trois courbes : `s(t)`, `ṡ(t)`, `s̈(t)`
- Avec l’affichage des **temps de commutation** (`t1`, `t2`, `tf`) :
  - imprimés dans la console
  - et tracés par des lignes verticales (pointillées) sur les graphes

2) **Coordonnées cartésiennes (plan X, Z)**
- Les trois courbes : `x(t)`, `ẋ(t)`, `ẍ(t)` avec `t1`, `t2`
- Les trois courbes : `z(t)`, `ż(t)`, `z̈(t)` avec `t1`, `t2`
- `y(t)`, `ẏ(t)`, `ÿ(t)` : inutile car `y` est constant

3) **Vitesse de l’outil**
- Courbe de la vitesse scalaire : `||V_OE(t)|| = ||Ẋ(t)||`

4) **Trajectoire opérationnelle**
- Trajectoire `X(s)` dans l’espace opérationnel (∈ R³) via un `plot3` (cercle dans l’espace)

5) **Trajectoires articulaires**
- Courbes `q(t)` et `q̇(t)` pour **chaque articulation** (UR3 = 6 liaisons)
- (Souvent affiché aussi) `q̈(t)`
- Avec l’affichage des temps de commutation `t1`, `t2`

6) **Erreurs consigne vs robot simulé**
- Erreurs sur la position : `e_X(t) = X_consigne(t) - X_robot(t)`
- Erreurs sur la vitesse : `e_Ẋ(t) = Ẋ_consigne(t) - Ẋ_robot(t)`
- Où `X_robot` est recalculé par MGD à partir de `q(t)` et `Ẋ_robot` par `J(q)·q̇(t)`
- Les erreurs attendues sont très faibles (principalement dues au bruit numérique)

---

## Structure du dépôt

```text
robot_UR3-main/
├── main_traj.py                 # Pipeline complet : V.1 -> V.4 + affichages + erreurs
├── main.py                      # Démo MGD/MGI
├── src/
│   ├── part1_loi_mouvement.py   # s(t), ṡ(t), s̈(t) + temps de commutation + plots
│   ├── part2_trajectoire_operationnelle.py  # X(t), Ẋ(t), Ẍ(t) + trajectoire 3D
│   ├── part3_analyse_tache.py   # vitesse outil + affichages + calcul erreurs X et Ẋ
│   ├── part4_generation_articulaire.py      # traj(O,R,V) + q, q̇, q̈ + plots
│   ├── const_v.py               # Constantes / paramètres (DH, etc.)
│   ├── matrice_tn.py            # Matrices homogènes / MGD
│   └── modele_differentiel.py   # Jacobienne / modèles différentiels
└── tests_*.py                   # Scripts simples de validation
```

---

## Scripts de test 

```bash
python test_mgd.py
python test_mgi.py
python test_mdd_mdi.py
python test_jacobienne.py
```

Ces scripts aident à vérifier séparément la MGD, la MGI et les Jacobiennes.

---

