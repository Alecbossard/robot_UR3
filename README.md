# UR3 Robot – Kinematics and Trajectory Generation (Python)

This repository contains a small educational project for modelling the kinematics and differential behaviour of the 6‑DOF **UR3** manipulator, and for generating a **circular Cartesian trajectory** followed by a **joint–space trajectory**.

The code was written in the context of a robotics course on serial‑robot modelling (modélisation robotique).  
Most comments are in French, but all function names and scripts are explicit and kept short for learning purposes.

---

## 1. Repository structure

```text
robot_UR3-main/
├── main.py                     # Demo: forward kinematics (MGD) + numerical inverse kinematics (MGI)
├── main_traj.py                # Full pipeline: motion law + Cartesian circle + task analysis + joint trajectory
├── test_mgd.py                 # Console tests for forward kinematics
├── test_mgi.py                 # Console tests for numerical inverse kinematics
├── test_mdd_mdi.py             # Console tests for MDD / MDI differential models
├── test_jacobienne.py          # Console tests for geometric & analytic Jacobians
├── src/
│   ├── __init__.py
│   ├── const_v.py              # UR3 geometric constants + modified DH parameters and helper conversions
│   ├── matrice_tn.py           # DH transformation matrices and global T_0^6 computation
│   ├── modele_differentiel.py  # Geometric & analytic Jacobians, MDD, MDI, numerical IK
│   ├── part1_loi_mouvement.py  # Time law s(t) for the circular trajectory
│   ├── part2_trajectoire_operationnelle.py  # Cartesian circle X(t), dX(t), ddX(t)
│   ├── part3_analyse_tache.py  # Task‑space analysis (speed of a point of interest)
│   ├── part4_generation_articulaire.py      # Joint‑space trajectory generation q(t), dq(t), ddq(t)
│   └── utils.py                # Small numerical / plotting helpers
└── README.md                   # This file
```

---

## 2. Features

### Kinematic model (UR3)

- **Modified DH model** (Khalil / Craig style) for the UR3 arm.
- **Forward kinematics (MGD)**:
  - Elemental transforms `T_{i-1}^i(q_i)` for each joint.
  - Global transform `T_0^6(q)` giving end‑effector pose.
- **Numerical inverse kinematics (MGI)**:
  - Iterative Jacobian‑based solver.
  - Position‑only IK (x, y, z) using a pseudo‑inverse of the Jacobian.
  - Simple damping / step‑size control for convergence.

### Jacobians and differential models

Implemented in `src/modele_differentiel.py`:

- **Geometric Jacobian** `J_geo(q)` (6×6) built from the homogeneous transforms.
- **Analytic Jacobian** `J_ana(q)` for position‑only tasks.
- **MDD (Modèle Différentiel Direct)**:
  - Joint velocities `dq` → end‑effector twist `V = J · dq`.
- **MDI (Modèle Différentiel Inverse)**:
  - Desired twist `V_d` → joint velocities `dq` via pseudo‑inverse `J^+`.

### Trajectory generation (circular path)

Implemented in the `partX_*.py` modules and orchestrated by `main_traj.py`:

1. **Time law `s(t)`** (V.1 – `part1_loi_mouvement.py`)
   - Piecewise motion profile A → B → C → A over a circle:
     - Acceleration phase
     - Constant‑speed phase
     - Deceleration phase
   - Returns time vector, `s(t)`, `ṡ(t)`, `s̈(t)` and switching instants.

2. **Operational (task‑space) trajectory** (V.2 – `part2_trajectoire_operationnelle.py`)
   - Circular trajectory in Cartesian space around a centre `O = [Cx, Cy, Cz]` with radius `R`.
   - Computes:
     - Position `X(t)` of the point on the circle.
     - Velocity `dX(t)` and acceleration `ddX(t)`.

3. **Task analysis** (V.3 – `part3_analyse_tache.py`)
   - Computes the speed of a specific point of interest of the robot along the path.
   - Allows visualising how the UR3 base motion translates into task‑space motion.

4. **Joint‑space trajectory** (V.4 – `part4_generation_articulaire.py`)
   - Uses the numerical IK and Jacobians to generate:
     - Joint positions `q(t)`
     - Joint velocities `dq(t)`
     - Joint accelerations `ddq(t)`
   - Includes simple plotting utilities to inspect each joint over time.

---

## 3. Requirements

- Python **3.9+** (recommended).
- Python packages:
  - `numpy`
  - `matplotlib`
  - `sympy` (used for some symbolic checks / utilities)

You can install the dependencies with:

```bash
pip install numpy matplotlib sympy
```

If you prefer, create a `requirements.txt`:

```text
numpy
matplotlib
sympy
```

then run:

```bash
pip install -r requirements.txt
```

---

## 4. Quick start

Clone or unzip this repository, then in a terminal:

```bash
cd robot_UR3-main
```

### 4.1. Kinematics demo (MGD + MGI)

```bash
python main.py
```

This script:

1. Chooses a **known joint configuration**.
2. Uses the **forward kinematics** to compute the corresponding end‑effector pose.
3. Feeds this pose as a **target** to the numerical IK solver.
4. Prints the resulting joint solution and the position error.

This is a simple sanity‑check that MGD and MGI are consistent.

### 4.2. Full circular trajectory pipeline

```bash
python main_traj.py
```

This script:

1. Defines a circular path in front of the robot (centre `O`, radius `R`, speed `V`).
2. Computes the **time law** `s(t)` and the Cartesian circle `X(t)`.
3. Evaluates the **task‑space velocities**.
4. Generates the **joint‑space trajectory** `q(t), dq(t), ddq(t)` using the IK.
5. Displays several **matplotlib plots**:
   - Motion law `s(t)`, `ṡ(t)`, `s̈(t)`
   - Cartesian position / velocity / acceleration
   - Joint trajectories over time

Feel free to adjust the circle parameters (`O`, `R`, `V`) directly in `main_traj.py`.

---

## 5. Tests and validation scripts

The project comes with small console‑based tests (no external testing framework required):

```bash
# Forward kinematics checks (positions, known configurations)
python test_mgd.py

# Numerical inverse kinematics checks (convergence, error)
python test_mgi.py

# Differential models (MDD / MDI) and twist consistency
python test_mdd_mdi.py

# Geometric vs analytic Jacobians and simple numerical comparisons
python test_jacobienne.py
```

Each script prints intermediate values and errors in the terminal to help understand what is happening.

---

## 6. How to reuse / extend the code

A few ideas if you want to go further:

- Replace the DH parameters in `src/const_v.py` to model another **6‑DOF robot**.
- Add **orientation control** in the numerical IK (full 6D task instead of position‑only).
- Connect the model to a **simulator** (e.g. PyBullet, Gazebo) by sending the joint trajectories `q(t)`.
- Wrap the core functions into a **Jupyter notebook** for interactive teaching.
- Implement alternative motion laws (polynomial trajectories, trapezoidal velocity, etc.).

---

## 7. Educational context

This repository is intended for **educational use** in a robotics / kinematics course:

- Understanding **DH modelling** and homogeneous transforms.
- Practising **geometric & analytic Jacobians**.
- Linking **task‑space trajectories** to **joint‑space trajectories**.
- Experimenting with **numerical inverse kinematics** and differential control.

You are free to adapt the code for:

- Lab sessions (TP)
- Course support material
- Quick prototypes for kinematic or trajectory‑generation experiments
