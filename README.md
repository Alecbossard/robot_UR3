# UR3 Robot Kinematics (MGD / MGI)

Small educational Python project for modelling the kinematics of the 6‑DOF **UR3** manipulator.

The code implements:

- **Modified DH model** (Khalil / Craig style) for the UR3 arm
- **Forward kinematics (MGD)**: computation of all `T_i^{i-1}` and global `T_0^6`
- **Geometric Jacobian** of the robot
- **Differential models**:
  - MDD (Modèle Différentiel Direct): joint velocities → end‑effector twist
  - MDI (Modèle Différentiel Inverse) via Jacobian pseudo‑inverse
- **Numerical inverse kinematics (MGI)** using an iterative Jacobian‑based solver
- A set of **validation tests** checking consistency between MGD and Jacobian

The project was written as part of a robotics course to validate the modelling of a UR3 robot in Python.

---

## 1. Repository structure

```text
robot_UR3-main/
├── main.py                    # Demo script: creates a target with MGD, solves it with MGI and checks the error
├── test_validation.py         # Validation tests for MGD and the Jacobian
├── src/
│   ├── __init__.py
│   ├── const_v.py             # UR3 geometric constants + modified DH parameters
│   ├── matrice_tn.py          # DH transformation matrices (T_{i-1}^i) and global T_0^6 computation
│   └── modele_differentiel.py # Geometric Jacobian, MDD, MDI, numerical MGI based on the Jacobian
└── README.md                  # This file
```

The comments in the source files are mostly in French, but the function names are explicit and kept short for educational use.

---

## 2. Requirements

- Python **3.9+** (recommended)
- Packages:
  - `numpy`

You can install the dependencies with:

```bash
pip install numpy
```

If you prefer, you can also create a small `requirements.txt` containing:

```text
numpy
```

and then run:

```bash
pip install -r requirements.txt
```

---

## 3. How it works (short explanation)

### 3.1 Kinematic model (DH)

The file `src/const_v.py` contains:

- The **geometric dimensions** of the UR3 (link lengths, offsets…)
- The **modified DH parameters** (arrays `a_i_m1`, `alpha_i_m1`, `r_i`, `theta_offset`, etc.)

The function `generate_transformation_matrices(...)` in `src/matrice_tn.py`:

1. Builds each homogeneous transform `T_{i-1}^i(q_i)` using the modified DH convention.
2. Stacks all 6 transforms in a list.
3. Multiplies them to obtain the global transform `T_0^6` with `calcul_T06_global(...)`.

This gives the **forward kinematics (MGD)** of the UR3.

### 3.2 Geometric Jacobian & differential models

In `src/modele_differentiel.py`:

- `Jacob_geo(matrices, Debug=False)` computes the **geometric Jacobian (6×6)** from the list of transforms:
  - For each joint, the `z_i` axis and origin `O_i` are extracted.
  - The linear and angular parts are assembled into the standard Jacobian form.

- `MDD(dq, J)` implements the **direct differential model**:
  - Input: joint velocities `dq`
  - Output: end‑effector twist (linear + angular velocity) `V = J · dq`

- `MGI_numerique(...)` implements a **numerical inverse kinematics**:
  - Iterative correction of the joint vector `q` to reduce the position error
  - Uses the pseudo‑inverse of the Jacobian (only the 3 first rows for XYZ position)
  - Simple damping / gain parameter for stability

The solver is **local**: convergence is guaranteed only near the initial guess and does not handle joint limits or obstacles.

---

## 4. Running the project

From the root of the project (`robot_UR3-main/`), run:

```bash
python main.py
```

What `main.py` does:

1. Chooses a **known configuration** `q_cible_connue` (e.g. “arm up & bent”).
2. Uses the **forward kinematics (MGD)** to compute the corresponding end‑effector position `T_0^6`.
3. Calls the **numerical MGI (`MGI_numerique`)** to find a configuration `q_sol` that reaches this target.
4. Re‑applies the MGD with `q_sol` and compares the obtained position with the initial target.
5. Prints the **final error** and a success/failure message depending on the precision (e.g. < 1e‑3 m).

You should see an output similar to:

```text
=== PROJET UR3 : Validation MGD & MGI ===

1. DEFINITION DE LA CIBLE (MGD)
   q cible connue       : [...]
   Position cible (xyz) : [...]

2. RESOLUTION MGI (Le robot cherche la cible...)
   Solution trouvée q*  : [...]
   Position atteinte    : [...]
   Précision (Erreur)   : 0.000xxx m

   >>> SUCCES : Le MGI et le MGD sont cohérents ! <<<
```

(Exact values depend on the chosen configuration and numerical tolerances.)

---

## 5. Validation tests

To run the validation scripts:

```bash
python test_validation.py
```

This script performs:

1. **Forward kinematics checks** on specific joint configurations:
   - Zero configuration
   - “Straight arm” or other analytically known poses
   - Compares the resulting position `T_0^6` to the expected one (within a tight tolerance)

2. **Jacobian validation**:
   - Computes an approximate Jacobian by finite differences using the MGD
   - Compares it to the analytic Jacobian from `Jacob_geo`
   - Prints a pass/fail message depending on the norm of the difference

If everything is consistent, you should see messages like:

```text
=== TEST 1 : VALIDATION DU MGD (Positions connues) ===
...
>>> SUCCESS : Le MGD est cohérent avec le modèle géométrique.

=== TEST 2 : VALIDATION DE LA JACOBIENNE ===
...
>>> SUCCESS : La Jacobienne est mathématiquement cohérente avec le MGD.
```

---

## 6. Limitations & possible extensions

Current limitations:

- Only **position** is used in the inverse kinematics (no orientation target).
- No handling of **joint limits** or **collision constraints**.
- The numerical solver is a basic Jacobian pseudo‑inverse with a fixed step size.

Possible extensions:

- Extend `MGI_numerique` to also control the **end‑effector orientation**.
- Add **joint limits** and basic **singularity handling**.
- Implement a **damped least‑squares** (Levenberg–Marquardt style) IK solver.
- Connect the model to a **simulator** (PyBullet, Gazebo, etc.) or to a real UR3 via ROS.

---

## 7. Author & context

This repository is intended for **educational use** in a robotics course (modélisation robotique / kinematics of serial robots).  

Feel free to adapt it to:

- Other 6‑DOF arms (by changing the DH parameters)
- Teaching material (TP / labs)
- Quick prototypes for kinematics and Jacobian‑based control.

