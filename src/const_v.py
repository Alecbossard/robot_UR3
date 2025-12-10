import numpy as np

# Dimensions du robot UR3 (en Mètres)
r1 = 0.152
r2 = 0.120
a2 = 0.244
a3 = 0.213
r4 = 0.010
r5 = 0.083
r6 = 0.082

# Paramètres DH Modifiés (Convention Khalil / Craig)
# Structure adaptée pour correspondre à tes résultats précédents
dh = {
    "a_i_m1":     [0,      0,        a2,     a3,     0,       0      ], # a_{i-1}
    "alpha_i_m1": [0,      np.pi/2,  0,      0,      np.pi/2, -np.pi/2], # alpha_{i-1}
    "r_i":        [r1,     -r2,       0,      r4,      r5,      r6     ], # r_i
    "theta_offset": [np.pi/2, 0,     0,      np.pi/2, 0,      -np.pi/2] # Offsets (q_fig)
}

