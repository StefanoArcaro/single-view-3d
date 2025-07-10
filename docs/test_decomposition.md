FUNCTION run_homography_roundtrip_tests(K: 3×3 intrinsics, N: number of trials)

  INITIALIZE empty lists:
    rot_errors      ← []
    trans_dir_errors← []
    scale_biases    ← []

  FOR i IN 1…N DO

    #— 1. Sample a random plane in front of the camera —#

    1.1  v ← [randn(), randn(), randn()]                   # 3×1 Gaussian
    1.2  n_true ← v / norm(v)                              # unit plane normal
    1.3  d_true ← Uniform(0.1, 10.0)                       # distance in meters
    1.4  t_true ← d_true * n_true                          # plane origin in camera coords

    #— 2. Sample a random “front‑facing” camera rotation —#

    2.1  α ← Uniform(–90°, +90°)                           # pitch
    2.2  β ← Uniform(–90°, +90°)                           # yaw
    2.3  γ ← Uniform(–180°, +180°)                         # roll

    2.4  R_x ← rotation_matrix_about_X(α)
    2.5  R_y ← rotation_matrix_about_Y(β)
    2.6  R_z ← rotation_matrix_about_Z(γ)
    2.7  R_true ← R_z • R_y • R_x                          # compose in Z–Y–X order

    #— 3. Build the ground‑truth homography —#

    3.1  # Extract the first two columns of R_true
         r1 ← R_true[:,0]
         r2 ← R_true[:,1]

    3.2  # Form the 3×3 homography mapping plane coords [u,v,1]→image:
         H_gt ← K • [ r1 , r2 , t_true ]

    #— 4. (Optional) verify H_gt by projecting a grid —#

    #    You may project a small u–v grid through H_gt and compare to
    #    direct K[R_true|t_true]·[u,v,0,1] to sanity‑check.

    #— 5. Recover poses and pick best solution —#

    5.1  solutions ← recover_all_poses_from_homography(H_gt, K)
    5.2  (R_rec, t_rec, _) ← select_best_solution(solutions, expected_z_positive=True)

    #— 6. Compute error metrics —#

    6.1  # Rotation‑error θ_e (deg)
         R_delta ← transpose(R_rec) • R_true
         cosθ     ← (trace(R_delta) – 1) / 2
         θ_e      ← arccos(clamp(cosθ, –1, +1)) * (180/π)

    6.2  # Translation‑direction error φ_e (deg)
         t̂_rec   ← t_rec   / norm(t_rec)
         t̂_true  ← t_true  / norm(t_true)
         cosφ     ← dot(t̂_rec, t̂_true)
         φ_e      ← arccos(clamp(cosφ, –1, +1)) * (180/π)

    6.3  # Scale bias s_e (%)
         s_e      ← abs(norm(t_rec) – norm(t_true)) / norm(t_true) * 100

    #— 7. Record errors —#

    APPEND θ_e to rot_errors
    APPEND φ_e to trans_dir_errors
    APPEND s_e to scale_biases

  END FOR

  RETURN { 
    rotation_errors      : rot_errors,
    translation_errors   : trans_dir_errors,
    scale_bias_percent   : scale_biases
  }