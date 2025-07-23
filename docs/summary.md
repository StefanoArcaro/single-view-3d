# Homography Decomposition Algorithms

## Algorithm 1: Recover All Poses from Homography

**Input:** Homography matrix H (3×3), Camera intrinsics K (3×3)  
**Output:** List of 4 possible pose solutions (R, t, n)

```
ALGORITHM RecoverAllPosesFromHomography(H, K)
BEGIN
    // Step 1: Normalize homography by removing camera intrinsics
    H_norm ← K⁻¹ × H
    
    // Step 2: Extract rotation columns and translation
    r₁ ← H_norm[:, 0]    // First rotation column
    r₂ ← H_norm[:, 1]    // Second rotation column  
    t ← H_norm[:, 2]     // Translation vector
    
    // Step 3: Compute scale factor for normalization
    scale ← ||r₁||
    
    // Step 4: Generate all possible solutions from sign ambiguity
    solutions ← empty list
    
    FOR each sign ∈ {+1, -1} DO
        // Apply sign and scale normalization
        r₁_scaled ← sign × r₁ / scale
        r₂_scaled ← sign × r₂ / scale
        t_scaled ← sign × t / scale
        
        // Compute third rotation column
        r₃ ← r₁_scaled × r₂_scaled    // Cross product
        
        // Construct approximate rotation matrix
        R_approx ← [r₁_scaled | r₂_scaled | r₃]
        
        // Step 5: Project to SO(3) using SVD
        U, Σ, Vᵀ ← SVD(R_approx)
        
        // Ensure proper rotation (det = +1)
        IF det(U × Vᵀ) < 0 THEN
            U[:, -1] ← -U[:, -1]    // Flip last column
        END IF
        
        R ← U × Vᵀ                  // Final rotation matrix
        n ← R[2, :]                 // Plane normal (third row)
        
        // Add both normal orientations
        solutions.append((R, t_scaled, n))
        solutions.append((R, t_scaled, -n))
    END FOR
    
    RETURN solutions
END
```

## Algorithm 2: Select Best Pose Solution

**Input:** List of pose solutions, expected_z_positive flag  
**Output:** Best pose solution (R, t, n) or NULL

```
ALGORITHM SelectBestSolution(solutions, expected_z_positive)
BEGIN
    IF solutions is empty THEN
        RETURN NULL
    END IF
    
    best_solution ← NULL
    best_score ← +∞
    
    // Evaluate each candidate solution
    FOR each (R, t, n) ∈ solutions DO
        // Constraint 1: Check Z-coordinate constraint
        IF expected_z_positive AND t[2] ≤ 0 THEN
            CONTINUE    // Skip: plane behind camera
        END IF
        
        // Constraint 2: Avoid numerical instability
        IF |t[2]| < ε THEN    // where ε = 1e-8
            CONTINUE    // Skip: very small Z values
        END IF
        
        // Scoring heuristic: prefer head-on views
        score ← (|t[0]| + |t[1]|) / |t[2]|
        
        // Penalty for backfacing plane normals
        IF n[2] > 0 THEN    // Normal pointing away from camera
            score ← score × 2
        END IF
        
        // Update best solution
        IF score < best_score THEN
            best_score ← score
            best_solution ← (R, t, n)
        END IF
    END FOR
    
    RETURN best_solution
END
```

## Algorithm 3: Derive Metric Homography

**Input:** Pixel homography H_px (3×3), template dimensions in pixels and metric units, origins  
**Output:** Metric homography H_metric (3×3) mapping real-world coordinates to image pixels

```
ALGORITHM DeriveMetricHomography(H_px, size_px, size_metric, origin_px, origin_metric)
BEGIN
    // Input validation
    VALIDATE H_px is 3×3 matrix with finite values
    VALIDATE all dimensions > 0
    
    // Extract dimensions and origins
    h_px, w_px ← size_px           // Height, width in pixels
    h_m, w_m ← size_metric         // Height, width in metric units
    u₀_px, v₀_px ← origin_px       // Pixel origin coordinates (x, y)
    X₀_m, Y₀_m ← origin_metric     // Metric origin coordinates (x, y)
    
    // Step 1: Translate metric origin to (0,0)
    T₁ ← [1   0  -X₀_m]
         [0   1  -Y₀_m]
         [0   0   1   ]
    
    // Step 2: Scale from metric units to pixel units
    s_x ← w_px / w_m              // Scale factor in x-direction
    s_y ← h_px / h_m              // Scale factor in y-direction
    
    S ← [s_x  0   0]
        [0   s_y  0]
        [0    0   1]
    
    // Step 3: Translate to template pixel origin
    T₂ ← [1   0  u₀_px]
         [0   1  v₀_px]
         [0   0   1   ]
    
    // Step 4: Combine transformations
    // Chain: metric coords → centered → scaled → positioned → scene
    M_metric_to_template ← T₂ × S × T₁
    H_metric ← H_px × M_metric_to_template
    
    // Validation
    VALIDATE H_metric contains only finite values
    
    RETURN H_metric
END
```


# Camera calibration algorithms

## Algorithm 4: Camera Calibration from Orthogonal Line Pairs

**Input:** 5 pairs of orthogonal lines (m, l) from rectangular template  
**Output:** Camera intrinsics matrix K (3×3) or failure indication

```
ALGORITHM CalibrateFromOrthogonalLines(pairs_lines)
BEGIN
    // Step 1: Set up constraint matrix for dual image of absolute conic (DIAC)
    // Each orthogonal line pair provides one constraint: mᵀ × D × l = 0
    // where D is the 3×3 symmetric DIAC matrix (6 unknowns)
    
    A ← zeros(5, 6)  // 5 constraints, 6 unknowns for symmetric matrix
    
    // Step 2: Fill constraint matrix
    // D = [d₀  d₁  d₃]  →  vectorized as [d₀, d₁, d₂, d₃, d₄, d₅]
    //     [d₁  d₂  d₄]
    //     [d₃  d₄  d₅]
    
    FOR i = 0 to 4 DO
        m, l ← pairs_lines[i]        // Get i-th orthogonal line pair
        
        // Constraint: mᵀ × D × l = 0 expanded to linear form
        A[i, 0] ← m[0] × l[0]        // d₀ coefficient
        A[i, 1] ← m[0] × l[1] + m[1] × l[0]  // d₁ coefficient  
        A[i, 2] ← m[1] × l[1]        // d₂ coefficient
        A[i, 3] ← m[0] × l[2] + m[2] × l[0]  // d₃ coefficient
        A[i, 4] ← m[1] × l[2] + m[2] × l[1]  // d₄ coefficient
        A[i, 5] ← m[2] × l[2]        // d₅ coefficient
    END FOR
    
    // Step 3: Solve homogeneous system A × d = 0
    U, Σ, Vᵀ ← SVD(A)
    d ← Vᵀ[-1, :]                   // Solution is last row (smallest singular value)
    
    // Step 4: Reconstruct symmetric DIAC matrix
    D ← [d[0]  d[1]  d[3]]
        [d[1]  d[2]  d[4]]
        [d[3]  d[4]  d[5]]
    
    // Step 5: Validate solution quality
    residual ← ||A × d||
    IF residual > threshold THEN
        RETURN FAILURE  // Poor solution quality
    END IF
    
    // Step 6: Enforce rank-2 constraint (DIAC must be singular)
    rank_D ← rank(D)
    IF rank_D = 3 THEN
        // Noise caused full rank - enforce theoretical rank-2 constraint
        U_D, Σ_D, Vᵀ_D ← SVD(D)
        Σ_D[-1] ← 0                 // Set smallest singular value to zero
        D ← U_D × diag(Σ_D) × Vᵀ_D  // Reconstruct with rank 2
    END IF
    
    // Step 7: Extract camera intrinsics via Cholesky decomposition
    // DIAC D = K⁻ᵀ × K⁻¹, so K can be recovered from D
    TRY
        L ← cholesky(D)             // D = L × Lᵀ
        K ← (L⁻¹)ᵀ × L[2,2]        // Normalize and transpose
        RETURN K
    CATCH decomposition_error
        RETURN FAILURE  // D not positive definite
    END TRY
END
```

## Algorithm 5: Adapted Zhang's Calibration for single view

**Input:** List of homographies H₁, H₂, ..., Hₙ from planar templates (n ≥ 3)  
**Output:** Camera intrinsics matrix K (3×3) or failure indication

```
ALGORITHM CalibrateFromHomographies(homographies)
BEGIN
    // Step 1: Validate input
    IF length(homographies) < 3 THEN
        RETURN FAILURE  // Insufficient homographies
    END IF
    
    // Step 2: Set up constraint matrix for absolute conic
    // Each homography provides 2 constraints on matrix B = K⁻ᵀ × K⁻¹
    // B is symmetric: B = [B₁₁  B₁₂  B₁₃]  →  vectorized as [B₁₁, B₁₂, B₂₂, B₁₃, B₂₃, B₃₃]
    //                     [B₁₂  B₂₂  B₂₃]
    //                     [B₁₃  B₂₃  B₃₃]
    
    V ← empty_matrix(2n, 6)  // 2 constraints per homography, 6 unknowns
    
    // Step 3: Build constraint matrix from homographies
    FOR each homography H in homographies DO
        H ← H / H[2,2]  // Normalize homography
        
        // Extract constraint vectors using Zhang's formulation
        v₁₂ ← BuildConstraintVector(H, 0, 1)  // Orthogonality constraint
        v₁₁ ← BuildConstraintVector(H, 0, 0)  // Equal scaling constraint  
        v₂₂ ← BuildConstraintVector(H, 1, 1)  // (part 1)
        
        // Add two constraints per homography
        V[row] ← v₁₂           // h₁ ⊥ h₂ (columns orthogonal)
        V[row+1] ← v₁₁ - v₂₂   // ||h₁|| = ||h₂|| (equal scaling)
        row ← row + 2
    END FOR
    
    // Step 4: Solve homogeneous system V × b = 0
    U, Σ, Vᵀ ← SVD(V)
    b ← Vᵀ[-1, :]  // Solution is last row (smallest singular value)
    
    // Step 5: Extract symmetric matrix B components
    B₁₁, B₁₂, B₂₂, B₁₃, B₂₃, B₃₃ ← b
    
    // Step 6: Recover intrinsic parameters from B matrix
    // Using closed-form solution from Zhang's method
    w ← B₁₁ × B₂₂ × B₃₃ - B₁₂² × B₃₃ - B₁₁ × B₂₃² + 2 × B₁₂ × B₁₃ × B₂₃ - B₂₂ × B₁₃²
    d ← B₁₁ × B₂₂ - B₁₂²
    
    // Compute principal point
    cₓ ← (B₁₂ × B₂₃ - B₂₂ × B₁₃) / d
    cᵧ ← (B₁₂ × B₁₃ - B₁₁ × B₂₃) / d
    
    // Compute focal lengths  
    fₓ ← √(w / (d × B₁₁))
    fᵧ ← √(w / (d² × B₁₁))
    
    // Compute skew parameter
    s ← √(w / (d² × B₁₁)) × w
    
    // Step 7: Construct intrinsic matrix
    K ← [fₓ   s   cₓ]
        [0   fᵧ   cᵧ]
        [0    0    1]
    
    RETURN K
END

// Helper function to build constraint vectors
FUNCTION BuildConstraintVector(H, i, j)
BEGIN
    hᵢ ← H[:, i]  // i-th column of H
    hⱼ ← H[:, j]  // j-th column of H
    
    // Return constraint vector for hᵢᵀ × B × hⱼ
    RETURN [hᵢ[0] × hⱼ[0],
            hᵢ[0] × hⱼ[1] + hᵢ[1] × hⱼ[0],
            hᵢ[1] × hⱼ[1],
            hᵢ[0] × hⱼ[2] + hᵢ[2] × hⱼ[0],
            hᵢ[1] × hⱼ[2] + hᵢ[2] × hⱼ[1],
            hᵢ[2] × hⱼ[2]]
END
```