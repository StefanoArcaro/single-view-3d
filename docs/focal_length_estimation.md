# Estimating the Focal Length from a Homography

This document details the derivation for estimating the focal length of a camera from a single planar homography, under simplified assumptions.

## Assumptions

We assume the following simplified camera intrinsic matrix:

$$ K = \begin{bmatrix} f & 0 & c_x \\ 0 & f & c_y \\ 0 & 0 & 1 \end{bmatrix} $$

Where:

* $f$ is the focal length (in pixels)
* $(c_x, c_y)$ is the known principal point (typically the image center)
* Skew is assumed to be zero
* Square pixels are assumed: $f_x = f_y = f$

The goal is to estimate $f$ given a single homography $H$ that maps world points lying on a plane to image points.


## Step-by-step Derivation

### 1. From Homography to Rotation

Given a homography $H$, and assuming that the world points lie on a plane $Z = 0$, the homography encodes both the camera intrinsics $K$ and part of the extrinsic rotation and translation:

$$ H = K \begin{bmatrix} \mathbf{r}_1 & \mathbf{r}_2 & \mathbf{t} \end{bmatrix} $$

From this, we can isolate the rotation directions:

$$
\mathbf{r}_1 \propto K^{-1} \mathbf{h}_1
\quad \text{and} \quad
\mathbf{r}_2 \propto K^{-1} \mathbf{h}_2
$$

Where $\mathbf{h}_1, \mathbf{h}_2$ are the first two columns of $H$.

---

### 2. Orthogonality Constraint

The rotation vectors must be orthogonal:

$$ \mathbf{r}_1^T \mathbf{r}_2 = 0 $$

Substituting $\mathbf{r}_i = K^{-1} \mathbf{h}_i$, we get:

$$
(K^{-1} \mathbf{h}_1)^T (K^{-1} \mathbf{h}_2) = 0
\quad \Rightarrow \quad
\mathbf{h}_1^T K^{-T} K^{-1} \mathbf{h}_2 = 0
$$

Let us define:

$$ \mathbf{v}_i = K^{-1} \mathbf{h}_i $$

We can write $\mathbf{v}_i$ explicitly using known $c_x, c_y$ and unknown $f$:

$$
\mathbf{v}_i
= \begin{bmatrix} x_i \\ y_i \\ z_i \end{bmatrix}
= \begin{bmatrix} (h_{i,0} - c_x h_{i,2}) / f \\ (h_{i,1} - c_y h_{i,2}) / f \\ h_{i,2} \end{bmatrix}
$$

Now compute the dot product:

$$ \mathbf{v}_1^T \mathbf{v}_2 = \frac{x_1 x_2 + y_1 y_2}{f^2} + z_1 z_2 = 0 $$

Solving for $f^2$:

$$ f^2 = -\frac{x_1 x_2 + y_1 y_2}{z_1 z_2} $$

---

### 3. Norm Equality Constraint

Since $\mathbf{r}_1$ and $\mathbf{r}_2$ are unit vectors:

$$
\|\mathbf{r}_1\|^2 = \|\mathbf{r}_2\|^2
\quad \Rightarrow \quad
\|\mathbf{v}_1\|^2 = \|\mathbf{v}_2\|^2
$$

Compute the norms:

$$
\|\mathbf{v}_1\|^2 = \frac{x_1^2 + y_1^2}{f^2} + z_1^2
\quad \text{and} \quad
\|\mathbf{v}_2\|^2 = \frac{x_2^2 + y_2^2}{f^2} + z_2^2
$$

Equating the two:

$$ \frac{x_1^2 + y_1^2}{f^2} + z_1^2 = \frac{x_2^2 + y_2^2}{f^2} + z_2^2 $$

Rearranging:

$$ \frac{(x_1^2 + y_1^2) - (x_2^2 + y_2^2)}{f^2} = z_2^2 - z_1^2 $$

Solving for $f^2$:

$$ f^2 = \frac{(x_1^2 + y_1^2) - (x_2^2 + y_2^2)}{z_2^2 - z_1^2} $$


## Practical Estimation

In practice:

* Compute both $f^2$ from orthogonality and from norm equality
* If both are valid ($> 0$), take the average for robustness

If one estimate is negative or numerically unstable, use the valid one.


## Summary Notes

* Only a single parameter ($f$) is estimated
* Principal point $(c_x, c_y)$ must be known or reliably guessed
* Works well when the plane is large and viewed under perspective
* Use multiple homographies for improved calibration (e.g. Zhang's method)
