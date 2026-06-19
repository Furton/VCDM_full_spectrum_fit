# Genesis-VCDM curvature perturbation: numerics and analytic fit

Reference: arXiv:________ (to appear)

Solves the mode evolution `|R_k(x)|` for the Genesis-VCDM model over a parameter scan and
fits a closed-form approximation to it. Each mode has three parameters: `d`, `alpha0`
(= `tilde alpha_0`), `kappa` (dimensionless wavenumber). Time variable is `x = tau/tau_B`;
the frozen plateau sits at small `x`, horizon crossing at `kappa*x ~ 1`. The mode equation is

    u'' + (c_R^2 kappa^2 - z''/z) u = 0,    R_k = u / z.

## Files

- `genesis_background.py` -- background/perturbation quantities (`z^2`, `c_R^2`, `z''/z`,
  `omega2`, plateau amplitude `L0_logC`); imported by the notebooks.
- `vcdm_R_evolution.ipynb` -- runs the scan -> `data/vcdm_R_scan.npz`.
- `generate_features.ipynb` -- builds the feature cache -> `data/fit_curve_features_part*.npz`.
- `fit_curve_pytorch.ipynb` -- PyTorch fit -> `data/fit_curve_constants.json`, figures in `fig/`,
  and the final fitted formula (with its LaTeX form).
- `verify_plateau_consistency.ipynb` -- plateau-spectrum consistency checks.
- `data/`, `fig/` -- generated files and figures. `run.bat` launches Jupyter.

## Pipeline

**1. Scan** (`vcdm_R_evolution.ipynb`). Per mode: `choose_x_ini` picks a deep sub-horizon
start (adiabatic ratio `>= 1e5`), sets the Bunch-Davies initial state `u = 1/sqrt(2 omega)`,
and `solve_ivp` (DOP853, Radau fallback) integrates down to `x_final = 1e-6`, storing `|R_k|`
at 1500 points. 4096 random `(d, alpha0, kappa)`, seed 123, run in parallel with joblib.

**2a. Features** (`generate_features.ipynb`). Evaluates the physics-derived features per data
point and caches them to `data/fit_curve_features_part*.npz`.

**2b. Fit** (`fit_curve_pytorch.ipynb`). Eight linear constants in

    ln|R_k| = W*uni + (1-W)*de + w_B2*B2 + w_V*V

- `uni`: uniform-Airy (Langer) envelope through the turning point, divided by `z`.
- `de`: super-horizon branch, `L0 - F3 + p1 + p_d d + p_C logC + (f1+f1d d) F1 + f2 F2`.
- `W`: logistic blend in the Airy variable (`zeta_*, q_sw = 0.6, 9.0`, held fixed).
- `B2`, `V`: Gaussian bump and blend-window correction.

The model is linear in the 8 constants. Constants are solved with PyTorch L-BFGS:
a smooth minimax stage enforces the worst-case/deep-plateau trade-off, followed by
a capped weighted-mean polish.

**3. Verify** (`verify_plateau_consistency.ipynb`). Checks the plateau spectrum
`Ps = kappa^3 |R|^2 / (2 pi^2)` computed from the scan curves and from the analytic fit.

## Results

- Fit vs scan: per-curve mean 0.23%, worst 2.17% (at horizon crossing), deep plateau `<= 0.20%`;
  train and verify nearly identical.
- `z^2`, `c_R^2`, `z''/z` agree with a 50-digit reference to ~1e-14; the scan plateau is
  converged to ~3e-5 and independent of solver/tolerances.

The overall amplitude is a free normalization (fixed by `tau_B`); the spectra above are
dimensionless.



## Priors

- Parameter ranges: `d` in [0.1, 0.4], `alpha0` in [1e-30, 1e-21], `kappa` in [1e-9, 1e-4];
  outside this is untested.
