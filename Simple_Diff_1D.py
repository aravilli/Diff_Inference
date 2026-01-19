import numpy as np

class SimpleDiffusion1D:
    """Forward + reverse diffusion (DDPM posterior) on a tiny 1D vector using numpy only."""
    def __init__(self, T=200, beta_start=1e-4, beta_end=2e-2):
        self.T = T
        self.betas = np.linspace(beta_start, beta_end, T, dtype=np.float64)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = np.cumprod(self.alphas)

    def q_sample(self, x0, t, rng):
        """Forward: x_t = sqrt(ab_t) x0 + sqrt(1-ab_t) eps."""
        eps = rng.standard_normal(size=x0.shape)
        ab = self.alpha_bars[t]
        xt = np.sqrt(ab) * x0 + np.sqrt(1.0 - ab) * eps
        return xt, eps

    def eps_from_x0_xt(self, x0, xt, t):
        """Oracle eps implied by x0 and xt."""
        ab = self.alpha_bars[t]
        return (xt - np.sqrt(ab) * x0) / np.sqrt(max(1e-12, 1.0 - ab))

    def x0_from_eps(self, xt, t, eps_hat):
        ab = self.alpha_bars[t]
        return (xt - np.sqrt(1.0 - ab) * eps_hat) / np.sqrt(ab)

    def p_mean(self, xt, t, x0_hat):
        """Posterior mean of q(x_{t-1}|x_t,x0) used in DDPM sampling."""
        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        ab_t = self.alpha_bars[t]
        ab_prev = self.alpha_bars[t-1] if t > 0 else 1.0

        coef1 = np.sqrt(ab_prev) * beta_t / (1.0 - ab_t)
        coef2 = np.sqrt(alpha_t) * (1.0 - ab_prev) / (1.0 - ab_t)
        return coef1 * x0_hat + coef2 * xt

    def p_sample(self, xt, t, eps_hat, rng, add_noise=True):
        """Reverse: compute x0_hat from eps_hat, then sample x_{t-1}."""
        x0_hat = self.x0_from_eps(xt, t, eps_hat)
        mean = self.p_mean(xt, t, x0_hat)
        if t == 0:
            return mean
        if not add_noise:
            return mean
        # posterior variance (DDPM)
        ab_t = self.alpha_bars[t]
        ab_prev = self.alpha_bars[t-1]
        beta_t = self.betas[t]
        var = beta_t * (1.0 - ab_prev) / (1.0 - ab_t)
        z = rng.standard_normal(size=xt.shape)
        return mean + np.sqrt(var) * z


def demo(seed=7, T=200):
    rng = np.random.default_rng(seed)
    diff = SimpleDiffusion1D(T=T)

    x0 = np.array([1.0, -0.5, 0.8, -0.3], dtype=np.float64)

    # Forward diffusion snapshots
    ts = [0, 50, 100, 150, 199]
    forward = {}
    for t in ts:
        xt, eps = diff.q_sample(x0, t, rng)
        forward[t] = (xt, eps)

    # Start reverse from x_T taken from the forward sample at t=199
    xT = forward[199][0]

    # Reverse (oracle eps_hat derived from known x0 and current xt)
    xt = xT.copy()
    rev_err = []
    for t in range(T-1, -1, -1):
        eps_hat = diff.eps_from_x0_xt(x0, xt, t)  # oracle denoiser
        xt = diff.p_sample(xt, t, eps_hat, rng, add_noise=True)
        rev_err.append(np.linalg.norm(xt - x0))

    x0_recon = xt

    return x0, forward, xT, x0_recon, rev_err

x0, forward, xT, x0_recon, rev_err = demo(seed=42, T=200)

print("="*88)
print("FORWARD + REVERSE DIFFUSION (numpy-only, 1D vector)")
print("="*88)
print("x0 (clean):", x0)
print("\nForward diffusion snapshots: x_t = sqrt(α̅_t)x0 + sqrt(1-α̅_t)ε")
print("-"*88)
print(f"{'t':>5} | {'alpha_bar':>10} | {'||x_t||':>10} | x_t")
print("-"*88)

for t in [0, 50, 100, 150, 199]:
    xt, _ = forward[t]
    ab = np.cumprod(1.0 - np.linspace(1e-4, 2e-2, 200, dtype=np.float64))[t]
    print(f"{t:5d} | {ab:10.6f} | {np.linalg.norm(xt):10.4f} | {np.round(xt,4)}")

print("\nReverse diffusion starting from x_T (t=199):")
print("x_T:", np.round(xT, 4))
print("Reconstructed x0 (after reverse chain):", np.round(x0_recon, 6))
print("L2 reconstruction error:", float(np.linalg.norm(x0_recon - x0)))

print("\nReverse error trajectory (||x_t - x0||) at selected steps:")
# rev_err[0] corresponds to after step t=199 -> 198, rev_err[-1] after t=0
sel = [0, 9, 49, 99, 149, 199]
print("step_index (0=after first reverse step) -> error")
for i in sel:
    print(f"{i:3d} -> {rev_err[i]:.6f}")
