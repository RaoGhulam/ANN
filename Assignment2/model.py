import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# -----------------------------
# Utility: Constrained Linear Layer
# -----------------------------
class ConstrainedLinear(nn.Module):
    def __init__(self, in_features, out_features, non_negative=False):
        super().__init__()
        self.non_negative = non_negative

        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        if self.non_negative:
            weight = F.softplus(self.weight)  # ensures positivity
        else:
            weight = self.weight

        return F.linear(x, weight, self.bias)


# -----------------------------
# ISNN-1 Model
# -----------------------------
class ISNN1(nn.Module):
    def __init__(self,
                 x_dim,
                 y_dim,
                 z_dim,
                 t_dim,
                 hidden_size=10,
                 H=2):

        super().__init__()

        self.H = H

        # Activations
        self.sigma_mc = F.softplus
        self.sigma_m = torch.sigmoid
        self.sigma_a = torch.sigmoid

        # -----------------------------
        # Y branch (convex + monotone)
        # -----------------------------
        self.y_layers = nn.ModuleList()
        self.y_layers.append(
            ConstrainedLinear(y_dim, hidden_size, non_negative=True)
        )
        for _ in range(H - 1):
            self.y_layers.append(
                ConstrainedLinear(hidden_size, hidden_size, non_negative=True)
            )

        # -----------------------------
        # Z branch (arbitrary)
        # -----------------------------
        self.z_layers = nn.ModuleList()
        self.z_layers.append(
            ConstrainedLinear(z_dim, hidden_size, non_negative=False)
        )
        for _ in range(H - 1):
            self.z_layers.append(
                ConstrainedLinear(hidden_size, hidden_size, non_negative=False)
            )

        # -----------------------------
        # T branch (monotone)
        # -----------------------------
        self.t_layers = nn.ModuleList()
        self.t_layers.append(
            ConstrainedLinear(t_dim, hidden_size, non_negative=True)
        )
        for _ in range(H - 1):
            self.t_layers.append(
                ConstrainedLinear(hidden_size, hidden_size, non_negative=True)
            )

        # -----------------------------
        # X branch
        # -----------------------------
        # first layer weights from x, y, z, t
        self.x0_layer = ConstrainedLinear(x_dim, hidden_size, non_negative=True)

        self.xy = ConstrainedLinear(hidden_size, hidden_size, non_negative=True)
        self.xz = ConstrainedLinear(hidden_size, hidden_size, non_negative=False)
        self.xt = ConstrainedLinear(hidden_size, hidden_size, non_negative=True)

        # remaining x layers
        self.x_layers = nn.ModuleList()
        for _ in range(H - 1):
            self.x_layers.append(
                ConstrainedLinear(hidden_size, hidden_size, non_negative=True)
            )

    # -----------------------------
    # Forward pass
    # -----------------------------
    def forward(self, x0, y0, z0, t0):

        # ---- Y branch ----
        y = y0
        for i, layer in enumerate(self.y_layers):
            y = layer(y)
            y = self.sigma_mc(y)

        # ---- Z branch ----
        z = z0
        for i, layer in enumerate(self.z_layers):
            z = layer(z)
            z = self.sigma_a(z)

        # ---- T branch ----
        t = t0
        for i, layer in enumerate(self.t_layers):
            t = layer(t)
            t = self.sigma_m(t)

        # ---- X branch (first merged layer) ----
        x = self.x0_layer(x0)

        x = (
            x
            + self.xy(y)
            + self.xz(z)
            + self.xt(t)
        )

        x = self.sigma_mc(x)

        # ---- deeper X layers ----
        for layer in self.x_layers:
            x = layer(x)
            x = self.sigma_mc(x)

        # final output (NO activation)
        return x

# -------------------------------------------------
# ISNN-2 Model
# -------------------------------------------------
class ISNN2(nn.Module):
    def __init__(self,
                 x_dim,
                 y_dim,
                 z_dim,
                 t_dim,
                 hidden_size=15,
                 H=2):

        super().__init__()

        self.H = H

        # activations
        self.softplus = F.softplus
        self.sigmoid = torch.sigmoid

        # =================================================
        # Y branch (convex + monotone)
        # =================================================
        self.y_layers = nn.ModuleList()
        self.y_layers.append(
            ConstrainedLinear(y_dim, hidden_size, non_negative=True)
        )
        for _ in range(H - 1):
            self.y_layers.append(
                ConstrainedLinear(hidden_size, hidden_size, non_negative=True)
            )

        # =================================================
        # Z branch (arbitrary)
        # =================================================
        self.z_layers = nn.ModuleList()
        self.z_layers.append(
            ConstrainedLinear(z_dim, hidden_size, non_negative=False)
        )
        for _ in range(H - 1):
            self.z_layers.append(
                ConstrainedLinear(hidden_size, hidden_size, non_negative=False)
            )

        # =================================================
        # T branch (monotone)
        # =================================================
        self.t_layers = nn.ModuleList()
        self.t_layers.append(
            ConstrainedLinear(t_dim, hidden_size, non_negative=True)
        )
        for _ in range(H - 1):
            self.t_layers.append(
                ConstrainedLinear(hidden_size, hidden_size, non_negative=True)
            )

        # =================================================
        # X branch (continuous merging)
        # =================================================

        # Layer 1: x0, y0, z0, t0 merge
        self.x0_layer = ConstrainedLinear(x_dim, hidden_size, non_negative=False)

        self.y0_layer = ConstrainedLinear(y_dim, hidden_size, non_negative=True)
        self.z0_layer = ConstrainedLinear(z_dim, hidden_size, non_negative=False)
        self.t0_layer = ConstrainedLinear(t_dim, hidden_size, non_negative=True)

        # Hidden layers (continuous merging)
        self.x_layers = nn.ModuleList()
        self.xy_layers = nn.ModuleList()
        self.xz_layers = nn.ModuleList()
        self.xt_layers = nn.ModuleList()
        self.x0_skip_layers = nn.ModuleList()

        for _ in range(H - 1):

            # main recurrent x path
            self.x_layers.append(
                ConstrainedLinear(hidden_size, hidden_size, non_negative=True)
            )

            # skip connection from x0 (unconstrained)
            self.x0_skip_layers.append(
                ConstrainedLinear(x_dim, hidden_size, non_negative=False)
            )

            # y influence (non-negative)
            self.xy_layers.append(
                ConstrainedLinear(hidden_size, hidden_size, non_negative=True)
            )

            # z influence (free)
            self.xz_layers.append(
                ConstrainedLinear(hidden_size, hidden_size, non_negative=False)
            )

            # t influence (non-negative)
            self.xt_layers.append(
                ConstrainedLinear(hidden_size, hidden_size, non_negative=True)
            )

        # final layer (no activation)
        self.final_layer = ConstrainedLinear(hidden_size, hidden_size, non_negative=False)

    # -------------------------------------------------
    # forward pass
    # -------------------------------------------------
    def forward(self, x0, y0, z0, t0):

        # =========================
        # Y branch
        # =========================
        y = y0
        for layer in self.y_layers:
            y = self.softplus(layer(y))

        # =========================
        # Z branch
        # =========================
        z = z0
        for layer in self.z_layers:
            z = self.sigmoid(layer(z))

        # =========================
        # T branch
        # =========================
        t = t0
        for layer in self.t_layers:
            t = self.sigmoid(layer(t))

        # =========================
        # X branch - Layer 1
        # =========================
        x = (
            self.x0_layer(x0)
            + self.y0_layer(y0)
            + self.z0_layer(z0)
            + self.t0_layer(t0)
        )

        x = self.softplus(x)

        # =========================
        # X branch - continuous merging
        # =========================
        for i in range(self.H - 1):

            x = (
                self.x_layers[i](x)
                + self.x0_skip_layers[i](x0)
                + self.xy_layers[i](y)
                + self.xz_layers[i](z)
                + self.xt_layers[i](t)
            )

            x = self.softplus(x)

        # =========================
        # final output (NO activation)
        # =========================
        x = self.final_layer(x)

        return x

# ---------------------------------------------------------------------------
# Numpy activation functions and their derivatives
# ---------------------------------------------------------------------------

def _softplus(x):
    """log(1 + exp(x)), numerically stable."""
    return np.where(x > 30, x, np.log1p(np.exp(np.clip(x, -500, 30))))


def _softplus_grad(x):
    """d/dx softplus(x) = sigmoid(x)."""
    return _sigmoid(x)


def _sigmoid(x):
    """1 / (1 + exp(-x)), numerically stable."""
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-np.clip(x, -500, 500))),
                    np.exp(np.clip(x, -500, 500)) / (1.0 + np.exp(np.clip(x, -500, 500))))


def _sigmoid_grad(x):
    """d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))."""
    s = _sigmoid(x)
    return s * (1.0 - s)


# ---------------------------------------------------------------------------
# Numpy linear-layer helpers
# ---------------------------------------------------------------------------

def _linear_forward(x, raw_W, b, non_negative):
    W_eff = _softplus(raw_W) if non_negative else raw_W
    return x @ W_eff.T + b, W_eff


def _linear_backward(d_out, x, raw_W, W_eff, non_negative):
    batch = x.shape[0]
    d_b    = d_out.sum(axis=0)                       # (out,)
    d_W_eff = d_out.T @ x                             # (out, in)
    d_x    = d_out @ W_eff                             # (batch, in)

    if non_negative:
        d_raw_W = d_W_eff * _sigmoid(raw_W)           # softplus' = sigmoid
    else:
        d_raw_W = d_W_eff

    return d_x, d_raw_W, d_b


# ---------------------------------------------------------------------------
# Adam state helpers
# ---------------------------------------------------------------------------

def _adam_init(params):
    """Return zeroed first/second moment dicts keyed by param id."""
    m = {id(p): np.zeros_like(p) for p in params}
    v = {id(p): np.zeros_like(p) for p in params}
    return m, v


def _adam_update(params, grads, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """In-place Adam update. t is the current step (1-indexed)."""
    for p, g in zip(params, grads):
        key = id(p)
        m[key] = beta1 * m[key] + (1 - beta1) * g
        v[key] = beta2 * v[key] + (1 - beta2) * g ** 2
        m_hat = m[key] / (1 - beta1 ** t)
        v_hat = v[key] / (1 - beta2 ** t)
        p -= lr * m_hat / (np.sqrt(v_hat) + eps)


# ===========================================================================
#  ISNN-1  –  numpy manual backprop
# ===========================================================================

class ISNN1Numpy:
    def __init__(self, x_dim=1, y_dim=1, z_dim=1, t_dim=1,
                 hidden_size=10, H=2, lr=1e-3):

        self.H           = H
        self.hidden_size = hidden_size
        self.lr          = lr
        self._step       = 0          # Adam step counter

        rng = np.random.default_rng(42)
        scale = 0.1

        def W(out, inp):  return rng.standard_normal((out, inp))  * scale
        def b(out):        return np.zeros(out)

        # ── Y branch: H layers, non_negative=True ──────────────────────────
        self.Wy  = [W(hidden_size, y_dim)] + [W(hidden_size, hidden_size) for _ in range(H-1)]
        self.by  = [b(hidden_size) for _ in range(H)]   # each layer its own array
        self.nn_y = [True] * H

        # ── Z branch: H layers, non_negative=False ─────────────────────────
        self.Wz  = [W(hidden_size, z_dim)] + [W(hidden_size, hidden_size) for _ in range(H-1)]
        self.bz  = [b(hidden_size) for _ in range(H)]
        self.nn_z = [False] * H

        # ── T branch: H layers, non_negative=True ──────────────────────────
        self.Wt  = [W(hidden_size, t_dim)] + [W(hidden_size, hidden_size) for _ in range(H-1)]
        self.bt  = [b(hidden_size) for _ in range(H)]
        self.nn_t = [True] * H

        # ── X branch layer-0 cross weights ─────────────────────────────────
        # W_x0: free  (convexity only requires h>=1 to be non-neg)
        self.Wx0  = W(hidden_size, x_dim)   ; self.bx0  = b(hidden_size)
        # W_xy: non-negative  (monotone + convex in y0)
        self.Wxy  = W(hidden_size, hidden_size)
        # W_xz: free  (arbitrary in z0)
        self.Wxz  = W(hidden_size, hidden_size)
        # W_xt: non-negative  (monotone in t0)
        self.Wxt  = W(hidden_size, hidden_size)

        # ── X branch deeper layers: H-1 layers, non_negative=True ──────────
        self.Wx  = [W(hidden_size, hidden_size) for _ in range(H-1)]
        self.bx  = [b(hidden_size) for _ in range(H-1)]

        # ── Final output layer: hidden → 1, free weights, no activation ────
        self.Wout = W(1, hidden_size)
        self.bout = b(1)

        # ── Collect all parameters for Adam ────────────────────────────────
        self._params = self._all_params()
        self._m, self._v = _adam_init(self._params)

    # -----------------------------------------------------------------------
    # Parameter list (preserves insertion order for grad alignment)
    # -----------------------------------------------------------------------
    def _all_params(self):
        ps = []
        for i in range(self.H):
            ps += [self.Wy[i], self.by[i]]
        for i in range(self.H):
            ps += [self.Wz[i], self.bz[i]]
        for i in range(self.H):
            ps += [self.Wt[i], self.bt[i]]
        ps += [self.Wx0, self.bx0, self.Wxy, self.Wxz, self.Wxt]
        for i in range(self.H - 1):
            ps += [self.Wx[i], self.bx[i]]
        ps += [self.Wout, self.bout]
        return ps

    # -----------------------------------------------------------------------
    # Forward pass – caches all intermediates needed for backprop
    # -----------------------------------------------------------------------
    def forward(self, x0, y0, z0, t0):
        cache = {}

        # ── Y branch ────────────────────────────────────────────────────────
        y = y0
        cache['y_pre']  = []   # pre-activation values
        cache['y_post'] = []   # post-activation values (= input to next layer)
        cache['y_in']   = []   # input to each linear layer
        cache['Wy_eff'] = []
        for i in range(self.H):
            cache['y_in'].append(y)
            pre, W_eff = _linear_forward(y, self.Wy[i], self.by[i], self.nn_y[i])
            y = _softplus(pre)
            cache['y_pre'].append(pre)
            cache['y_post'].append(y)
            cache['Wy_eff'].append(W_eff)
        y_out = y   # final Y branch output

        # ── Z branch ────────────────────────────────────────────────────────
        z = z0
        cache['z_pre']  = []
        cache['z_post'] = []
        cache['z_in']   = []
        cache['Wz_eff'] = []
        for i in range(self.H):
            cache['z_in'].append(z)
            pre, W_eff = _linear_forward(z, self.Wz[i], self.bz[i], self.nn_z[i])
            z = _sigmoid(pre)
            cache['z_pre'].append(pre)
            cache['z_post'].append(z)
            cache['Wz_eff'].append(W_eff)
        z_out = z

        # ── T branch ────────────────────────────────────────────────────────
        t = t0
        cache['t_pre']  = []
        cache['t_post'] = []
        cache['t_in']   = []
        cache['Wt_eff'] = []
        for i in range(self.H):
            cache['t_in'].append(t)
            pre, W_eff = _linear_forward(t, self.Wt[i], self.bt[i], self.nn_t[i])
            t = _sigmoid(pre)
            cache['t_pre'].append(pre)
            cache['t_post'].append(t)
            cache['Wt_eff'].append(W_eff)
        t_out = t

        # ── X branch layer-0 (merge) ────────────────────────────────────────
        # out = softplus( x0@Wx0.T + bx0  +  y@Wxy.T  +  z@Wxz.T  +  t@Wxt.T )
        # Only Wx0 carries a learnable bias (bx0); Wxy/Wxz/Wxt have no bias.
        _zeros = np.zeros(self.hidden_size)
        lin_x0,  Wx0_eff  = _linear_forward(x0,    self.Wx0,  self.bx0, False)
        lin_xy,  Wxy_eff  = _linear_forward(y_out, self.Wxy,  _zeros,   True)
        lin_xz,  Wxz_eff  = _linear_forward(z_out, self.Wxz,  _zeros,   False)
        lin_xt,  Wxt_eff  = _linear_forward(t_out, self.Wxt,  _zeros,   True)

        x1_pre = lin_x0 + lin_xy + lin_xz + lin_xt    # bias already in lin_x0
        x1     = _softplus(x1_pre)

        cache.update({
            'x0': x0, 'y_out': y_out, 'z_out': z_out, 't_out': t_out,
            'lin_x0': lin_x0, 'lin_xy': lin_xy, 'lin_xz': lin_xz, 'lin_xt': lin_xt,
            'x1_pre': x1_pre,
            'Wx0_eff': Wx0_eff, 'Wxy_eff': Wxy_eff, 'Wxz_eff': Wxz_eff, 'Wxt_eff': Wxt_eff,
        })

        # ── X branch deeper layers ───────────────────────────────────────────
        x = x1
        cache['xd_in']   = []   # input to each deeper layer
        cache['xd_pre']  = []   # pre-activation
        cache['Wxd_eff'] = []
        for i in range(self.H - 1):
            cache['xd_in'].append(x)
            pre, W_eff = _linear_forward(x, self.Wx[i], self.bx[i], True)
            x = _softplus(pre)
            cache['xd_pre'].append(pre)
            cache['Wxd_eff'].append(W_eff)
        cache['x_before_out'] = x

        # ── Final output layer (no activation) ──────────────────────────────
        out, Wout_eff = _linear_forward(x, self.Wout, self.bout, False)
        cache['Wout_eff'] = Wout_eff

        return out, cache

    # -----------------------------------------------------------------------
    # Backward pass – full manual backpropagation
    # -----------------------------------------------------------------------
    def backward(self, d_out, cache):
        """
        d_out : upstream gradient of shape (batch, 1)  (= dL/d_output)
        Returns list of gradients in the same order as _all_params().
        """
        grads = {}

        # ── Output layer ────────────────────────────────────────────────────
        x_bo = cache['x_before_out']
        d_x, d_Wout, d_bout = _linear_backward(d_out, x_bo,
                                                self.Wout, cache['Wout_eff'], False)
        grads['Wout'] = d_Wout
        grads['bout'] = d_bout

        # ── X deeper layers (backward) ──────────────────────────────────────
        d_Wx  = [None] * (self.H - 1)
        d_bx  = [None] * (self.H - 1)
        for i in reversed(range(self.H - 1)):
            d_x = d_x * _softplus_grad(cache['xd_pre'][i])
            inp = cache['xd_in'][i]
            d_x, d_Wx[i], d_bx[i] = _linear_backward(
                d_x, inp, self.Wx[i], cache['Wxd_eff'][i], True)

        # ── X branch layer-0 (merge) ─────────────────────────────────────────
        # The pre-activation x1_pre = lin_x0 + lin_xy + lin_xz + lin_xt,
        # so d_pre flows equally into every branch's linear call.
        d_x1 = d_x * _softplus_grad(cache['x1_pre'])

        # Wx0, bx0 (free weights, learnable bias)
        d_x0_from_merge, d_Wx0, d_bx0 = _linear_backward(
            d_x1, cache['x0'], self.Wx0, cache['Wx0_eff'], False)

        # Wxy (non-negative, NO learnable bias – zero vector only)
        d_y_from_merge, d_Wxy, _ = _linear_backward(
            d_x1, cache['y_out'], self.Wxy, cache['Wxy_eff'], True)

        # Wxz (free, NO learnable bias)
        d_z_from_merge, d_Wxz, _ = _linear_backward(
            d_x1, cache['z_out'], self.Wxz, cache['Wxz_eff'], False)

        # Wxt (non-negative, NO learnable bias)
        d_t_from_merge, d_Wxt, _ = _linear_backward(
            d_x1, cache['t_out'], self.Wxt, cache['Wxt_eff'], True)

        grads.update({'Wx0': d_Wx0, 'bx0': d_bx0,
                      'Wxy': d_Wxy, 'Wxz': d_Wxz, 'Wxt': d_Wxt})

        # ── Y branch (backward) ──────────────────────────────────────────────
        d_y = d_y_from_merge
        d_Wy = [None] * self.H
        d_by = [None] * self.H
        for i in reversed(range(self.H)):
            d_y = d_y * _softplus_grad(cache['y_pre'][i])
            d_y, d_Wy[i], d_by[i] = _linear_backward(
                d_y, cache['y_in'][i], self.Wy[i], cache['Wy_eff'][i], self.nn_y[i])

        # ── Z branch (backward) ──────────────────────────────────────────────
        d_z = d_z_from_merge
        d_Wz = [None] * self.H
        d_bz = [None] * self.H
        for i in reversed(range(self.H)):
            d_z = d_z * _sigmoid_grad(cache['z_pre'][i])
            d_z, d_Wz[i], d_bz[i] = _linear_backward(
                d_z, cache['z_in'][i], self.Wz[i], cache['Wz_eff'][i], self.nn_z[i])

        # ── T branch (backward) ──────────────────────────────────────────────
        d_t = d_t_from_merge
        d_Wt = [None] * self.H
        d_bt = [None] * self.H
        for i in reversed(range(self.H)):
            d_t = d_t * _sigmoid_grad(cache['t_pre'][i])
            d_t, d_Wt[i], d_bt[i] = _linear_backward(
                d_t, cache['t_in'][i], self.Wt[i], cache['Wt_eff'][i], self.nn_t[i])

        # ── Assemble gradient list matching _all_params() order ──────────────
        grad_list = []
        for i in range(self.H):
            grad_list += [d_Wy[i], d_by[i]]
        for i in range(self.H):
            grad_list += [d_Wz[i], d_bz[i]]
        for i in range(self.H):
            grad_list += [d_Wt[i], d_bt[i]]
        grad_list += [d_Wx0, d_bx0, d_Wxy, d_Wxz, d_Wxt]
        for i in range(self.H - 1):
            grad_list += [d_Wx[i], d_bx[i]]
        grad_list += [d_Wout, d_bout]

        return grad_list

    # -----------------------------------------------------------------------
    # Single training step: forward → MSE loss → backward → Adam update
    # -----------------------------------------------------------------------
    def step(self, x0, y0, z0, t0, y_true):
        """
        x0, y0, z0, t0, y_true : numpy arrays of shape (batch, dim).
        Returns scalar MSE loss.
        """
        self._step += 1

        out, cache = self.forward(x0, y0, z0, t0)

        batch = out.shape[0]
        diff  = out - y_true                    # (batch, 1)
        loss  = np.mean(diff ** 2)

        d_out = (2.0 / batch) * diff            # dMSE/d_out
        grads = self.backward(d_out, cache)

        _adam_update(self._params, grads, self._m, self._v,
                     self._step, lr=self.lr)

        return float(loss)

    # -----------------------------------------------------------------------
    # Convenience: predict without storing gradients
    # -----------------------------------------------------------------------
    def predict(self, x0, y0, z0, t0):
        out, _ = self.forward(x0, y0, z0, t0)
        return out

    def count_parameters(self):
        return sum(p.size for p in self._params)


# ===========================================================================
#  ISNN-2  –  numpy manual backprop
# ===========================================================================

class ISNN2Numpy:
    def __init__(self, x_dim=1, y_dim=1, z_dim=1, t_dim=1,
                 hidden_size=15, H=2, lr=1e-3):

        self.H           = H
        self.hidden_size = hidden_size
        self.x_dim       = x_dim
        self.lr          = lr
        self._step       = 0

        rng   = np.random.default_rng(0)
        scale = 0.1

        def W(out, inp): return rng.standard_normal((out, inp)) * scale
        def b(out):       return np.zeros(out)

        # ── Y branch: H-1 layers, non_negative=True ────────────────────────
        #   (runs from y0; if H=1 this branch has 0 layers, y_out = y0)
        n_side = max(H - 1, 0)
        self.Wy   = [W(hidden_size, y_dim)] + [W(hidden_size, hidden_size) for _ in range(n_side - 1)] if n_side > 0 else []
        self.by   = [b(hidden_size)] * n_side
        self.nn_y = [True] * n_side

        # ── Z branch: H-1 layers, non_negative=False ───────────────────────
        self.Wz   = [W(hidden_size, z_dim)] + [W(hidden_size, hidden_size) for _ in range(n_side - 1)] if n_side > 0 else []
        self.bz   = [b(hidden_size)] * n_side
        self.nn_z = [False] * n_side

        # ── T branch: H-1 layers, non_negative=True ────────────────────────
        self.Wt   = [W(hidden_size, t_dim)] + [W(hidden_size, hidden_size) for _ in range(n_side - 1)] if n_side > 0 else []
        self.bt   = [b(hidden_size)] * n_side
        self.nn_t = [True] * n_side

        # ── X branch layer-0: merges raw x0,y0,z0,t0 (Eq. 9) ──────────────
        self.Wx0_l0  = W(hidden_size, x_dim)  ; self.bx0_l0 = b(hidden_size)
        self.Wy0_l0  = W(hidden_size, y_dim)  # non-negative
        self.Wz0_l0  = W(hidden_size, z_dim)  # free
        self.Wt0_l0  = W(hidden_size, t_dim)  # non-negative

        # ── X branch hidden layers h=1…H-1 (Eq. 10) ────────────────────────
        #   Each layer: x_h = σ_mc( x_{h-1}@W_xx.T + x0@W_xx0.T + y@W_xy.T + z@W_xz.T + t@W_xt.T + b )
        self.Wxx   = [W(hidden_size, hidden_size) for _ in range(H - 1)]  # non-negative
        self.Wxx0  = [W(hidden_size, x_dim)       for _ in range(H - 1)]  # free (skip from x0)
        self.Wxy_h = [W(hidden_size, hidden_size) for _ in range(H - 1)]  # non-negative
        self.Wxz_h = [W(hidden_size, hidden_size) for _ in range(H - 1)]  # free
        self.Wxt_h = [W(hidden_size, hidden_size) for _ in range(H - 1)]  # non-negative
        self.bx_h  = [b(hidden_size)               for _ in range(H - 1)]

        # ── Final output layer: hidden → 1, free, no activation ─────────────
        self.Wout = W(1, hidden_size)
        self.bout = b(1)

        self._params = self._all_params()
        self._m, self._v = _adam_init(self._params)

    # -----------------------------------------------------------------------
    def _all_params(self):
        ps = []
        for i in range(len(self.Wy)):
            ps += [self.Wy[i], self.by[i]]
        for i in range(len(self.Wz)):
            ps += [self.Wz[i], self.bz[i]]
        for i in range(len(self.Wt)):
            ps += [self.Wt[i], self.bt[i]]
        ps += [self.Wx0_l0, self.bx0_l0,
               self.Wy0_l0, self.Wz0_l0, self.Wt0_l0]
        for i in range(self.H - 1):
            ps += [self.Wxx[i], self.Wxx0[i],
                   self.Wxy_h[i], self.Wxz_h[i], self.Wxt_h[i],
                   self.bx_h[i]]
        ps += [self.Wout, self.bout]
        return ps

    # -----------------------------------------------------------------------
    def forward(self, x0, y0, z0, t0):
        cache = {'x0': x0, 'y0': y0, 'z0': z0, 't0': t0}
        n_side = len(self.Wy)

        # ── Y branch ────────────────────────────────────────────────────────
        y = y0
        cache['y_in'] = []; cache['y_pre'] = []; cache['Wy_eff'] = []
        for i in range(n_side):
            cache['y_in'].append(y)
            pre, W_eff = _linear_forward(y, self.Wy[i], self.by[i], self.nn_y[i])
            y = _softplus(pre)
            cache['y_pre'].append(pre); cache['Wy_eff'].append(W_eff)
        y_out = y
        cache['y_out'] = y_out

        # ── Z branch ────────────────────────────────────────────────────────
        z = z0
        cache['z_in'] = []; cache['z_pre'] = []; cache['Wz_eff'] = []
        for i in range(n_side):
            cache['z_in'].append(z)
            pre, W_eff = _linear_forward(z, self.Wz[i], self.bz[i], self.nn_z[i])
            z = _sigmoid(pre)
            cache['z_pre'].append(pre); cache['Wz_eff'].append(W_eff)
        z_out = z
        cache['z_out'] = z_out

        # ── T branch ────────────────────────────────────────────────────────
        t = t0
        cache['t_in'] = []; cache['t_pre'] = []; cache['Wt_eff'] = []
        for i in range(n_side):
            cache['t_in'].append(t)
            pre, W_eff = _linear_forward(t, self.Wt[i], self.bt[i], self.nn_t[i])
            t = _sigmoid(pre)
            cache['t_pre'].append(pre); cache['Wt_eff'].append(W_eff)
        t_out = t
        cache['t_out'] = t_out

        # ── X layer-0: merge raw inputs ──────────────────────────────────────
        lin_x0, Wx0_l0_eff = _linear_forward(x0, self.Wx0_l0, self.bx0_l0, False)
        lin_y0, Wy0_l0_eff = _linear_forward(y0, self.Wy0_l0, np.zeros(self.hidden_size), True)
        lin_z0, Wz0_l0_eff = _linear_forward(z0, self.Wz0_l0, np.zeros(self.hidden_size), False)
        lin_t0, Wt0_l0_eff = _linear_forward(t0, self.Wt0_l0, np.zeros(self.hidden_size), True)
        x1_pre = lin_x0 + lin_y0 + lin_z0 + lin_t0
        x1     = _softplus(x1_pre)

        cache.update({
            'x1_pre': x1_pre,
            'lin_x0_l0': lin_x0, 'lin_y0_l0': lin_y0,
            'lin_z0_l0': lin_z0, 'lin_t0_l0': lin_t0,
            'Wx0_l0_eff': Wx0_l0_eff, 'Wy0_l0_eff': Wy0_l0_eff,
            'Wz0_l0_eff': Wz0_l0_eff, 'Wt0_l0_eff': Wt0_l0_eff,
        })

        # ── X hidden layers h=1…H-1 ──────────────────────────────────────────
        x = x1
        cache['xh_in']    = []   # x_{h-1}
        cache['xh_pre']   = []   # pre-activation of layer h
        cache['Wxx_eff']  = []
        cache['Wxx0_eff'] = []
        cache['Wxyh_eff'] = []
        cache['Wxzh_eff'] = []
        cache['Wxth_eff'] = []

        for i in range(self.H - 1):
            cache['xh_in'].append(x)
            l_xx,  Wxx_eff  = _linear_forward(x,     self.Wxx[i],   self.bx_h[i], True)
            l_xx0, Wxx0_eff = _linear_forward(x0,    self.Wxx0[i],  np.zeros(self.hidden_size), False)
            l_xy,  Wxyh_eff = _linear_forward(y_out, self.Wxy_h[i], np.zeros(self.hidden_size), True)
            l_xz,  Wxzh_eff = _linear_forward(z_out, self.Wxz_h[i], np.zeros(self.hidden_size), False)
            l_xt,  Wxth_eff = _linear_forward(t_out, self.Wxt_h[i], np.zeros(self.hidden_size), True)
            pre = l_xx + l_xx0 + l_xy + l_xz + l_xt
            x   = _softplus(pre)
            cache['xh_pre'].append(pre)
            cache['Wxx_eff'].append(Wxx_eff)
            cache['Wxx0_eff'].append(Wxx0_eff)
            cache['Wxyh_eff'].append(Wxyh_eff)
            cache['Wxzh_eff'].append(Wxzh_eff)
            cache['Wxth_eff'].append(Wxth_eff)

        cache['x_before_out'] = x

        # ── Output layer ─────────────────────────────────────────────────────
        out, Wout_eff = _linear_forward(x, self.Wout, self.bout, False)
        cache['Wout_eff'] = Wout_eff

        return out, cache

    # -----------------------------------------------------------------------
    def backward(self, d_out, cache):
        n_side = len(self.Wy)

        # Accumulators for gradients that receive contributions from multiple layers
        d_y_out_acc = np.zeros_like(cache['y_out'])
        d_z_out_acc = np.zeros_like(cache['z_out'])
        d_t_out_acc = np.zeros_like(cache['t_out'])
        d_x0_acc    = np.zeros_like(cache['x0'])

        # ── Output layer ────────────────────────────────────────────────────
        d_x, d_Wout, d_bout = _linear_backward(
            d_out, cache['x_before_out'], self.Wout, cache['Wout_eff'], False)

        # ── X hidden layers (backward, reversed) ────────────────────────────
        d_Wxx  = [None] * (self.H - 1)
        d_Wxx0 = [None] * (self.H - 1)
        d_Wxyh = [None] * (self.H - 1)
        d_Wxzh = [None] * (self.H - 1)
        d_Wxth = [None] * (self.H - 1)
        d_bx_h = [None] * (self.H - 1)

        for i in reversed(range(self.H - 1)):
            d_pre = d_x * _softplus_grad(cache['xh_pre'][i])

            # W_xx (non-neg, carries bias)
            d_x, d_Wxx[i], d_bx_h[i] = _linear_backward(
                d_pre, cache['xh_in'][i], self.Wxx[i], cache['Wxx_eff'][i], True)

            # W_xx0 skip (free, no separate bias)
            d_x0_from_skip, d_Wxx0[i], _ = _linear_backward(
                d_pre, cache['x0'], self.Wxx0[i], cache['Wxx0_eff'][i], False)
            d_x0_acc += d_x0_from_skip

            # W_xy_h (non-neg)
            d_y_from_h, d_Wxyh[i], _ = _linear_backward(
                d_pre, cache['y_out'], self.Wxy_h[i], cache['Wxyh_eff'][i], True)
            d_y_out_acc += d_y_from_h

            # W_xz_h (free)
            d_z_from_h, d_Wxzh[i], _ = _linear_backward(
                d_pre, cache['z_out'], self.Wxz_h[i], cache['Wxzh_eff'][i], False)
            d_z_out_acc += d_z_from_h

            # W_xt_h (non-neg)
            d_t_from_h, d_Wxth[i], _ = _linear_backward(
                d_pre, cache['t_out'], self.Wxt_h[i], cache['Wxth_eff'][i], True)
            d_t_out_acc += d_t_from_h

        # ── X layer-0 backward ───────────────────────────────────────────────
        d_x1 = d_x * _softplus_grad(cache['x1_pre'])

        d_x0_from_l0, d_Wx0_l0, d_bx0_l0 = _linear_backward(
            d_x1, cache['x0'], self.Wx0_l0, cache['Wx0_l0_eff'], False)
        d_x0_acc += d_x0_from_l0

        d_y0_from_l0, d_Wy0_l0, _ = _linear_backward(
            d_x1, cache['y0'], self.Wy0_l0, cache['Wy0_l0_eff'], True)

        d_z0_from_l0, d_Wz0_l0, _ = _linear_backward(
            d_x1, cache['z0'], self.Wz0_l0, cache['Wz0_l0_eff'], False)

        d_t0_from_l0, d_Wt0_l0, _ = _linear_backward(
            d_x1, cache['t0'], self.Wt0_l0, cache['Wt0_l0_eff'], True)

        # Note: y0_layer, z0_layer, t0_layer in Eq.9 take raw inputs,
        # so their gradients go directly to y0/z0/t0 (not back through branches).
        # d_y0_from_l0, d_z0_from_l0, d_t0_from_l0 propagate to the raw inputs,
        # not accumulated into d_y_out_acc (which is the Y-branch output gradient).

        # ── Y branch backward ────────────────────────────────────────────────
        d_Wy = [None] * n_side
        d_by = [None] * n_side
        d_y  = d_y_out_acc
        for i in reversed(range(n_side)):
            y_in_i = cache['y_in'][i]
            d_y = d_y * _softplus_grad(cache['y_pre'][i])
            d_y, d_Wy[i], d_by[i] = _linear_backward(
                d_y, y_in_i, self.Wy[i], cache['Wy_eff'][i], self.nn_y[i])

        # ── Z branch backward ────────────────────────────────────────────────
        d_Wz = [None] * n_side
        d_bz = [None] * n_side
        d_z  = d_z_out_acc
        for i in reversed(range(n_side)):
            d_z = d_z * _sigmoid_grad(cache['z_pre'][i])
            d_z, d_Wz[i], d_bz[i] = _linear_backward(
                d_z, cache['z_in'][i], self.Wz[i], cache['Wz_eff'][i], self.nn_z[i])

        # ── T branch backward ────────────────────────────────────────────────
        d_Wt = [None] * n_side
        d_bt = [None] * n_side
        d_t  = d_t_out_acc
        for i in reversed(range(n_side)):
            d_t = d_t * _sigmoid_grad(cache['t_pre'][i])
            d_t, d_Wt[i], d_bt[i] = _linear_backward(
                d_t, cache['t_in'][i], self.Wt[i], cache['Wt_eff'][i], self.nn_t[i])

        # ── Assemble gradient list matching _all_params() order ──────────────
        grad_list = []
        for i in range(n_side):
            grad_list += [d_Wy[i], d_by[i]]
        for i in range(n_side):
            grad_list += [d_Wz[i], d_bz[i]]
        for i in range(n_side):
            grad_list += [d_Wt[i], d_bt[i]]
        grad_list += [d_Wx0_l0, d_bx0_l0, d_Wy0_l0, d_Wz0_l0, d_Wt0_l0]
        for i in range(self.H - 1):
            grad_list += [d_Wxx[i], d_Wxx0[i],
                          d_Wxyh[i], d_Wxzh[i], d_Wxth[i],
                          d_bx_h[i]]
        grad_list += [d_Wout, d_bout]

        return grad_list

    # -----------------------------------------------------------------------
    def step(self, x0, y0, z0, t0, y_true):
        self._step += 1
        out, cache = self.forward(x0, y0, z0, t0)
        batch  = out.shape[0]
        diff   = out - y_true
        loss   = np.mean(diff ** 2)
        d_out  = (2.0 / batch) * diff
        grads  = self.backward(d_out, cache)
        _adam_update(self._params, grads, self._m, self._v,
                     self._step, lr=self.lr)
        return float(loss)

    def predict(self, x0, y0, z0, t0):
        out, _ = self.forward(x0, y0, z0, t0)
        return out

    def count_parameters(self):
        return sum(p.size for p in self._params)