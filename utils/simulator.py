import random
import numpy as np
import sdeint
import igraph as ig
from scipy.integrate import odeint
from scipy.special import expit as sigmoid

def decade(n_v, n_z, weight_range):
    B = np.zeros((n_z, n_v))
    # have_v = set(range(n_v))
    # while len(have_v) >= n_z:
    #     select_v = set(random.sample(have_v, n_z))
    #     have_v = have_v - select_v
    #     B[range(n_z), list(select_v)] = 1
    # if len(have_v) > 0:
    #     for j in have_v:
    #         i = random.randint(0, n_z-1)
    #         B[i, j] = 1
    # have_v = random.sample(range(n_v), n_z)
    # B[range(n_z), have_v] = 1
    # for j in range(n_v):
    #     if j not in have_v:
    #         i = random.randint(0, n_z - 1)
    #         B[i, j] = 1
    have_z = set()
    for i in range(n_v):
        j = random.randint(0, n_z - 1)
        have_z.add(j)
        B[j, i] = 1
    for j in range(n_z):
        if j not in have_z:
            i = random.randint(0, n_v - 1)
            for k in range(n_z):
                if B[k, i] == 1:
                    for l in range(n_v):
                        B[j, l] = B[k, l]
    U = np.random.uniform(low=weight_range[0], high=weight_range[1], size=[n_z, n_v])
    U[np.random.rand(n_z, n_v) < 0.5] *= -1
    W = (B != 0).astype(float) * U
    return B, W

def make_var_stationary(beta, radius=0.97):
    """Rescale coefficients of VAR model to make stable."""
    p = beta.shape[0]
    lag = beta.shape[1] // p
    bottom = np.hstack((np.eye(p * (lag - 1)), np.zeros((p * (lag - 1), p))))
    beta_tilde = np.vstack((beta, bottom))
    eigvals = np.linalg.eigvals(beta_tilde)
    max_eig = max(np.abs(eigvals))
    nonstationary = max_eig > radius
    if nonstationary:
        return make_var_stationary((beta / max_eig) * 0.7, radius)
    else:
        return beta


def simulate_var(p, T, lag, sparsity=0.2, beta_value=1.0, sd=0.1, seed=0):
    if seed is not None:
        np.random.seed(seed)

    # Set up coefficients and Granger causality ground truth.
    GC = np.eye(p, dtype=int)
    beta = np.eye(p) * beta_value

    num_nonzero = int(p * sparsity) - 1
    for i in range(p):
        choice = np.random.choice(p - 1, size=num_nonzero, replace=False)
        choice[choice >= i] += 1
        beta[i, choice] = beta_value
        GC[i, choice] = 1

    beta = np.hstack([beta for _ in range(lag)])
    beta = make_var_stationary(beta)

    # Generate data.
    burn_in = 100
    errors = np.random.normal(scale=sd, size=(p, T + burn_in))
    X = np.zeros((p, T + burn_in))
    X[:, :lag] = errors[:, :lag]
    for t in range(lag, T + burn_in):
        X[:, t] = np.dot(beta, X[:, (t - lag) : t].flatten(order="F"))
        X[:, t] += +errors[:, t - 1]

    return X.T[burn_in:], beta.T, GC.T

def simulate_dag(p, e, graph_type, lags, ins, es):

    def _random_permutation(M):
        # np.random.permutation permutes first axis only
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    def _random_acyclic_orientation(B_und):
        return np.tril(_random_permutation(B_und), k=-1)

    def _graph_to_adjmat(G):
        return np.array(G.get_adjacency().data)

    d_total = p * (lags + 1)
    B_time = np.zeros((d_total, d_total))
    if ins:
        if graph_type == 'ER':
            # Erdos-Renyi
            G_und = ig.Graph.Erdos_Renyi(n=p, m=p * e)
            B_und = _graph_to_adjmat(G_und)
            B = _random_acyclic_orientation(B_und)
        elif graph_type == 'SF' or graph_type == 'BA':
            # Scale-free, Barabasi-Albert
            G = ig.Graph.Barabasi(n=p, m=int(round(e)), directed=True)
            B = _graph_to_adjmat(G)
        elif graph_type == 'BP':
            # Bipartite, Sec 4.1 of (Gu, Fu, Zhou, 2018)
            top = int(0.2 * p)
            G = ig.Graph.Random_Bipartite(top, p - top, m=p * e, directed=True, neimode=ig.OUT)
            B = _graph_to_adjmat(G)
        else:
            raise ValueError('unknown graph type')
        B_perm = _random_permutation(B)
        if lags == 0:
            assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
            return B_perm
        B_time[-p:, -p:] = B_perm
    for lag in range(lags, 0, -1):
        for from_node in range(0, p):
            from_node_index = p * (lags - lag) + from_node
            for to_node_index in range(-p, 0, 1):
                random_number = np.random.uniform(low=0.0, high=1.0)
                threshold = 1.0 / p * es[-lag]
                if random_number <= threshold:
                    B_time[from_node_index, to_node_index] = 1
    assert ig.Graph.Adjacency(B_time.tolist()).is_dag()
    return B_time

def lorenz(x, t, F=5):
    """Partial derivatives for Lorenz-96 ODE."""
    p = len(x)
    dxdt = np.zeros(p)
    for i in range(p):
        dxdt[i] = (x[(i + 1) % p] - x[(i - 2) % p]) * x[(i - 1) % p] - x[i] + F
    return dxdt

def simulate_latent_lorenz(n_v, n_z, weight_range, T, delta_t=0.1, F=5.0, sd=0.1, burn_in=1000, method='linear', seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    Z_B, B_W = decade(n_v, n_z, weight_range)
    z0 = np.random.normal(scale=0.01, size=n_z)
    t = np.linspace(0, (T + burn_in) * delta_t, T + burn_in)
    Z = odeint(lorenz, z0, t, args=(F, ))
    if method == 'linear':
        datas = _simulate_linear_sem(n_v, Z, B_W, T, sd, burn_in)
    elif method == 'nonlinear':
        datas = _simulate_nonlinear_sem(n_v, Z, B_W, T, sd, burn_in)
    B = np.zeros((n_z, n_z), dtype=int)
    for i in range(n_z):
        B[i, i] = 1
        B[(i + 1) % n_z, i] = 1
        B[(i - 1) % n_z, i] = 1
        B[(i - 2) % n_z, i] = 1
    GC = np.matmul(Z_B.T, np.matmul(B, Z_B))
    GC = (abs(GC) > 0).astype(int)
    return datas[burn_in:], Z[burn_in:], GC, B, Z_B

def _simulate_linear_sem(n_v, Z, B_W, T, sd,  burn_in):
    X = Z @ B_W
    X += np.random.normal(scale=sd, size=(T + burn_in, n_v))
    return X

def _simulate_nonlinear_sem(n_v, Z, B_W, T, sd,  burn_in):
    X = np.zeros([T +burn_in, n_v])
    # W1 = np.random.uniform(low=0.1, high=0.2, size=[Z.shape[1], 100])
    # W1[np.random.rand(*W1.shape) < 0.5] *= -1
    # W2 = np.random.uniform(low=0.5, high=1, size=[100, n_v])
    # W2[np.random.rand(*W2.shape) < 0.5] *= -1
    # h1 = np.matmul(Z, W1)
    # h1 = np.maximum(0.2 * h1, h1)
    # h2 = np.matmul(h1, W2)
    # print(h2.shape)

    for i in range(n_v):
        parents = np.nonzero(B_W[:, i])
        pa_size = len(parents[0])
        hidden = 10
        W1 = np.random.uniform(low=0.5, high=1, size=[pa_size, hidden])
        W1[np.random.rand(*W1.shape) < 0.5] *= -1
        W2 = np.random.uniform(low=0.5, high=1, size=hidden)
        W2[np.random.rand(hidden) < 0.5] *= -1
        h1 = np.matmul(Z[:, parents], W1)
        h1 = np.maximum(0.2 * h1, h1)
        h2 = np.matmul(h1, W2)
        X[:, i] = h2.squeeze()
    X += np.random.normal(scale=sd, size=(T + burn_in, n_v))
    return X

def rossler(x, t, a=0, eps=0.1, b=4, d=2):
    """Partial derivatives for rossler ODE."""
    p = len(x)
    dxdt = np.zeros(p)
    dxdt[0] = a * x[0] - x[1]
    dxdt[p - 2] = x[(p - 3)]
    dxdt[p - 1] = eps + b * x[(p - 1)] * (x[(p - 2)] - d)

    for i in range(1, p - 2):
        dxdt[i] = np.sin(x[(i - 1)]) - np.sin(x[(i + 1)])

    return dxdt

def simulate_rossler(p, T, sigma=0.5, a=0, eps=0.1, b=4, d=2, delta_t=0.05, sd=0.1, burn_in=1000, seed=None):
    if seed is not None:
        np.random.seed(seed)

    def GG(x, t):
        p = len(x)
        return np.diag([sigma] * p)

    # Use scipy to solve ODE.
    x0 = np.random.normal(scale=0.01, size=p)
    t = np.linspace(0, (T + burn_in) * delta_t, T + burn_in)
    # X = odeint(rossler, x0, t, args=(a,eps,b,d,))
    # X += np.random.normal(scale=sd, size=(T + burn_in, p))

    X = sdeint.itoint(rossler, GG, x0, t)

    # Set up Granger causality ground truth.
    GC = np.zeros((p, p), dtype=int)
    GC[0, 0] = 1
    GC[0, 1] = 1
    GC[p - 2, p - 3] = 1
    GC[p - 1, p - 1] = 1
    GC[p - 1, p - 2] = 1
    for i in range(1, p - 2):
        # GC[i, i] = 1
        GC[i, (i + 1)] = 1
        GC[i, (i - 1)] = 1

    return 400 * X[burn_in:], GC

def glycolytic(
    x, t, k1=0.52, K1=100, K2=6, K3=16, K4=100, K5=1.28, K6=12, K=1.8, kappa=13, phi=0.1, q=4, A=4, N=1, J0=2.5
):
    """Partial derivatives for Glycolytic oscillator model.

    source:
    https://www.pnas.org/content/pnas/suppl/2016/03/23/1517384113.DCSupplemental/pnas.1517384113.sapp.pdf

    Args:
    - r (np.array): vector of self-interaction
    - alpha (pxp np.array): matrix of interactions"""
    dxdt = np.zeros(7)

    dxdt[0] = J0 - (K1 * x[0] * x[5]) / (1 + (x[5] / k1) ** q)
    dxdt[1] = (2 * K1 * x[0] * x[5]) / (1 + (x[5] / k1) ** q) - K2 * x[1] * (N - x[4]) - K6 * x[1] * x[4]
    dxdt[2] = K2 * x[1] * (N - x[4]) - K3 * x[2] * (A - x[5])
    dxdt[3] = K3 * x[2] * (A - x[5]) - K4 * x[3] * x[4] - kappa * (x[3] - x[6])
    dxdt[4] = K2 * x[1] * (N - x[4]) - K4 * x[3] * x[4] - K6 * x[1] * x[4]
    dxdt[5] = (-2 * K1 * x[0] * x[5]) / (1 + (x[5] / k1) ** q) + 2 * K3 * x[2] * (A - x[5]) - K5 * x[5]
    dxdt[6] = phi * kappa * (x[3] - x[6]) - K * x[6]

    return dxdt


def simulate_glycolytic(T, sigma=0.5, delta_t=0.001, sd=0.01, burn_in=1000, seed=None, scale=True):
    if seed is not None:
        np.random.seed(seed)

    def GG(x, t):
        p = len(x)
        return np.diag([sigma] * p)

    x0 = np.zeros(7)
    x0[0] = np.random.uniform(0.15, 1.6)
    x0[1] = np.random.uniform(0.19, 2.16)
    x0[2] = np.random.uniform(0.04, 0.2)
    x0[3] = np.random.uniform(0.1, 0.35)
    x0[4] = np.random.uniform(0.08, 0.3)
    x0[5] = np.random.uniform(0.14, 2.67)
    x0[6] = np.random.uniform(0.05, 0.1)

    # Use scipy to solve ODE.
    t = np.linspace(0, (T + burn_in) * delta_t, T + burn_in)
    # X = odeint(glycolytic, x0, t)
    # X += np.random.normal(scale=sd, size=(T + burn_in, 7))

    X = sdeint.itoint(glycolytic, GG, x0, t)

    # Set up ground truth.
    GC = np.zeros((7, 7), dtype=int)
    GC[0, :] = np.array([1, 0, 0, 0, 0, 1, 0])
    GC[1, :] = np.array([1, 1, 0, 0, 1, 1, 0])
    GC[2, :] = np.array([0, 1, 1, 0, 1, 1, 0])
    GC[3, :] = np.array([0, 0, 1, 1, 1, 1, 1])
    GC[4, :] = np.array([0, 1, 0, 0, 1, 1, 0])
    GC[5, :] = np.array([1, 1, 0, 0, 0, 1, 0])
    GC[6, :] = np.array([0, 0, 0, 1, 0, 0, 1])

    if scale:
        X = np.transpose(
            np.array([(X[:, i] - X[:, i].min()) / (X[:, i].max() - X[:, i].min()) for i in range(X.shape[1])])
        )

    return 10 * X[burn_in:], GC.T