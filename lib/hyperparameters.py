import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
# from sklearn_extra.cluster import KMedoids
from tqdm.notebook import tqdm

# ------------------------------------------------------------
# 1) PCA: choose n_components by cumulative explained variance
# ------------------------------------------------------------

def choose_pca_components(
    X, var_targets=(0.95, 0.99), whiten=False,
    fast_pca=False, max_components=10_000, max_samples=20_000, iterated_power=None, # parameters for fast_pca
    center=True, randomized=True, random_state=0, show_plot=True
):
    """
    X: (n_samples, n_features) 2D array of features (e.g., vectorized dFC windows concatenated)
    var_targets: tuple of cumulative variance targets (e.g., 0.95, 0.99)
    center: subtract column mean before PCA
    randomized: use randomized SVD for speed on large problems
    
    If fast_pca=True, fit PCA on a SUBSET of windows and with a CAP on n_components.
    Much faster and typically sufficient to decide 95/99% cumulative variance.
    X: (n_samples, n_features) 2D array (e.g., all dFC windows concatenated)
    max_samples: random subset of rows to fit PCA (e.g., 20k)
    max_components: cap on computed PCs (e.g., 300)
    
    Returns:
        result = {
            "cumvar": cumulative_explained_variance,   # (n_components_max,)
            "targets": {target: suggested_n for each target},
            "pca": fitted_PCA,
            "X_mean": column_mean_used (or zeros if center=False)
        }
    """
    
    X = np.asarray(X)
    n_samples, _ = X.shape
    
    if fast_pca and n_samples > max_samples:
        # 1) Row-subset to speed up the PCA fit
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n_samples, size=max_samples, replace=False)
        X = X[idx]
        
    # 2) Cast to float32 to halve memory and speed up SVD
    X = X.astype(np.float32, copy=False)
    
    # 3) Center (like training mean) — keep this mean to center full data later
    if center:
        X_mean = X.mean(axis=0, keepdims=True)
        # Xc = X - X_mean
        scaler = StandardScaler(with_mean=True, with_std=False).fit(X)
        Xc = scaler.transform(X)
    else:
        X_mean = np.zeros((1, X.shape[1]), dtype=X.dtype)
        Xc = X

    # 4) Fit PCA with full spectrum or with limited components
    n_effective_samples = min(max_components, *Xc.shape) if fast_pca else min(*Xc.shape)
    pca = PCA(
        n_components=n_effective_samples, whiten=whiten,
        svd_solver="randomized" if randomized else "auto",
        iterated_power=iterated_power if iterated_power else "auto",
        random_state=random_state
    )
    pca.fit(Xc)

    evr = pca.explained_variance_ratio_
    cumvar = np.cumsum(evr)

    # 5) Find smallest n reaching each target (95/99…)
    targets = {}
    for t in var_targets:
        idx = np.searchsorted(cumvar, t) + 1  # +1 because components are 1...n
        idx = int(min(idx, len(cumvar)))
        targets[t] = idx

    # 6) Plot cumulative explained variance with targets
    if show_plot:
        plot_cumulative_variance(cumvar, targets, pca, X, fast_pca)
        
    return {
        "cumvar": cumvar, 
        "targets": targets, 
        "pca": pca,         # PCA fitted on subset (handy for exploration), if fast_pca
        "X_mean": X_mean    # mean of subset for re-fitting full PCA later, if fast_pca
    }

# --------------------------------------------
# 1.a) PCA: plot cumulative explained variance
# --------------------------------------------

def plot_cumulative_variance(cumvar, targets, pca, X, fast_pca=False, scree_plot=False, save_fig=False):
    colors = ['tab:red', 'tab:purple', 'tab:blue', 'tab:orange', 'tab:green']
    fig, ax = plt.subplots(1, 2 if scree_plot else 1, figsize=(14, 5.5))
    if not scree_plot:
        ax = [ax]
    ax[0].plot(np.arange(1, len(cumvar)+1), cumvar, marker='o', markersize=8, markeredgewidth=0.1, markeredgecolor='lightgrey', lw=1, color=colors[0])
    handles, labels = [], []
    for kk, (t, n) in enumerate(targets.items(), start=2):
        ax[0].axhline(y=t, ls='--', lw=1, color=colors[kk])
        ax[0].axvline(x=n, ls='--', lw=1, color=colors[kk])
        labels.append(rf"{100*t:.1f}\% → {n} PCs")
        handles.append(plt.Line2D([0], [0], color=colors[kk], lw=0))
        
    ax[0].set_xlabel("Number of principal components")
    ax[0].set_ylabel("Cumulative explained variance", fontsize=20)
    title = f"PCA " + (f"(subset={X.shape[0]}, max_components={pca.n_components_})" if fast_pca else "cumulative explained variance")
    ax[0].set_title(title, fontsize=30)
    ax[0].set_ylim(0, 1.1)
    ax[0].set_xscale('log')  # linear
    ax[0].grid(True, alpha=0.3)
    leg= ax[0].legend(
        handles, labels,
        loc='best',
        frameon=False)
    for txt, col in zip(leg.get_texts(), colors[2:2+len(labels)]):
        txt.set_color(col)

    if scree_plot:
        ax[1].plot(np.arange(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_, 
                marker='o', linewidth=1, color=colors[1])
        ax[1].set_xlabel("Component")
        ax[1].set_ylabel("Explained variance ratio")
        ax[1].set_title("Scree plot " + (f"(subset={X.shape[0]}, max_components={pca.n_components_})" if fast_pca else ""))
        ax[1].grid(True, alpha=0.3)
        ax[1].set_xscale('log')
    
    plt.tight_layout()
    if save_fig:
        plt.savefig(f'imgs/{"_".join(title.split(" "))}.png', dpi=300)
    plt.show()


# --------------------------------------------------------------
# 2) K selection for k-means: silhouette (+ CH & Davies–Bouldin)
# --------------------------------------------------------------

def kmeans_k_selection(
    X, k_min=2, k_max=15, center=True,
    fast_k_means=False,         # if True, evaluate K on a SUBSET, in PCA space if provided.
    pca_for_metric=None, 
    n_pcs=None,                 # if given, take first n_pcs of pca.transform(X)
    L1_norm_for_metric=False,   # if True, select for clustering matrices whose L1norm is 1.5 standard deviation away from the L1-norm of the mean correlation matrix
    static_FC=None,             # if given, comparison is made wrt these matrices rather than median or mean sliding windows values
    ROIs=None, win_len_per_subj=None,
    sample_for_k=20000,         # subset of rows for scoring
    n_init=50, random_state=0, show_plot=True
):
    """
    Choose number of clusters K for k-means using Silhouette (and optionally CH, DB).
    X: (n_samples, n_features)
    center: subtract column mean before evaluation
    pca_for_metric: None or fitted PCA to project X before scoring (recommended in very high-D)
                    If None, scores are computed in original feature space.
    fast_k_means: if True, evaluate K on a SUBSET of rows (sample_for_k) in PCA space if provided.
    n_pcs: if pca_for_metric is given, take only the first n_pcs of the projection.
    sample_for_k: subset of rows for scoring (e.g., 20k)
    n_init: number of k-means initializations (e.g., 50 for final fit, 10 for model selection)
    Returns:
        result = {
            "Ks": array([k_min..k_max]),
            "silhouette": array,
            "calinski_harabasz": array,
            "davies_bouldin": array,
            "best_k_silhouette": int,
            "models": {k: fitted_kmeans_for_that_k}
        }
    """
    X = np.asarray(X)
    n_samples, _ = X.shape
    
    if fast_k_means and n_samples > sample_for_k:
        # 1) Subsample rows
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n_samples, size=sample_for_k, replace=False)
        X = X[idx]
    # 2) Cast to float32 to halve memory and speed up SVD
    X = X.astype(np.float32, copy=False)
    
    # 3) Center (like training mean) — keep this mean to center full data later
    if center:
        X_mean = X.mean(axis=0, keepdims=True)
        # Xc = X - X_mean
        scaler = StandardScaler(with_mean=True, with_std=False).fit(X)
        Xc = scaler.transform(X)
    else:
        X_mean = np.zeros((1, X.shape[1]), dtype=X.dtype)
        Xc = X

    # 4) Optionally project to PCA space (reduced if n_pcs) for scoring
    if pca_for_metric is not None and not L1_norm_for_metric:
        Z = pca_for_metric.transform(Xc)
        if n_pcs is not None:
            Z = Z[:, :n_pcs]
    elif L1_norm_for_metric:
        Z = select_salient_windows(Xc, win_len_per_subj=win_len_per_subj, N=ROIs, static_FC=static_FC)
    else:
        Z = Xc
    print(f'From {Xc.shape = } to {Z.shape = }')

    # 5) Fit k-means for each K and compute scores
    Ks = np.arange(k_min, k_max+1, dtype=int)
    sils, chs, dbs = [], [], []
    inertias = []
    models = {}

    for k in tqdm(Ks, desc="K selection"):
        km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
        labels = km.fit_predict(Z)
        models[k] = km
        inertias.append(km.inertia_)

        # Silhouette: needs at least 2 clusters and less than n_samples
        sil = silhouette_score(Z, labels, metric='euclidean') if k > 1 and len(np.unique(labels)) > 1 else np.nan
        sils.append(sil)

        # CH and DB indices
        chs.append(calinski_harabasz_score(Z, labels))
        dbs.append(davies_bouldin_score(Z, labels))

    sils = np.array(sils, dtype=float)
    chs  = np.array(chs, dtype=float)
    dbs  = np.array(dbs, dtype=float)

    # 6) Find best K by silhouette (max)
    best_idx = np.nanargmax(sils)
    best_k_sil = Ks[best_idx]
    best_k_ch = Ks[np.nanargmax(chs)]
    best_k_db = Ks[np.nanargmin(dbs)]
    
    # 7) Find elbow in inertia plot
    best_k_elbow, dists = elbow_from_curve(Ks, inertias, logy=True)

    # 8) Plot scores
    if show_plot:
        plot_k_selection(Ks, sils, chs, dbs, best_k_sil, X, fast_k_means)
        plot_elbow(Ks, inertias, best_k=best_k_elbow, logy=True)

    return {
        "Ks": Ks,
        "sils": sils,                   # Silhouette
        "chs": chs,                     # Calinski-Harabasz
        "dbs": dbs,                     # Davies-Bouldin
        "inertias": inertias,           # Inertias
        "best_k_sil": int(best_k_sil),  # best_k from Silhouette
        "best_k_ch": int(best_k_ch),    # best_k from CH
        "best_k_db": int(best_k_db),    # best_k from DB
        "dists": dists,                 # maximum distance from the K_min-K_max line
        "best_k_elbow": best_k_elbow,   # point corresponding to the max distance
        "models": models,               # fitted on subset only if fast_k_means
        "subset_size": Z.shape[0],
        "used_pcs": Z.shape[1],
    }

# -----------------------------------------
# 2.a) K selection for k-means: plot scores
# -----------------------------------------

def plot_k_selection(Ks, sils, chs, dbs, best_k, X, fast_k_means=False):
    colors = ['tab:red', 'tab:purple', 'tab:orange', 'tab:blue', 'tab:green']
    fig, ax = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    ax[0].plot(Ks, sils, marker='o', color=colors[0])
    ax[0].axvline(best_k, ls='--', lw=1, color=colors[-1])
    ax[0].set_ylabel("Silhouette (↑ better)", fontsize=12)
    ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(Ks, chs, marker='o', color=colors[1])
    ax[1].axvline(Ks[np.argmax(chs)], ls='--', lw=1, color=colors[-1])
    ax[1].set_ylabel("Calinski-Harabasz (↑ better)", fontsize=12)
    ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[1].grid(True, alpha=0.3)

    ax[2].plot(Ks, dbs, marker='o', color=colors[2])
    ax[2].axvline(Ks[np.argmin(dbs)], ls='--', lw=1, color=colors[-1])
    ax[2].set_ylabel("Davies-Bouldin (↓ better)", fontsize=12)
    ax[2].set_xlabel("Number of clusters (K)", fontsize=15)
    ax[2].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[2].grid(True, alpha=0.3)

    ax[0].set_title("K-means model selection " + (f"(subset={X.shape[0]})" if fast_k_means else ""), fontsize=17)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
   
# ----------------------------------------
# 2.b) Elbow from inertia curve estimation
# ---------------------------------------- 
    
def elbow_from_curve(Ks, inertias, logy=True):
    """
    Estimates the 'elbow' as the point of maximum distance from the K_min-K_max line.
    If logy=True, applies log inertia to stabilize the scales.
    """
    y = np.log(inertias) if logy else inertias
    x = Ks.astype(float)

    # line between first and last point
    p1 = np.array([x[0], y[0]]); p2 = np.array([x[-1], y[-1]])
    v  = p2 - p1
    v /= np.linalg.norm(v) + 1e-12

    # perpecndicular distance of each point on the rope
    diffs = np.stack([x, y], axis=1) - p1
    proj  = (diffs @ v)[:, None] * v
    perp  = diffs - proj
    dist  = np.linalg.norm(perp, axis=1)

    best_idx = int(np.argmax(dist))
    best_k = int(Ks[best_idx])
    return best_k, dist

# ---------------
# 2.c) Elbow plot
# ---------------

def plot_elbow(Ks, inertias, best_k=None, logy=True, title="K-means elbow"):
    plt.figure(figsize=(7, 4))
    y = np.log(inertias) if logy else inertias
    plt.plot(Ks, y, marker='o')
    if best_k is not None:
        plt.axvline(best_k, ls='--', lw=1, color='tab:green')
        plt.text(best_k, y[Ks.tolist().index(best_k)], f"  elbow @ K={best_k}", va='bottom')
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("log(inertia)")
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# ------------------------------
# 2.d) L1 norm windows selection
# ------------------------------

def vec_upper(C):
    N = C.shape[0]
    iu = np.triu_indices(N, k=1)
    return C[iu]

def infer_N_from_E(E):
    # Solve N(N-1)/2 = E  ->  N = (1 + sqrt(1+8E))/2
    N = int((1 + np.sqrt(1 + 8*E)) / 2)
    if N*(N-1)//2 != E:
        raise ValueError(f"E={E} is not a valid upper-triangular size.")
    return N

def vec_to_sym_matrix(V, N=None, fill_diag=1.0):
    """
    V: (W, E) array of upper-triangular (k=1) entries per window
    N: optional matrix size; inferred from E if None
    fill_diag: value for the diagonal (e.g., 1.0 for correlations; None to leave zeros)
    Returns: M of shape (W, N, N)
    """
    V = np.asarray(V)
    W, E = V.shape
    if N is None:
        N = infer_N_from_E(E)

    i, j = np.triu_indices(N, k=1)
    M = np.zeros((W, N, N), dtype=V.dtype)

    # fill upper triangle for all windows at once
    M[:, i, j] = V
    # mirror to lower triangle
    M[:, j, i] = V

    if fill_diag is not None:
        idx = np.arange(N)
        M[:, idx, idx] = fill_diag
    return M

def vec_to_matrix(V, N=None, fill_diag=1.0):
    """
    V: (W, E) array of upper-triangular (k=1) entries per window
    N: optional matrix size; inferred from E if None
    fill_diag: value for the diagonal (e.g., 1.0 for correlations; None to leave zeros)
    Returns: M of shape (W, N, N)
    """
    V = np.asarray(V)
    E = V.shape
    if N is None:
        N = infer_N_from_E(E)
    M = np.zeros((N, N), dtype=V.dtype)

    # fill upper triangle for all windows at once
    M[np.triu_indices(N, k=1)]   = V
    # mirror to lower triangle
    M.T[np.triu_indices(N, k=1)] = V

    if fill_diag is not None:
        idx = np.arange(N)
        M[idx, idx] = fill_diag
    return M

def select_salient_windows(vec_FC, win_len_per_subj, N=None, static_FC=None, z_thresh=1.5, median=True):    
    """

    Args:
        vec_FC (list): list of (W_i) correlation matrices (W, E) per subject
        win_len_per_subj (int): length of each window per subject
        N (int, optional): number of ROIs. If not given is inferred with `infer_N_from_E` function.
        z_thresh (float, optional): outliers cutoff threshold. Defaults to 1.5.
        median (bool, optional): select filter type. Defaults to True. 
                                 If False, 'mean' is used instead

    Returns:
        list: list of selected edge vectors
    """    
    # average subject FC 
    Z = []
    for mat, subj in enumerate(range(0, vec_FC.shape[0], win_len_per_subj)):
        subj_windows_vec = vec_FC[subj:subj+win_len_per_subj, :]    # (W_i, E), E = Nx(N-1)/2
        subj_windows_FC = vec_to_sym_matrix(subj_windows_vec, N=N)  # matrix (W_i, N, N)
        # mean or median
        filter_func = np.median if median else np.mean
        C_mean = np.array(static_FC[mat]) if static_FC is not None else filter_func(subj_windows_FC, axis=0) 
    
        v_mean = vec_upper(C_mean)
        d = np.sum(np.abs(subj_windows_vec - v_mean), axis=1)       # L1 distance to subject mean
        mu, sd = d.mean(), d.std(ddof=1)
        z = (d - mu) / (sd + 1e-12)
        keep = z > z_thresh
        Z.append(subj_windows_vec[keep])
    return np.concatenate(Z, axis=0)
