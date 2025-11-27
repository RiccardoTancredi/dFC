import numpy as np
import pandas as pd
from sliding_window import fisher_r_to_z
from scipy.stats import ttest_rel, ttest_ind
from scipy.stats import wilcoxon, mannwhitneyu, pearsonr, linregress, norm
import statsmodels.formula.api as smf
import numpy as np

# ---------- A) Centroids by condition/state ----------

def centroid_per_condition_state(X_vec, labels, cond_of_win, K, thr=100):
    """

    Args:
        X_vec (array): all dFC windows concatenated in vectorial form (W,E)
                        W = # windows, E = # edges (upper FC_{i} triangle)
        labels (list): labels of states (0,...,K-1) for each window W
        cond_of_win (list): label of condition. A list matching each window 
                        to each condition. E.g.: [“sham_pre”, “sham_post”, ...]
        K (int): total # states 
        thr (int): minimum threshold of windows for considering a centroid valid. 

    Returns:
        out (dict): for each condition a matrix (K,E) is assigned. 
                    Each row is the centroid of the k-th state in that condition.
    """    
    # return dict cond -> (K, E)
    out = {}
    for cond in sorted(set(cond_of_win)):
        C = []
        for k in range(K):
            sel = (cond_of_win==cond) & (labels==k)
            C.append(np.median(X_vec[sel], axis=0) if np.sum(sel) > thr else np.zeros(X_vec.shape[1]))
        out[str(cond)] = np.vstack(C)
    return out


# ---------- B) Observed similarity between two conditions ----------

def centroid_similarity_stats(CA, CB, K, use_states=None):
    """
    Compare two (K,E) centroid sets state-by-state (same K index), returning:
      - r_k: Pearson correlations per state (shape K), NaN if any centroid is NaN
      - l2_k: L2 distances per state (shape K), NaN if any centroid is NaN, in r-space
      - zbar: mean Fisher-z across states with valid r
      - l2_mean: mean L2 across states with valid distances

    Parameters
    ----------
    CA, CB : dictionary of (K, E) elements
        Centroid matrices for condition A and B (same states index).
    K: number of states states
    use_states : array-like or None
        Optional list of state indices to include. If None, use all 0..K-1.

    Returns
    -------
    r_k : (K,) float
    l2_k : (K,) float
    zbar : float
    l2_mean : float
    """
    if use_states is None:
        use_states = np.arange(K)

    r_k = np.full(K, np.nan, dtype=float)
    l2_k = np.full(K, np.nan, dtype=float)

    for k in use_states:
        a = CA[k]; b = CB[k]
        if np.any(~np.isfinite(a)) or np.any(~np.isfinite(b)):
            continue
        # Pearson correlation
        r, _ = pearsonr(a, b)
        r_k[k] = r
        # L2 distance
        l2_k[k] = np.linalg.norm(np.tanh(a) - np.tanh(b))

    # Aggregate statistics (ignore NaNs)
    valid_r = np.isfinite(r_k)
    valid_d = np.isfinite(l2_k)
    rbar = np.nan
    l2_mean = np.nan
    if np.any(valid_r):
        # Fisher z (arctanh), then mean
        rbar = np.tanh(np.mean(fisher_r_to_z(r_k[valid_r])))
    if np.any(valid_d):
        l2_mean = np.mean(l2_k[valid_d])
    return r_k, l2_k, float(rbar), float(l2_mean)


# ---------- C) Permutation test at subject level ----------

def perm_test_centroid_similarity(
    X_vec, labels, cond_of_win, subj_of_win,
    K, condA, condB,
    paired=True, n_perm=1000, reducer="median", min_windows=10,
    stat_kind="zbar"
):
    """
    Block-permutation test for centroid similarity between two conditions.

    Parameters
    ----------
    X_vec, labels, cond_of_win, subj_of_win : arrays of length W
        All aligned (same order of windows).
    K : int
        Number of states.
    condA, condB : str
        The two conditions to compare (must be present in cond_of_win).
    paired : bool
        If True, do within-subject swap between the two conditions (A<->B) for each subject
        that has both conditions. If False (between-group), shuffle group labels across subjects.
    n_perm : int
        Number of permutations.
    reducer, min_windows : passed to centroids_by_condition_state
    stat_kind : {"zbar","l2"}
        Which summary statistic to test: mean Fisher-z across states ("zbar") or mean L2 ("l2").

    Returns
    -------
    observed : float
        Observed summary statistic (zbar or l2_mean).
    pvalue : float
        One-sided permutation p-value (greater-or-equal for zbar; less-or-equal for l2).
    dist : (n_perm,) float
        Null distribution of the statistic.
    detail : dict
        Extra info (r_k, l2_k for observed; counts per state, etc.).
    """
    # 1) Observed centroids
    centroids = centroid_per_condition_state(X_vec, labels, cond_of_win, K)
    CA = centroids[str(condA)]
    CB = centroids[str(condB)]
    r_k, l2_k, zbar, l2m = centroid_similarity_stats(CA, CB, K)
    observed = zbar if stat_kind == "zbar" else l2m

    # 2) Prepare permutation blocks at subject level
    subjects = np.unique(subj_of_win)
    # Which windows belong to A or B
    isA = (cond_of_win == condA)
    isB = (cond_of_win == condB)

    # Mapping subj -> indices for A and B
    subj_idx_A = {}
    subj_idx_B = {}
    for s in subjects:
        sel_s = (subj_of_win == s)
        subj_idx_A[s] = np.where(sel_s & isA)[0]
        subj_idx_B[s] = np.where(sel_s & isB)[0]

    # 3) Permutations
    dist = np.empty(n_perm, dtype=float)
    rng = np.random.default_rng(0)

    # Make a working copy of cond_of_win (object dtype) for shuffling
    cond_perm = cond_of_win.copy()

    for b in range(n_perm):
        # -- paired case: swap A<->B within each subject with p=0.5 if both present
        if paired:
            for s in subjects:
                idxA = subj_idx_A[s]; idxB = subj_idx_B[s]
                if idxA.size > 0 and idxB.size > 0 and rng.random() < 0.5:
                    # swap labels for this subject between A and B
                    cond_perm[idxA] = condB
                    cond_perm[idxB] = condA
                else:
                    # keep as is
                    cond_perm[idxA] = condA
                    cond_perm[idxB] = condB
        else:
            # between-group: shuffle subject group labels, then assign all windows of a subject accordingly
            # Build current subj->group based on majority of their windows (A or B)
            subj_group = {}
            for s in subjects:
                nA = subj_idx_A[s].size
                nB = subj_idx_B[s].size
                if nA == 0 and nB == 0:
                    subj_group[s] = None
                elif nA >= nB:
                    subj_group[s] = condA
                else:
                    subj_group[s] = condB
            # Shuffle the group labels among subjects that have a group
            valid_subj = [s for s in subjects if subj_group[s] is not None]
            groups = [subj_group[s] for s in valid_subj]
            rng.shuffle(groups)
            for s, g in zip(valid_subj, groups):
                # assign all A/B windows of s to the shuffled group label g (the other cond becomes the other)
                if g == condA:
                    cond_perm[subj_idx_A[s]] = condA
                    cond_perm[subj_idx_B[s]] = condB
                else:
                    cond_perm[subj_idx_A[s]] = condB
                    cond_perm[subj_idx_B[s]] = condA

        # Recompute centroids under permuted labels
        cent_p = centroid_per_condition_state(X_vec, labels, cond_perm, K)
        CAp = cent_p[str(condA)]
        CBp = cent_p[str(condB)]
        r_kp, l2_kp, zbar_p, l2m_p = centroid_similarity_stats(CAp, CBp)
        dist[b] = zbar_p if stat_kind == "zbar" else l2m_p

    # 4) One-sided p-value
    if stat_kind == "zbar":
        # larger or equal is more similar than null
        pval = (np.sum(dist >= observed) + 1.0) / (n_perm + 1.0)
    else:
        # smaller or equal L2 is more similar than null
        pval = (np.sum(dist <= observed) + 1.0) / (n_perm + 1.0)

    detail = dict(r_k=r_k, l2_k=l2_k, r_kp=r_kp, l2_kp=l2_kp)
    return observed, pval, dist, detail


# ------------------ Dynamic metrics beyond centroids ------------------

def _sample_pairs(nA, nB=None, max_pairs=50000, rng=None):
    """Return indices to sample up to max_pairs pairs without replacement."""
    rng = np.random.default_rng(rng)
    if nB is None:
        s = min(nA, int((1 + np.sqrt(1 + 8*max_pairs)) // 2))
        idx = rng.choice(nA, size=s, replace=False)
        ia, ib = np.triu_indices(s, k=1)
        return idx[ia], idx[ib]
    tot = nA * nB
    if tot == 0:
        return np.array([], int), np.array([], int)
    m = min(max_pairs, tot)
    k = rng.choice(tot, size=m, replace=False)
    ia = k // nB
    ib = k %  nB
    return ia, ib

def state_distance_distributions(X_vec, labels, cond_of_win, condA, condB, k,
                                 metric="cosine", max_pairs=50000, rng=0):
    """
    Compare distributions of window-to-window distances within and across conditions
    for a given state k.

    Returns
    -------
    d_within_A : (M1,) distances among windows in condA & state k
    d_within_B : (M2,) distances among windows in condB & state k
    d_cross    : (M3,) distances between condA/state k and condB/state k
    nA, nB     : counts of windows per group
    """
    selA = (cond_of_win == condA) & (labels == k)
    selB = (cond_of_win == condB) & (labels == k)
    XA = X_vec[selA]; XB = X_vec[selB]
    nA, nB = XA.shape[0], XB.shape[0]
    if nA < 2: 
        dA = np.array([])
    else:
        ia, ib = _sample_pairs(nA, max_pairs=max_pairs, rng=rng)
        dA = np.linalg.norm(XA[ia] - XA[ib], axis=1) if metric=="euclidean" else \
             (1.0 - np.einsum('ij,ij->i', XA[ia], XA[ib]) /
              (np.linalg.norm(XA[ia], axis=1) * np.linalg.norm(XA[ib], axis=1) + 1e-12))
    if nB < 2:
        dB = np.array([])
    else:
        ia, ib = _sample_pairs(nB, max_pairs=max_pairs, rng=rng)
        dB = np.linalg.norm(XB[ia] - XB[ib], axis=1) if metric=="euclidean" else \
             (1.0 - np.einsum('ij,ij->i', XB[ia], XB[ib]) /
              (np.linalg.norm(XB[ia], axis=1) * np.linalg.norm(XB[ib], axis=1) + 1e-12))
    if nA==0 or nB==0:
        dX = np.array([])
    else:
        ia, ib = _sample_pairs(nA, nB, max_pairs=max_pairs, rng=rng)
        dX = np.linalg.norm(XA[ia] - XB[ib], axis=1) if metric=="euclidean" else \
             (1.0 - np.einsum('ij,ij->i', XA[ia], XB[ib]) /
              (np.linalg.norm(XA[ia], axis=1) * np.linalg.norm(XB[ib], axis=1) + 1e-12))
    return dA, dB, dX, nA, nB


def dispersion_by_condition_state(X_vec, labels, cond_of_win, K, reducer="median", min_windows=1000):
    """
    For each condition and state, compute:
      - centroid (median/mean)
      - per-window distances to its centroid (Euclidean)
      - summary stats of dispersion (mean, std)

    Returns
    -------
    out : dict
      cond -> {
        "centroids": (K,E) with NaN rows if insufficient windows,
        "counts":    (K,),
        "disp_mean": (K,),
        "disp_std":  (K,),
        "dist_list": list of length K with 1D arrays of per-window distances
      }
    """
    conds = np.unique(cond_of_win)
    E = X_vec.shape[1]
    out = {}
    for cond in conds:
        C     = np.full((K, E), np.nan)
        cnt   = np.zeros(K, dtype=int)
        dmean = np.full(K, np.nan); dstd = np.full(K, np.nan)
        dlist = [None]*K
        sel_c = (cond_of_win == cond)
        for k in range(K):
            sel = sel_c & (labels == k)
            n = np.count_nonzero(sel)
            cnt[k] = n
            if n >= min_windows:
                X = X_vec[sel]
                cent = np.median(X, axis=0) if reducer=="median" else np.mean(X, axis=0)
                C[k] = cent
                d = np.linalg.norm(X - cent[None,:], axis=1)
                dlist[k] = d
                dmean[k] = float(d.mean()); dstd[k] = float(d.std(ddof=1))
        out[str(cond)] = dict(centroids=C, counts=cnt, disp_mean=dmean, disp_std=dstd, dist_list=dlist)
    return out


# ------------------- Effect size Cohen's-d test -------------------

def cohens_d(x, y, equal_var=True):
    """
    Cohen's d per due gruppi indipendenti.
    equal_var=True -> pooled std; False -> usa std separata (meno comune per d).
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    mx, my = np.mean(x), np.mean(y)
    if equal_var:
        nx, ny = len(x), len(y)
        sx2 = np.var(x, ddof=1); sy2 = np.var(y, ddof=1)
        sp = np.sqrt(((nx-1)*sx2 + (ny-1)*sy2) / (nx+ny-2))
        return (mx - my) / sp if sp > 0 else np.nan
    else:
        sx = np.std(x, ddof=1); sy = np.std(y, ddof=1)
        denom = np.sqrt((sx**2 + sy**2) / 2.0)
        return (mx - my) / denom if denom > 0 else np.nan


# ------------------- Edge-Wise Analysis -------------------

def bh_fdr(pvals, alpha=0.05):
    """Benjamini-Hochberg: return q-values and a significance mask."""
    pvals = np.asarray(pvals, float)
    n = pvals.size
    order = np.argsort(pvals)
    ranked = pvals[order]
    bh = ranked * n / (np.arange(1, n+1))
    # q-values monotone
    qvals_ranked = np.minimum.accumulate(bh[::-1])[::-1]
    qvals = np.empty_like(qvals_ranked)
    qvals[order] = qvals_ranked
    return qvals, qvals <= alpha


def effect_size_independent(x, y):
    """Hedges' g (independent, potential different variances)."""
    nx, ny = len(x), len(y)
    sx2, sy2 = np.var(x, ddof=1), np.var(y, ddof=1)
    sp = np.sqrt(((nx-1)*sx2 + (ny-1)*sy2) / (nx+ny-2) + 1e-12)
    g = (np.mean(x) - np.mean(y)) / sp
    # small-sample correction
    J = 1 - (3/(4*(nx+ny)-9))
    return g * J

def wilcoxon_rank_biserial(x, y):
    # r_rb = (2*W - T) / T, with T = n(n+1)/2 and W = sum of positive dei ranks
    # Note: SciPy wilcoxon ignore even-diff=0; here we assume at least one non-zero difference.
    res = wilcoxon(x, y, zero_method='wilcox', alternative='two-sided')
    n = np.sum((x - y) != 0)
    if n == 0:
        return 0.0
    T = n * (n + 1) / 2.0
    W = res.statistic
    return float((2.0 * W - T) / T)

def ttests_edgewise(
    fc_by_cond,
    condA, condB,
    *,
    paired=True,
    already_fisher_z=True,
    alpha=0.05,
    tail="two-sided",  # "two-sided", "greater" (A>B), "less" (A<B)
    node_network=None, # array/list of length N: network labels
    edge_mask=None,    # NxN boolean matrix to narrow the analysis
    # --- NEW ---
    inference_level="edge",  # "edge" | "network" | "both"
    method="parametric",     # "parametric" | "perm" (used for network analysis)
    n_perm=5000,
    fwer=False,              # if True, use max-T instead of FDR-BH for network analysis
    random_state=0,
    network_stat="auto"      # "auto" | "mean_abs_t" | "mean_t"
):
    """
    Returns (df_edge, df_summary).
        - df_edge: As before (edgewise t, p, q, etc.)
        - df_summary:
        - If inference_level includes "network":
            Adds p_perm / q columns (or p_fwer if fwer=True) for network-pair.
        - Otherwise, it remains the same as before.
    """
    rng = np.random.default_rng(random_state)

    A = np.array(fc_by_cond[condA])  # (S1, N, N)
    B = np.array(fc_by_cond[condB])  # (S2, N, N)
    N = A.shape[1]
    assert A.shape[1] == B.shape[1] == A.shape[2] == B.shape[2], f"Matrix must have the same N: {N}"

    if not already_fisher_z:
        A = np.arctanh(A)
        B = np.arctanh(B)

    iu, ju = np.triu_indices(N, k=1)

    # Apply edge_mask if given
    if edge_mask is not None:
        assert edge_mask.shape == (N, N), f"Edge mask matrix must have the same N: {N}"
        keep = edge_mask[iu, ju]
        iu, ju = iu[keep], ju[keep]

    rows = []
    pvals = []

    # ---------- EDGEWISE PARAMETRIC ----------
    for i, j in zip(iu, ju):
        x = A[..., i, j]
        y = B[..., i, j]

        # Avoid NaN o inf
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            continue

        if paired:
            # Require same number of subjects and order matching
            assert x.shape[0] == y.shape[0], "Paired=True requires same number of subjects"
            tstat, p = ttest_rel(x, y, nan_policy='omit', alternative=tail)
            es = cohens_d(x, y)
        else:
            tstat, p = ttest_ind(x, y, equal_var=False, nan_policy='omit', alternative=tail)
            es = effect_size_independent(x, y)
        
        pvals.append(p)
        row = {
            "i": int(i), "j": int(j),
            "t": float(tstat),
            "p": float(p),
            "sig_p": np.array(p) <= alpha,
            "meanA(r)": np.tanh(float(np.mean(x))),
            "meanB(r)": np.tanh(float(np.mean(y))),
            "diff_mean": np.tanh(float(np.mean(x))) - np.tanh(float(np.mean(y))),
            "effect_size": float(es),
        }
        rows.append(row)

    # No valid edge
    if not rows:
        return (
            pd.DataFrame(columns=["i","j","t","p","q","meanA(r)","meanB(r)","diff_mean","effect_size",
                                  "sig_p","sig_q","t_pos","net_i","net_j","net_pair"]),
            pd.DataFrame(columns=["net_pair","n_edges","n_sig_p","prop_sig_p",
                                  "n_sig_pos_p","n_sig_neg_p","prop_sig_pos_p","prop_sig_neg_p",
                                  "n_sig_q","prop_sig_ q",
                                  "n_sig_pos_q","n_sig_neg_q","prop_sig_pos_q","prop_sig_neg_q",
                                  "n_sig_coherent","prop_sig_coherent",
                                  "mean_effect","mean_diff","mean_t",
                                  "p_perm","q","sig_q","p_fwer"])
        )

    df = pd.DataFrame(rows)

    # FDR (BH) edgewise
    qvals, sigmask = bh_fdr(df["p"].values, alpha=alpha)
    df["q"] = qvals
    df["sig_q"] = sigmask
    df["t_pos"] = df["t"] > 0  # True if A>B in terms of t
    df = df.sort_values(["q","p","i","j"]).reset_index(drop=True)

    # Add network labels if defined
    if node_network is not None:
        node_network = np.asarray(node_network)
        df["net_i"] = node_network[df["i"].values]
        df["net_j"] = node_network[df["j"].values]
        df["net_pair"] = [
            " - ".join(sorted([str(a), str(b)])) for a, b in zip(df["net_i"], df["net_j"])
        ]
    else:
        df["net_i"] = None
        df["net_j"] = None
        df["net_pair"] = None

    # ---------- SUMMARY ----------
    rows_sum = []
    grp = df.groupby("net_pair", dropna=False)
    for name, g in grp:
        n_edges = len(g)
        n_sig_p   = int((g["sig_p"]).sum())
        n_sig_pos_p = int(((g["sig_p"]) & (g["t_pos"])).sum())
        n_sig_neg_p = int(((g["sig_p"]) & (~g["t_pos"])).sum())
        prop_sig_p     = n_sig_p / n_edges if n_edges else np.nan
        prop_sig_pos_p = n_sig_pos_p / n_edges if n_edges else np.nan
        prop_sig_neg_p = n_sig_neg_p / n_edges if n_edges else np.nan
        n_sig_q   = int((g["sig_q"]).sum())
        n_sig_pos_q = int(((g["sig_q"]) & (g["t_pos"])).sum())
        n_sig_neg_q = int(((g["sig_q"]) & (~g["t_pos"])).sum())
        prop_sig_q     = n_sig_q / n_edges if n_edges else np.nan
        prop_sig_pos_q = n_sig_pos_q / n_edges if n_edges else np.nan
        prop_sig_neg_q = n_sig_neg_q / n_edges if n_edges else np.nan

        out = {
            "net_pair": name,
            "n_edges": n_edges,
            "n_sig_p": n_sig_p,
            "prop_sig_p": prop_sig_p,   # n_sig / n_edges 
            "n_sig_pos_p": n_sig_pos_p, # t > 0 (A > B)
            "n_sig_neg_p": n_sig_neg_p, # t < 0 (A < B)
            "prop_sig_pos_p": prop_sig_pos_p, 
            "prop_sig_neg_p": prop_sig_neg_p,
            "n_sig_q": n_sig_q,
            "prop_sig_q": prop_sig_q,   # n_sig / n_edges 
            "n_sig_pos_q": n_sig_pos_q, # t > 0 (A > B)
            "n_sig_neg_q": n_sig_neg_q, # t < 0 (A < B)
            "prop_sig_pos_q": prop_sig_pos_q, 
            "prop_sig_neg_q": prop_sig_neg_q,
            "mean_effect": g["effect_size"].mean(),
            "mean_diff": g["diff_mean"].mean(),
            "mean_t": g["t"].mean(),
        }
        rows_sum.append(out)

    df_summary = pd.DataFrame(rows_sum).sort_values(
        ["prop_sig_q","n_sig_q"], ascending=[False, False]
    ).reset_index(drop=True)

    # ---------- NETWORK-LEVEL INFERENCE (perm or parametric) ----------
    want_network = inference_level in ("network", "both")
    if want_network and node_network is None:
        # We can't do network-level inference without labels
        # Return what we've calculated so far 
        return df, df_summary

    if want_network:
        # Prepare edge index -> network-pair
        nets = np.asarray(node_network)
        net_i = nets[df["i"].values]
        net_j = nets[df["j"].values]
        net_pair = np.array([
            " - ".join(sorted([str(a), str(b)]))
            for a, b in zip(net_i, net_j)
        ])
        pairs = np.unique(net_pair)

        # t_edge observed
        t_edge_obs = df["t"].values

        # Define statistics
        if network_stat == "auto":
            stat_mode = "mean_abs_t" if tail == "two-sided" else "mean_t"
        else:
            stat_mode = network_stat  # "mean_abs_t" or "mean_t"

        def agg_stat(tvals):
            if stat_mode == "mean_abs_t":
                return np.mean(np.abs(tvals))
            elif stat_mode == "mean_t":
                return np.mean(tvals)
            else:
                raise ValueError("network_stat must be 'auto', 'mean_abs_t' o 'mean_t'")

        # Observed statistics per pair
        obs_stat = {p: agg_stat(t_edge_obs[net_pair == p]) for p in pairs}

        if method == "parametric":
            # Quick option: combine p of edges with Stouffer, then BH on pairs
            # df["p"] are already the p edgewise
            comb_p = {}
            for p in pairs:
                pv = np.clip(df.loc[net_pair == p, "p"].values, 1e-300, 1-1e-16)
                z = norm.isf(pv)  # one-sided
                zc = z.sum() / np.sqrt(len(z))
                p_comb = norm.sf(zc)
                comb_p[p] = p_comb

            pvals_pairs = np.array([comb_p[p] for p in pairs])

            # FDR correction
            q_pairs, sig_pairs = bh_fdr(pvals_pairs, alpha=alpha)
            add_cols = dict(p_perm=pvals_pairs, q=q_pairs, sig_q=sig_pairs)

        elif method == "perm":
            # Data preparation to recalculate t edgewise at each perm
            # Construct subject x edge arrays for A and B
            E = len(df)
            # Map (i,j) in postion edge
            if paired:
                # Subject x edge differences
                # Construct D from the original data for numerical consistency with parametric
                S = A.shape[0]
                # Stack differences on edges only in the DataFrame
                D = np.empty((S, E), dtype=float)
                k = 0
                for i, j in zip(df["i"].values, df["j"].values):
                    D[:, k] = (A[:, i, j] - B[:, i, j])
                    k += 1

                # t observed recalculated consistently
                mu = D.mean(axis=0)
                sd = D.std(axis=0, ddof=1) + 1e-12
                t_edge_obs_perm = mu / (sd / np.sqrt(S))
                obs_stat = {p: agg_stat(t_edge_obs_perm[net_pair == p]) for p in pairs}

                # Permutation (sign-flip)
                perm_stats = {p: np.empty(n_perm) for p in pairs}
                maxT = np.empty(n_perm) if fwer else None
                for b in range(n_perm):
                    flips = rng.choice([-1, 1], size=S)
                    Db = D * flips[:, None]
                    mu_b = Db.mean(axis=0)
                    sd_b = Db.std(axis=0, ddof=1) + 1e-12
                    t_b  = mu_b / (sd_b / np.sqrt(S))

                    # aggregate per network-pair
                    cur_vals = []
                    for p in pairs:
                        te = t_b[net_pair == p]
                        val = agg_stat(te)
                        perm_stats[p][b] = val
                        cur_vals.append(val)
                    if fwer:
                        maxT[b] = np.max(cur_vals)

            else:
                # Independent: labels shuffled by subject
                S1, S2 = A.shape[0], B.shape[0]
                S = S1 + S2
                # Construct subject x edge matrices for the two groups
                A_e = np.empty((S1, E), dtype=float)
                B_e = np.empty((S2, E), dtype=float)
                k = 0
                for i, j in zip(df["i"].values, df["j"].values):
                    A_e[:, k] = A[:, i, j]
                    B_e[:, k] = B[:, i, j]
                    k += 1

                # recompute observed t (Welch)
                m1 = A_e.mean(axis=0); v1 = A_e.var(axis=0, ddof=1)
                m2 = B_e.mean(axis=0); v2 = B_e.var(axis=0, ddof=1)
                n1 = float(S1); n2 = float(S2)
                se = np.sqrt(v1/n1 + v2/n2) + 1e-12
                t_edge_obs_perm = (m1 - m2) / se
                obs_stat = {p: agg_stat(t_edge_obs_perm[net_pair == p]) for p in pairs}

                # Permutations (label-shuffle)
                X = np.vstack([A_e, B_e])  # S x E
                labels = np.r_[np.zeros(S1, dtype=int), np.ones(S2, dtype=int)]
                perm_stats = {p: np.empty(n_perm) for p in pairs}
                maxT = np.empty(n_perm) if fwer else None

                for b in range(n_perm):
                    rng.shuffle(labels)
                    idx1 = labels == 0
                    idx2 = ~idx1
                    X1 = X[idx1]; X2 = X[idx2]
                    m1 = X1.mean(axis=0); v1 = X1.var(axis=0, ddof=1)
                    m2 = X2.mean(axis=0); v2 = X2.var(axis=0, ddof=1)
                    n1 = float(X1.shape[0]); n2 = float(X2.shape[0])
                    se = np.sqrt(v1/n1 + v2/n2) + 1e-12
                    t_b = (m1 - m2) / se

                    cur_vals = []
                    for p in pairs:
                        te = t_b[net_pair == p]
                        val = agg_stat(te)
                        perm_stats[p][b] = val
                        cur_vals.append(val)
                    if fwer:
                        maxT[b] = np.max(cur_vals)

            # p-values per pair (upper-tail with respect to the chosen statistics)
            pvals_pairs = np.array([
                (1 + (perm_stats[p] >= obs_stat[p]).sum()) / (1 + n_perm)
                for p in pairs
            ])

            if fwer:
                # p_fwer for pair versus maximum distribution
                p_fwer = np.array([
                    (1 + (maxT >= obs_stat[p]).sum()) / (1 + n_perm)
                    for p in pairs
                ])
                add_cols = dict(p_perm=pvals_pairs, q=np.nan, sig_q=np.nan, p_fwer=p_fwer)
            else:
                # FDR within pairs
                q_pairs, sig_pairs = bh_fdr(pvals_pairs, alpha=alpha)
                add_cols = dict(p_perm=pvals_pairs, q=q_pairs, sig_q=sig_pairs, p_fwer=np.nan)

        else:
            raise ValueError("method must be 'parametric' or 'perm'")

        # Merge output summary
        df_pairs = pd.DataFrame({
            "net_pair": pairs,
            "stat": [obs_stat[p] for p in pairs],
            **add_cols
        }).sort_values(["q" if not fwer else "p_fwer",
                        "p_perm" if not fwer else "p_fwer"]).reset_index(drop=True)

        df_summary = df_summary.merge(df_pairs, on="net_pair", how="left")

    return df, df_summary


def compare_state_metrics(metric_by_condition, condA, condB, *, paired=True,
                          test='ttest', alpha=0.05):
    """
    metric_by_condition: dict {cond: array (n_subj, K)} for occupancy o dwell
    condA, condB: condotions names
    paired: True = same subjects (PRE vs POST)
    test: 'ttest' | 'wilcoxon'
    Ritorna: df with a row per state
    """
    A = np.asarray(metric_by_condition[condA])  # (S, K)
    B = np.asarray(metric_by_condition[condB])  # (S, K)
    assert A.shape[1] == B.shape[1], "different K for different conditions"
    if paired:
        assert A.shape[0] == B.shape[0], "Paired=True requires same number of subject"
    K = A.shape[1]

    rows, pvals = [], []
    for k in range(K):
        x, y = A[:, k], B[:, k]
        if paired:
            if test == 'ttest':
                t, p = ttest_rel(x, y, nan_policy='omit')
                es = cohens_d(x, y)
                stat_name, stat_val = 't', t
            elif test == 'wilcoxon':
                res = wilcoxon(x, y, zero_method='wilcox', alternative='two-sided')
                p = res.pvalue
                es = wilcoxon_rank_biserial(x, y)
                stat_name, stat_val = 'W', res.statistic
            else:
                raise ValueError("test must be 'ttest' or 'wilcoxon'")
        else:
            # independent groups
            t, p = ttest_ind(x, y, equal_var=False, nan_policy='omit')
            # effect size Hedges' g (independent)
            es = effect_size_independent(x, y)
            stat_name, stat_val = 't', t

        pvals.append(p)
        rows.append({
            "state": k,
            "meanA": float(np.mean(x)),
            "meanB": float(np.mean(y)),
            "diff_mean": float(np.mean(x) - np.mean(y)),
            stat_name: float(stat_val),
            "p": float(p),
            "effect_size": float(es),
        })

    df = pd.DataFrame(rows)
    q, sigmask = bh_fdr(df["p"].values, alpha=alpha)
    df["q"] = q
    df["sig"] = sigmask
    return df.sort_values(["q","p","state"]).reset_index(drop=True)



# ------------------------ Correlation/Connectivity analysis ------------------------

def global_FC_measure(vec_window, func='mean'):
    """
    vec_window: (W, N*(N-1)/2) in Fisher-z
    Returns g_t: (W,) = average (upper-tri) per window
    """
    return vec_window.mean(axis=1) if func=='mean' else np.median(vec_window, axis=1)

def summarize_series(g):
    """
    g: (W,)
    Return: mu, sd, vol, slope
    - mu: mean
    - sd: std (ddof=1)
    - vol: median(|Δg|)
    - slope: linear regression g ~ t
    """
    g = np.asarray(g, float).ravel()
    mu = float(np.mean(g))
    sd = float(np.std(g, ddof=1)) if len(g) > 1 else 0.0
    vol = float(np.median(np.abs(np.diff(g)))) if len(g) > 1 else 0.0 # volatility, frame-to-frame
    t = np.arange(len(g))
    slope = float(linregress(t, g).slope) if len(g) > 1 else 0.0
    return mu, sd, vol, slope

def global_metrics_per_subject(dfc_by_condition):
    """
    dfc_by_condition: {cond: [ (W,E)_subj0, (W,E)_subj1, ... ] }
    Returns:
      - series_by_cs: {cond: {subj_idx: g_t (W,)}}
      - df_metrics: DataFrame with columns [condition, subj, mu, sd, vol, slope]
    """
    series_by_cs = {}
    rows = []
    for cond, subj_list in dfc_by_condition.items():
        ind = {}
        for s_idx, vec_window in enumerate(subj_list, start=1):
            g = global_FC_measure(vec_window)   # global connectivity level, mean FC in z-space
            ind[s_idx] = g
            mu, sd, vol, slope = summarize_series(g)
            rows.append(dict(condition=cond, subj=s_idx, mu=mu, sd=sd, vol=vol, slope=slope))
        series_by_cs[cond] = ind
    df_metrics = pd.DataFrame(rows)
    return series_by_cs, df_metrics


# ------------------------ Aggregate by network ------------------------

def build_edge_masks_by_netpair(roi_net_masks, net_mask_names):
    """
    roi_net_masks: 
    net_mask_names:
    Retuns:
      - masks: dict { "A - B": boolean (N*(N-1)/2) mask su upper-tri }
      - pairs: list of pair's names
    """
    net_names = list(net_mask_names.keys())
    masks = {}
    for i, a in enumerate(net_names, start=0):
        mask_a = roi_net_masks[net_mask_names[a]]
        for b in net_names[i:]:
            name = "-".join(sorted([str(a), str(b)]))
            mask_b = roi_net_masks[net_mask_names[b]]
            combined_mask = mask_a | mask_b # (N*(N-1)/2,)
            masks[name] = combined_mask
    return masks, list(masks.keys())

def netpair_series(vec_windows, mask, func='mean'):
    """
    vec_windows: (W,N*(N-1)/2), mask: (N*(N-1)/2) bool upper-tri
    Returns: average series per window (W,) for each network pair
    """
    W = vec_windows.shape[0]    # number of windows
    N = mask.shape[0]
    vals = vec_windows[:, (mask.reshape(N,1)*mask)[np.triu_indices(N, k=1)]] # (W, E_pair)
    if vals.size:
        if func=='mean':
            return vals.mean(axis=1)  
        elif func=='median': 
            return np.median(vals, axis=1)
        elif type(func) == function:
            return func(vals, axis=1)
        else:
            raise ValueError(f'func must be "mean", "median" or a callable function')
    else:
        return np.zeros((W,))

def netpair_metrics_per_subject(dfc_by_condition, roi_net_masks, net_mask_names, masks=None, **kwargs):
    """
    For each subject and condition, compute the series for each network pair and their summaries.
    Returns:
      - series_by_cs_pair: { (cond, subj_idx, net_pair): g_t (W,) }
      - df_metrics: DataFrame [condition, subj, net_pair, mu, sd, vol, slope]
    """
    if masks is not None:
        pass
    else: 
        masks, _ = build_edge_masks_by_netpair(roi_net_masks, net_mask_names)
    series_by_cs_pair = {}
    rows = []
    for cond, subj_list in dfc_by_condition.items():
        net_pairs = {}
        for pair_name, mask in masks.items():
            ind = {}
            for s_idx, vec_window in enumerate(subj_list, start=1):
                g = netpair_series(vec_window, mask, **kwargs)  # kwargs: e.g. func='median'
                ind[s_idx] = g
                mu, sd, vol, slope = summarize_series(g)
                rows.append(dict(condition=cond, subj=s_idx, net_pair=pair_name,
                                 mu=mu, sd=sd, vol=vol, slope=slope))
            net_pairs[pair_name] = ind
        series_by_cs_pair[cond] = net_pairs
    df_metrics = pd.DataFrame(rows)
    return series_by_cs_pair, df_metrics

# ------ Paired and mixed model REAL×TIME ------

def paired_tests_on_global(df_metrics, condA, condB, metrics=("mu","sd","vol","slope"), test="ttest", alpha=0.05):
    """
    df_metrics: from global_metrics_per_subject
    Returns dataframe with test paired PRE vs POST for each metric.
    """
    A = df_metrics[df_metrics["condition"]==condA].sort_values("subj")
    B = df_metrics[df_metrics["condition"]==condB].sort_values("subj")
    assert len(A)==len(B) and np.all(A["subj"].values==B["subj"].values), "Misaligned subjects"

    rows = []
    for m in metrics:
        x = A[m].values; y = B[m].values
        if test == "ttest":
            stat, p = ttest_rel(x, y, nan_policy="omit")
            test_name = "t"
        else:   
            res = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
            stat, p = res.statistic, res.pvalue
            test_name = "W"
        rows.append(dict(metric=m, stat=float(stat), p=float(p),
                         meanA=float(np.mean(x)), meanB=float(np.mean(y)), 
                         diff=float(np.mean(x)-np.mean(y)), sig=p<=alpha))
    return pd.DataFrame(rows).sort_values("p").reset_index(drop=True)

# --- Mixed effects: interazione REAL×TIME ---

def build_longform_for_mixed(df_metrics, time_map, stim_map):
    """
    df_metrics: [condition, subj, mu, sd, vol, slope]
    time_map: dict {condition -> "PRE"/"POST"}
    stim_map: dict {condition -> "REAL"/"SHAM"}
    Returns long DF with columns: subj, TIME, STIM, metric, value
    """
    rows = []
    for _, r in df_metrics.iterrows():
        cond = r["condition"]; subj = r["subj"]
        TIME = time_map[cond]; STIM = stim_map[cond]
        for metric in ("mu","sd","vol","slope"):
            rows.append(dict(subj=subj, TIME=TIME, STIM=STIM, metric=metric, value=float(r[metric])))
    return pd.DataFrame(rows)

def fit_mixed_interaction(df_long, metric_name, center_time=True):
    """
    df_long: columns [subj, TIME, STIM, metric, value]
    Filter the metric and MixedLM estimate: value ~ TIME*STIM + (1|subj)
    Returns statsmodels summary.
    """
    d = df_long[df_long["metric"]==metric_name].copy()
    d["TIME"] = pd.Categorical(d["TIME"], categories=["PRE","POST"])
    d["STIM"] = pd.Categorical(d["STIM"], categories=["SHAM","REAL"])
    # Numeric indicator for POST (0=PRE, 1=POST)
    d["TIME_post"] = (d["TIME"] == "POST").astype(float)
    if center_time:
        # cetering helps recude the correlation between random intercept and random slope
        d["TIME_post_c"] = d["TIME_post"] - 0.5
        re_form = "~TIME_post_c"
    else:
        re_form = "~TIME_post"

    model = smf.mixedlm(
        "value ~ TIME * STIM",  # interction (diff-in-diff)
        data=d,
        groups=d["subj"],
        re_formula=re_form      # <-- random slope on TIME_post(_c)
    )
    res = model.fit(reml=False, method="lbfgs", maxiter=2000, disp=False)
    return res  # .summary()


# ---------- Multi-layer graph metrics ----------

def flexibility(labels):
    """
    labels: (W, N)
    Ritorna: flex (N,), frazione di cambi stato su W-1 transizioni.
    """
    W, N = labels.shape
    if W < 2: return np.zeros(N)
    changes = (labels[1:] != labels[:-1])  # (W-1, N) boolean
    return changes.mean(axis=0)  # per nodo

def promiscuity(labels):
    W, N = labels.shape
    pr = np.zeros(N, float)
    for n in range(N):
        pr[n] = len(np.unique(labels[:, n])) / max(1, W)
    return pr

def effective_num_communities(labels: np.ndarray) -> int:
    """
    K_eff: numero di community effettivamente presenti lungo tutte le finestre.
    labels: (W, N) etichette intere (una partizione per finestra).
    """
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError("labels deve essere (W, N)")
    return int(np.unique(labels).size)

def dispersity(labels: np.ndarray) -> np.ndarray:
    """
    Dispersity/Promiscuity per nodo n: (# di community diverse visitate da n) / K_eff, in [0,1].
    labels: (W, N)
    return: (N,) valori per nodo
    """
    labels = np.asarray(labels)
    W, N = labels.shape
    K_eff = effective_num_communities(labels)
    if K_eff == 0:
        return np.zeros(N, dtype=float)

    disp = np.zeros(N, dtype=float)
    for n in range(N):
        disp[n] = np.unique(labels[:, n]).size / K_eff
    return disp    

def module_allegiance_over_time(labels):
    """
    labels: (W, N) singola soluzione → allegiance media su W
    Ritorna: (N, N)
    """
    W, N = labels.shape
    M = np.zeros((N, N), float)
    for w in range(W):
        c = labels[w]
        M += (c[:, None] == c[None, :]).astype(float)
    M /= W
    return M
