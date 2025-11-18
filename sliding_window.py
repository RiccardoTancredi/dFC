import numpy as np
import pandas as pd
from tqdm import tqdm

def z_score_normalize(data, axis=0, ddof=0):
    mean = np.mean(data, axis=axis)
    std = np.std(data, axis=axis, ddof=ddof)
    return (data - mean) / std

def fisher_r_to_z(r):
    r = np.clip(r, -0.999999, 0.999999)
    # In z-space, for i.i.d. samples, the distribution is closer to a normal distribution 
    # with variance ≈ 1/(L−3), r-independent: variance stabilization
    return 0.5 * np.log((1 + r) / (1 - r)) # atanh(r)

def corrcoef_weighted(X, w=None):
    # X: (L, N); w: (L,) normalized (sum to 1). Returns (N, N)
    L, N = X.shape
    if w is None:
        Xc = X - X.mean(axis=0, keepdims=True)
        cov = (Xc.T @ Xc) / (L - 1)
        # cov = np.einsum('tn,tm->nm', Xc, Xc) / (L - 1)
        # print(f'Is everything finite? {"YES" if np.isfinite(cov).all() else "NO"}')
    else:
        w = w / w.sum() # sum to 1
        mu = (w[:, None] * X).sum(axis=0, keepdims=True)
        Xc = X - mu
        cov = (Xc * w[:, None]).T @ Xc
        # cov = np.einsum('t,tn,tm->nm', w, Xc, Xc)
        cov /= (1.0 - (w**2).sum())
    sd = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    C = cov / (sd[:, None] * sd[None, :])
    np.fill_diagonal(C, 1.0)
    return C


# Static FC
def compute_static_fc(timeseries, already_z_scored=True, fisher_z=True):
    """
    Compute static FC (ROI x ROI) from time series of shape (T, N).
    
    Parameters
    ----------
    timeseries : array (T, N)
        fMRI time series (T time points, N ROIs).
    fisher_z : bool
        If True, apply Fisher r-to-z transform for better comparability.
    
    Returns
    -------
    FC : array (N, N)
        Static functional connectivity matrix.
    """
    ts = timeseries if already_z_scored else timeseries - timeseries.mean(axis=0, keepdims=True)
    corr = np.corrcoef(ts, rowvar=False)   # (N, N)
    corr = fisher_r_to_z(corr) if fisher_z else corr
    # np.fill_diagonal(corr, 0)
    return corr

# Static for each subject
def static_fc_for_subject(ts, zscore=True, fisher_z=True, ddof=0):
    """
        ts: (T,N) of a subject. Gives back FC (W,E) or (W,N,N)
    """
    X = z_score_normalize(ts, axis=0, ddof=ddof) if zscore else ts
    return compute_static_fc(X, zscore, fisher_z)

def static_fc_for_condition(list_of_subject_TS, **kwargs):
    """
    list_of_subject_TS: list of array (T_i, N); returns list of static FC (per subject).
    kwargs passed to dfc_for_subject.
    """
    return [static_fc_for_subject(ts, **kwargs) for ts in list_of_subject_TS]

# Sliding window FC
def sliding_dfc(ts, win_len, step=1, gaussian=False, sigma=None, fisher_z=False, vectorize=False):
    # ts: (T, N)
    T, N = ts.shape
    if gaussian:
        if sigma is None:
            sigma = win_len / 3.0
        x = np.arange(win_len) - (win_len - 1)/2
        w = np.exp(-(x**2) / (2*sigma**2))
        w = w.astype(np.float32)
    else:
        w = None

    starts = range(0, T - win_len + 1, step)
    iu = np.triu_indices(N, k=1)
    out = []
    for s in starts:
        seg = ts[s:s+win_len]
        C = corrcoef_weighted(seg, w)
        if fisher_z:
            # Check first C std
            C = fisher_r_to_z(C)
        out.append(C[iu] if vectorize else C)
    # W = ⌊sT−L​⌋+1 (s = step, L = win_len, T = ts length)
    return np.stack(out, axis=0)  # (W, E) or (W, N, N)


def dfc_for_subject(ts, win_len=10, step=1, gaussian=False, sigma=None,
                    fisher_z=True, vectorize=True, zscore=True, ddof=0):
    """
        ts: (T,N) of a subject. Gives back dFC (W,E) or (W,N,N)
    """
    X = z_score_normalize(ts, axis=0, ddof=ddof) if zscore else ts
    return sliding_dfc(X, win_len, step, gaussian, sigma, fisher_z, vectorize)

def dfc_for_condition(list_of_subject_TS, **kwargs):
    """
        list_of_subject_TS: list of array (T_i, N); returns list of dFC (per subject).
        kwargs passed to dfc_for_subject (window parameters, gaussian, etc.).
    """
    return [dfc_for_subject(ts, **kwargs) for ts in list_of_subject_TS]


# -------- dFC per (win_len, step) and autocorrelations within adjacent windows --------
def mean_adjacent_autocorr(V):
    """
    V: (W,E) vettori di FC per finestra (Fisher-z).
    Calcola la correlazione (Pearson) tra finestre adiacenti e ne fa la media.
    """
    if V is None or V.shape[0] < 2:
        return np.nan
    W = V.shape[0]
    ac = []
    for t in range(W - 1):
        v1 = V[t]
        v2 = V[t + 1]
        # correlazione tra vettori (centra e normalizza)
        v1c = v1 - v1.mean()
        v2c = v2 - v2.mean()
        denom = (np.linalg.norm(v1c) * np.linalg.norm(v2c) + 1e-12)
        r = float((v1c @ v2c) / denom)
        ac.append(r)
    return float(np.mean(ac))


# -------- GRID SEARCH: for each win_len e step=1..win_len --------
def grid_autocorr(ts_by_condition, win_lens, step_max_mode="win_len", 
                  gaussian=True,
                  sigma=None, 
                  fisher_z=True,
                  vectorize=True,
                  zscore=True
):
    """
    ts_by_condition: {cond: [Ts_subj (T,N), ...]}  z-scored per ROI.
    win_lens: iterable di window length (in samples/TR).
    step_max_mode: "win_len" => for each win_len evaluate step=1..win_len,
                   or a fixed int, or a dict {win_len: max_step}.
    Return df with columns: condition, subj, win_len, step, ac_mean
    and df_mean (average on subjects per condition) and df_grand (average on cond).
    """
    rows = []
    for cond, subj_list in tqdm(ts_by_condition.items(), colour='blue'):
        for s_idx, Ts in enumerate(subj_list):
            T, N = Ts.shape
            for L in win_lens:
                if L < 5 or L > T-1:
                    continue
                # define step range
                if step_max_mode == "win_len":
                    max_step = L
                elif isinstance(step_max_mode, int):
                    max_step = min(step_max_mode, L)
                elif isinstance(step_max_mode, dict):
                    max_step = min(step_max_mode.get(L, L), L)
                else:
                    max_step = L
                for step in range(1, max_step + 1):
                    V = dfc_for_subject(Ts, win_len=L, step=step, 
                                        gaussian=gaussian, sigma=sigma,
                                        fisher_z=fisher_z, vectorize=vectorize, 
                                        zscore=zscore)
                    W = V.shape[0]
                    if not (np.isfinite(W)):
                        print(f'{W=}')
                    ac = mean_adjacent_autocorr(V)
                    ess = (W * (1 - ac) / (1 + ac)) if (np.isfinite(ac)) else np.nan
                    rows.append(dict(condition=cond, subj=s_idx, win_len=L, step=step,
                                     ac_mean=ac, W=W, ESS_AR1_subj=ess))
    df = pd.DataFrame(rows)
    if df.empty:
        return df, None, None

    # media su soggetti (per condizione)
    g = df.groupby(["condition", "win_len", "step"], as_index=False)
    df_cond_mean = g.agg(ac_mean=("ac_mean","mean"),
                         W_mean=("W","mean"),
                         ESS_AR1_mean=("ESS_AR1_subj","mean"))
    # media across condizioni
    g2 = df_cond_mean.groupby(["win_len","step"], as_index=False)
    df_grand = g2.agg(ac_mean_all=("ac_mean","mean"),
                      W_mean_all=("W_mean","mean"),
                      ESS_AR1_all=("ESS_AR1_mean","mean"))
    return df, df_cond_mean, df_grand


#  ----- Helpers from extracting networks to perform sliding_window.py -----
def load_roi_net_from_lut(yeo_atlas_path, tian_atlas_path, combine_networks=False):
    """
    Parse a LUT file (e.g., Schaefer 17-networks) and build ROI-to-network mapping.

    Parameters
    ----------
    yeo_atlas_path : str
        Path to LUT file (2 lines per ROI: name, then ID+RGBA).
    tian_atlas_path : str
        Subcortex Atlas (1 line per ROI: name).

    Returns
    -------
    roi_names : list of str
        List of ROI names.
    roi_net_masks : np.ndarray (R, N)
        Array containing masks mapping each ROI index to a network.
        R is number of ROIs, N is number of networks.
    """
    with open(f'{yeo_atlas_path}', 'r') as f:
        yeo_networks = [line.strip().split('_')[2] for line in f.readlines()[::2]]
        
    # print(f'Yeo Networks: {np.unique(yeo_networks)}, {len(np.unique(yeo_networks))} unique Yeo networks.')

    # Add subcorticals regions, from atlast, to Yeo's networks
    with open(f'{tian_atlas_path}', 'r') as f:
        tian_subcorticals = [line.strip().split('-')[0] for line in f.readlines()]

    roi_to_network = yeo_networks + tian_subcorticals

    ROIs = len(roi_to_network)
    if not combine_networks:
        unique_networks = np.unique(roi_to_network)
    else:
        nets = []
        for net in roi_to_network:
            if net[-1] == 'A' or net[-1] == 'B' or net[-1] == 'C':
                nets.append(net[:-1])
            else:
                nets.append(net)
        unique_networks = np.unique(nets)
        roi_to_network = np.array(nets)

    N = len(unique_networks)
    roi_net_masks = []
    net_mask_names = {}
    for numb, n in enumerate(unique_networks):
        net_mask_names[str(n)] = numb
        if not combine_networks:
            roi_net_masks.append(np.array([1 if net == n else 0 for net in roi_to_network], dtype=bool))
        else:
            roi_net_masks.append(np.array([1 if (net == n or net[:-1] == n) else 0 for net in roi_to_network], dtype=bool))
    
    assert len(roi_net_masks) == N, "Something went wrong in parsing the networks."
    
    print(f'Networks: {unique_networks}, {N} unique networks.')
    
    return roi_to_network, roi_net_masks, net_mask_names
