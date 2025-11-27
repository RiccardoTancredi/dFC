import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def cluster_states(dfc_vec, n_components=50, k=4, random_state=0):
    # df_vec: (W, E) for one subject or concatenated over multiple subjects
    X = dfc_vec - dfc_vec.mean(axis=0, keepdims=True)
    pca = PCA(n_components=n_components, random_state=random_state)
    Z = pca.fit_transform(X)
    km = KMeans(n_clusters=k, n_init=20, random_state=random_state)
    labels = km.fit_predict(Z)
    return labels, pca, km, X.mean(axis=0, keepdims=True)

def fit_kmeans_on_dfc(list_of_dfc_vec, n_components=50, k=4, random_state=0):
    """
        list_of_dfc_vec: list of (W_i, E) for multiple subjects or concatenated over multiple subjects 
        Returns: pca, kmeans, global_mean
    """
    X = np.concatenate(list_of_dfc_vec, axis=0)            # (sum W_i, E)
    Xc = X - X.mean(axis=0, keepdims=True)
    pca = PCA(n_components=n_components, random_state=random_state)
    Z = pca.fit_transform(Xc)
    km = KMeans(n_clusters=k, n_init=50, random_state=random_state)
    km.fit(Z)
    # X.mean(axis=0, keepdims=True)=global mean of all dFC for centering: 
    # represents the average connectivity aggregated over all subjects and conditions
    return pca, km, X.mean(axis=0, keepdims=True) 


def predict_states(dfc_vec, pca, km, global_mean):
    """
        dfc_vec: (W,E) of a subject; pca+km already fitted for labels prediction
        Returns: labels (W,)
    """
    Z = pca.transform(dfc_vec - global_mean)
    labels = km.predict(Z)
    return labels


def state_metrics(labels, K=None):
    """
    labels: (W,) state sequence
    K: total number of states (e.g., km.n_clusters). If None, uses max(label)+1.
    Returns:
      occ:   (K,) fractional occupancy
      dwell: (K,) mean dwell (in windows)
      trans: (K, K) row-stochastic transition matrix (rows with no counts -> zeros)
    """
    labels = np.asarray(labels)
    W = len(labels)
    if K is None:
        K = int(labels.max()) + 1

    # Occupancy (ensure fixed length K)
    occ = np.zeros(K, dtype=float)
    for k in range(K):
        occ[k] = (labels == k).mean()

    # Mean dwell (runs) per state
    dwell = np.zeros(K, dtype=float)
    # collect run lengths per state
    run_lengths = [[] for _ in range(K)]
    prev = labels[0]
    run = 1
    for t in range(1, W):
        if labels[t] == prev:
            run += 1
        else:
            if prev < K:
                run_lengths[prev].append(run)
            prev = labels[t]
            run = 1
    if prev < K:
        run_lengths[prev].append(run)
    for k in range(K):
        rl = run_lengths[k]
        dwell[k] = np.mean(rl) if rl else 0.0

    # Transition counts -> probabilities (fixed KxK)
    trans_counts = np.zeros((K, K), dtype=float)
    for i in range(W - 1):
        a, b = labels[i], labels[i + 1]
        if a < K and b < K:
            trans_counts[a, b] += 1
    with np.errstate(invalid='ignore', divide='ignore'):
        trans = trans_counts / trans_counts.sum(axis=1, keepdims=True)
    trans[np.isnan(trans)] = 0.0

    return occ, dwell, run_lengths, trans


def state_metrics_full(labels, K=None, step=1, TR=None):
    """
    Compute multiple temporal metrics of state sequences.

    Parameters
    ----------
    labels : array-like, shape (W,)
        State labels for each sliding window of a subject.
    step : int, optional (default=1)
        Step size between windows, in TRs. Used only if TR is provided.
    TR : float or None, optional (default=None)
        Repetition time in seconds. If given, dwell/run metrics are also returned in seconds.

    Returns
    -------
    out : dict
        Dictionary containing:
          - "occupancy"        : fraction of time spent in each state (K,)
          - "n_runs"           : number of contiguous blocks (runs) per state (K,)
          - "mean_dwell"       : average run length in windows per state (K,)
          - "median_dwell"     : median run length in windows per state (K,)
          - "run_lengths"      : list of arrays, each with run lengths for one state
          - "trans_counts"     : raw transition counts matrix (K,K)
          - "trans_matrix"     : row-stochastic transition probability matrix (K,K)
          - "dwell_from_Pkk"   : expected dwell time from Markov estimate 1/(1-P_kk) (K,)
          - If TR is provided, also returns time in seconds:
              * "mean_dwell_sec"
              * "median_dwell_sec"
              * "run_lengths_sec"
              * "dwell_from_Pkk_sec"
    """
    labels = np.asarray(labels)
    W = len(labels)
    if K is None:
        K = int(labels.max()) + 1

    # Fraction of time in each state
    occupancy = np.array([(labels == k).mean() for k in range(K)])

    # Collect contiguous run lengths for each state
    run_lengths = [[] for _ in range(K)]
    prev = labels[0]
    run = 1
    for t in range(1, W):
        if labels[t] == prev:
            run += 1
        else:
            run_lengths[prev].append(run)
            prev = labels[t]
            run = 1
    run_lengths[prev].append(run)

    # Number of runs, mean and median dwell time
    n_runs       = np.array([len(rl) for rl in run_lengths])
    mean_dwell   = np.array([np.mean(rl) if rl else 0.0 for rl in run_lengths])
    median_dwell = np.array([np.median(rl) if rl else 0.0 for rl in run_lengths])

    # Transition counts and transition probability matrix
    trans_counts = np.zeros((K, K), dtype=float)
    for i in range(W-1):
        trans_counts[labels[i], labels[i+1]] += 1
    with np.errstate(invalid='ignore', divide='ignore'):
        trans_matrix = trans_counts / trans_counts.sum(axis=1, keepdims=True)
    trans_matrix[np.isnan(trans_matrix)] = 0.0

    # Expected dwell time from Markov model (1/(1 - P_kk))
    with np.errstate(divide='ignore'):
        dwell_from_Pkk = np.where(1 - np.diag(trans_matrix) > 0,
                                  1.0 / (1.0 - np.diag(trans_matrix)),
                                  np.inf)

    out = {
        "occupancy": occupancy,
        "n_runs": n_runs,
        "mean_dwell": mean_dwell,
        "median_dwell": median_dwell,
        "run_lengths": [np.array(rl, dtype=int) for rl in run_lengths],
        "trans_counts": trans_counts,
        "trans_matrix": trans_matrix,
        "dwell_from_Pkk": dwell_from_Pkk,
    }

    # Optional: convert dwell/run metrics into seconds
    if TR is not None:
        factor = step * TR
        out["mean_dwell_sec"]     = mean_dwell * factor
        out["median_dwell_sec"]   = median_dwell * factor
        out["run_lengths_sec"]    = [rl * factor for rl in out["run_lengths"]]
        out["dwell_from_Pkk_sec"] = dwell_from_Pkk * factor

    return out