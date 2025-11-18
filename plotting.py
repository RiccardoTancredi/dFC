import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from scipy.stats import ttest_ind, ttest_rel
import seaborn as sns
import pandas as pd
from stats import cohens_d
from itertools import combinations
from stats import bh_fdr


# -------- Plot: step → autocorr for each win_len (one line per condition) --------
def plot_autocorr_vs_step(df_cond_mean, win_lens=None, conditions=None, save=False,
                          TR=1., title="Mean adjacent-window autocorrelation"):
    """
    df_cond_mean: mean output per condition (columns: condition, win_len, step, ac_mean)
    """
    if df_cond_mean is None or df_cond_mean.empty:
        print("No plot.")
        return
    if win_lens is None:
        win_lens = sorted(df_cond_mean["win_len"].unique())
    if conditions is None:
        conditions = sorted(df_cond_mean["condition"].unique())

    n_plots = len(win_lens)
    n_cols = 3
    n_rows = int(np.ceil(n_plots / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows), sharey=True)
    axes = np.array(axes).reshape(n_rows, n_cols)

    colors = {c: plt.rcParams['axes.prop_cycle'].by_key()['color'][i % 10] for i, c in enumerate(conditions)}

    for idx, L in enumerate(win_lens):
        r, c = divmod(idx, n_cols)
        ax = axes[r, c]

        sub = df_cond_mean[df_cond_mean["win_len"] == L]
        for cond in conditions:
            ss = sub[sub["condition"] == cond]
            ax.plot(ss["step"] * TR, ss["ac_mean"], marker='o', linewidth=1.5, 
                    label=' '.join(cond.split('_')), color=colors[cond])
        ax.set_title(f"Window length = {L}", fontsize=20)
        ax.set_xlabel(r"step $\times$ TR (s)", fontsize=14)
        ax.set_xticks([i-1 for i in ss["step"][::10]], [i-1 for i in ss["step"][::10]])
        if c == 0:
            ax.set_ylabel(r"mean corr(window$_t$, window$_{t+1}$)", fontsize=20)
        ax.grid(True, alpha=0.3)

    # Rimuove gli assi vuoti (se non multiplo di 3)
    for idx in range(n_plots, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        fig.delaxes(axes[r, c])

    # Legenda fuori a destra
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, bbox_to_anchor=(0.9, 0.8), loc='upper left')

    fig.suptitle(title, y=0.98, fontsize=35)
    fig.tight_layout(rect=[0, 0, 0.9, 0.95])
    fig.subplots_adjust(hspace=0.3, wspace=0.25)

    if save:
        plt.savefig(f'imgs/{"_".join(title.split(" "))}.png', dpi=300, bbox_inches='tight')
    plt.show()

# -------- Average heatmap across conditions --------    
def plot_heatmap(df_grand, value_col="ESS_AR1_all", TR=1., save=False,
                     title="Average ESS (AR1) across conditions"):
    df_grand[['win_len', 'step']] *= TR
    piv = df_grand.pivot(index="win_len", columns="step", values=value_col).sort_index()
    X, Y = np.meshgrid(piv.columns.values, piv.index.values)
    plt.figure(figsize=(8, 4.5))
    pc = plt.pcolormesh(X, Y, piv.values, shading='nearest')
    plt.colorbar(pc, label=' '.join(value_col.split('_')))
    plt.xlabel(r"step $\times$ TR (s)"); plt.ylabel(r"Window length $\times$ TR (s)", fontsize=20)
    # plt.xticks(piv.columns.values); 
    plt.yticks(piv.index.values)
    plt.title(title); plt.tight_layout()
    if save:
        plt.savefig(f'imgs/{"_".join(title.split(" "))}.png', dpi=300)
    plt.show()


def plot_state_centroids(km, pca, global_mean, n_rois):
    centroids = km.cluster_centers_ @ pca.components_ + global_mean
    for k, c in enumerate(centroids):
        mat = np.zeros((n_rois, n_rois))
        iu = np.triu_indices(n_rois, 1)
        mat[iu] = c
        mat += mat.T
        plt.figure()
        plt.imshow(mat, vmin=-1, vmax=1, cmap="coolwarm")
        plt.title(f"State {k+1} centroid")
        plt.colorbar()

def barcode_plots(labels, title="State sequence (example subject)"):
    plt.figure(figsize=(10, 2))
    plt.imshow(labels[None, :], aspect="auto",
               cmap="tab20", vmin=0, vmax=labels.max())
    plt.yticks([])
    plt.xlabel("Window")
    plt.title(title)
    plt.colorbar(ticks=range(labels.max()+1), label="State")
    plt.show()

def plot_fractional_occupancy(occ_dict):
    """
    occ_dict = {
        "SHAM_PRE": occ_SHAM_PRE,  # shape (n_subj, K)
        "SHAM_POST": ...
        ...
    }
    """
    rows = []
    for cond, occ in occ_dict.items():
        n_subj, K = occ.shape
        for i in range(n_subj):
            for k in range(K):
                rows.append({"Condition": cond, "State": k +
                            1, "Occupancy": occ[i, k]})
    df = pd.DataFrame(rows)
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df, x="State", y="Occupancy", hue="Condition")
    plt.title("Fractional occupancy across conditions")
    plt.show()


# ---------- Helpers ----------

def triu_to_sym_matrix(vec, n_rois):
    """
    Reconstruct full symmetric NxN matrix from a vectorized upper triangle (k=1).
    vec: (E,) with E = n_rois*(n_rois-1)//2
    """
    mat = np.zeros((n_rois, n_rois), dtype=float)
    iu = np.triu_indices(n_rois, 1)
    mat[iu] = vec
    mat += mat.T
    np.fill_diagonal(mat, 1.0)
    return mat


def centroids_back_to_full(kmeans, pca, global_mean, n_rois):
    """
    Project k-means centroids from PCA space back to full vectorized FC (upper triangle),
    then to symmetric NxN matrices.

    Returns
    -------
    mats : list of (n_rois, n_rois) arrays, one per state.
    """
    # centroids in PCA space -> original vectorized space
    vecs = kmeans.cluster_centers_ @ pca.components_ + global_mean
    mats = [triu_to_sym_matrix(v, n_rois) for v in vecs]
    return mats


# ---------- 1) State centroids (FC patterns) ----------

def plot_state_centroids(kmeans, pca, global_mean, n_rois, vmin=-1.0, vmax=1.0):
    """
    Plot k-means state centroids as NxN FC heatmaps.
    """
    mats = centroids_back_to_full(kmeans, pca, global_mean, n_rois)
    K = len(mats)
    for k in range(K):
        plt.figure()
        plt.imshow(mats[k], vmin=vmin, vmax=vmax, aspect='equal')
        plt.colorbar()
        plt.title(f"State {k+1} centroid")
        plt.xlabel("ROI")
        plt.ylabel("ROI")
        plt.tight_layout()
    plt.show()

# ---------- 2) Barcode (state sequence for one subject) ----------

def _discrete_cmap(K, base='tab10'):
    """Return a ListedColormap with exactly K distinct colors."""
    base_cmap = mpl.colormaps[base] # .cm.get_cmap(base)
    if hasattr(base_cmap, 'colors') and len(base_cmap.colors) >= K:
        colors = base_cmap.colors[:K]
    else:
        colors = base_cmap(np.linspace(0, 1, K))
    return ListedColormap(colors)

def plot_state_barcode(labels, K=None, title="State sequence", 
                       state_names=None, colors=None, legend_loc="upper center"):
    """
    Plot a single subject's state sequence as a 1xW barcode with a discrete legend.

    Parameters
    ----------
    labels : array-like (W,)
        State labels for each window (integers starting at 0).
    K : int or None
        Total number of states (useful to keep colors consistent across subjects).
        If None, inferred as max(labels)+1.
    state_names : list[str] or None
        Names to show in the legend; defaults to ["State 0", ..., "State K-1"].
    colors : list or None
        Optional list of K color specs (e.g., hex strings); overrides the default cmap.
    legend_loc : str
        Legend location (e.g. "upper center", "lower center", "center right", ...).
    """
    labels = np.asarray(labels).astype(int)
    if K is None:
        K = int(labels.max()) + 1

    # build a discrete colormap with exactly K colors
    if colors is not None:
        cmap = ListedColormap(colors[:K])
    else:
        cmap = _discrete_cmap(K, base='tab10')  # or 'tab20', 'Set3', 'Dark2'

    if state_names is None:
        state_names = [f"State {k+1}" for k in range(K)]

    plt.figure(figsize=(10, 1.8))
    im = plt.imshow(labels[None, :],
                    aspect="auto",
                    interpolation="nearest",
                    cmap=cmap,
                    vmin=0, vmax=K-1)
    plt.yticks([])
    plt.xlabel("Window index")
    plt.title(title)

    # legend with one patch per state (no colorbar)
    handles = [mpl.patches.Patch(color=cmap(k), label=state_names[k]) for k in range(K)]
    plt.legend(handles=handles,
               loc=legend_loc,
               bbox_to_anchor=(1.3, 1.) if "center" in legend_loc else None,
               ncol=min(K, 6), frameon=False)

    plt.tight_layout()
    plt.show()


# ---------- 3) Fractional occupancy per condition (boxplots) ----------

def plot_fractional_occupancy_boxplots(occ_by_condition, order=None, colors=None, showfliers=True,
                                       title="Fractional occupancy by condition", cond_pairs=None, sig_df_map=None,
                                       sig_df=None, cond_pair=None, alpha=0.05, connector_kwargs=None):
    """
    occ_by_condition: dict {condition_name: occ_matrix}, occ_matrix shape = (n_subj, K)
    order: optional list with the order of conditions on the legend (and color mapping)
    colors: optional dict {condition_name: color}; if None uses Matplotlib cycle
    showfliers: if True, shows whiskers in the plot: they live in the range given by 1.5 x IQR (where IQR = Q3 - Q1)
    title: optional plot title
    sig_df: DataFrame con colonne almeno ['state','q'] (es. df_occ_real)
    cond_pair: tuple/list (condA, condB) da collegare (es. ("REAL_PRE","REAL_POST"))
    alpha: soglia FDR per disegnare i connettori
    connector_kwargs: dict matplotlib per stile linea (linestyle, linewidth, color, alpha)
    """
    conditions = order if order is not None else list(occ_by_condition.keys())
    K = list(occ_by_condition.values())[0].shape[1]

    # default colors if not provided
    if colors is None:
        prop_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
        colors = {cond: prop_cycle[i % len(prop_cycle)]
                  for i, cond in enumerate(conditions)}

    width = 0.8 / max(1, len(conditions))
    plt.figure(figsize=(2.4 + 2*K, 4))

    legend_handles = []
    positions_by_condition = {}

    for c_idx, cond in enumerate(conditions):
        occ = occ_by_condition[cond]  # (n_subj, K)
        positions = np.arange(K) + (c_idx - (len(conditions)-1)/2)*width
        positions_by_condition[cond] = positions

        data = [occ[:, k] for k in range(K)]
        bp = plt.boxplot(
            data, positions=positions, widths=width, manage_ticks=False,
            patch_artist=True, showfliers=showfliers    # needed to color boxes
        )
        # color boxes/medians for this condition
        for box in bp['boxes']:
            box.set_facecolor(colors[cond])
            box.set_alpha(0.35)
            box.set_edgecolor(colors[cond])
        for med in bp['medians']:
            med.set_color(colors[cond])
            med.set_linewidth(2.0)
        for whisk in bp['whiskers']:
            whisk.set_color(colors[cond])
        for cap in bp['caps']:
            cap.set_color(colors[cond])
        for flier in bp.get('fliers', []):
            flier.set_markerfacecolor(colors[cond])
            flier.set_markeredgecolor(colors[cond])
            flier.set_alpha(0.4)

        legend_handles.append(
            Patch(facecolor=colors[cond], edgecolor=colors[cond], alpha=0.35, label=cond))

    plt.xticks(np.arange(K), [f"State {k+1}" for k in range(K)])
    plt.ylabel("Fractional occupancy")
    plt.title(title)
    plt.legend(handles=legend_handles, title="Condition",
               bbox_to_anchor=(1, 1),
               frameon=False, ncol=1)

    pairs = []
    if cond_pairs is not None:
        pairs.extend([tuple(p) for p in cond_pairs])
    if cond_pair is not None:
        pairs.append(tuple(cond_pair))

    if pairs:
        ax = plt.gca()
        y_min, y_max = ax.get_ylim()
        dy = (y_max - y_min)
        base = y_max + 0.02 * dy
        bump = 0.0

        # connector_kwargs può essere:
        # - dict unico per tutte le coppie (es. {"linestyle":"--", "color":"k"})
        # - dict di dict, mappato per coppia (es. {("REAL_PRE","REAL_POST"):{...}, ("SHAM_PRE","SHAM_POST"):{...}})
        def get_ckw(pair):
            if isinstance(connector_kwargs, dict):
                # se è un dict-of-dicts e ha la coppia:
                if all(isinstance(v, dict) for v in connector_kwargs.values()):
                    return connector_kwargs.get(pair, dict(linestyle='--', linewidth=1.5, color='k', alpha=0.9))
                # altrimenti è un unico stile
                return connector_kwargs
            return dict(linestyle='--', linewidth=1.5, color='k', alpha=0.9)

        for pair in pairs:
            # recupera il df della coppia
            df_cur = None
            if sig_df_map is not None:
                df_cur = sig_df_map.get(pair)
            if df_cur is None:
                df_cur = sig_df  # fallback: usa lo stesso df per tutte le coppie

            # posizioni dei box per le due condizioni della coppia
            if (df_cur is None) or (pair[0] not in positions_by_condition) or (pair[1] not in positions_by_condition):
                continue
            posA = positions_by_condition[pair[0]]
            posB = positions_by_condition[pair[1]]

            ckw = get_ckw(pair)

            for _, row in df_cur.iterrows():
                # Disegna solo se significativo
                is_sig = (("sig" in row and bool(row["sig"])) or
                        ("q" in row and float(row["q"]) <= alpha))
                if not is_sig:
                    continue

                k = int(row["state"])
                x1, x2 = posA[k], posB[k]
                y = base + bump
                ax.plot([x1, x2], [y, y], **ckw)

                # stelle in base alla q
                qv = float(row["q"]) if "q" in row else np.nan
                if np.isfinite(qv):
                    if qv <= 0.001: stars = '***'
                    elif qv <= 0.01: stars = '**'
                    elif qv <= alpha: stars = '*'
                    else: stars = None
                    if stars:
                        ax.text((x1 + x2) / 2.0, y + 0.01 * dy, stars,
                                ha='center', va='bottom', fontsize=9)

                bump += 0.06 * dy  # increase to avoid overlap

        ax.set_ylim(y_min, y_max + bump + 0.08 * dy)

    plt.tight_layout()
    plt.show()


# ---------- 4) Dwell time per condition (boxplots) ----------

def plot_dwell_time_boxplots(dwell_by_condition, in_seconds=False, order=None, colors=None, showfliers=True,
                             title="Dwell time by condition", cond_pairs=None, sig_df_map=None,
                             sig_df=None, cond_pair=None, alpha=0.05, connector_kwargs=None):
    """
    dwell_by_condition: dict {condition_name: dwell_matrix}, shape = (n_subj, K)
    If in_seconds=True, y-label reflects seconds.
    sig_df/cond_pair/alpha/connector_kwargs come sopra.
    """
    conditions = order if order is not None else list(dwell_by_condition.keys())
    K = list(dwell_by_condition.values())[0].shape[1]

    # default colors if not provided
    if colors is None:
        prop_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
        colors = {cond: prop_cycle[i % len(prop_cycle)]
                  for i, cond in enumerate(conditions)}

    width = 0.8 / max(1, len(conditions))
    plt.figure(figsize=(2.4 + 2*K, 4.5))

    legend_handles = []
    positions_by_condition = {}

    for c_idx, cond in enumerate(conditions):
        dwell = dwell_by_condition[cond]
        positions = np.arange(K) + (c_idx - (len(conditions)-1)/2)*width
        positions_by_condition[cond] = positions

        data = [dwell[:, k] for k in range(K)]
        bp = plt.boxplot(
            data, positions=positions, widths=width, manage_ticks=False,
            patch_artist=True, showfliers=showfliers
        )
        # color boxes/medians for this condition
        for box in bp['boxes']:
            box.set_facecolor(colors[cond])
            box.set_alpha(0.35)
            box.set_edgecolor(colors[cond])
        for med in bp['medians']:
            med.set_color(colors[cond])
            med.set_linewidth(2.0)
        for whisk in bp['whiskers']:
            whisk.set_color(colors[cond])
        for cap in bp['caps']:
            cap.set_color(colors[cond])
        for flier in bp.get('fliers', []):
            flier.set_markerfacecolor(colors[cond])
            flier.set_markeredgecolor(colors[cond])
            flier.set_alpha(0.4)

        legend_handles.append(
            Patch(facecolor=colors[cond], edgecolor=colors[cond], alpha=0.35, label=cond))

    plt.xticks(np.arange(K), [f"State {k+1}" for k in range(K)])
    plt.ylabel("Dwell time (s)" if in_seconds else "Dwell time (windows)")
    plt.title(title)
    plt.legend(handles=legend_handles, title="Condition",
               bbox_to_anchor=(1, 1),
               frameon=False, ncol=1)

    pairs = []
    if cond_pairs is not None:
        pairs.extend([tuple(p) for p in cond_pairs])
    if cond_pair is not None:
        pairs.append(tuple(cond_pair))

    if pairs:
        ax = plt.gca()
        y_min, y_max = ax.get_ylim()
        dy = (y_max - y_min)
        base = y_max + 0.02 * dy
        bump = 0.0

        # connector_kwargs può essere:
        # - dict unico per tutte le coppie (es. {"linestyle":"--", "color":"k"})
        # - dict di dict, mappato per coppia (es. {("REAL_PRE","REAL_POST"):{...}, ("SHAM_PRE","SHAM_POST"):{...}})
        def get_ckw(pair):
            if isinstance(connector_kwargs, dict):
                # se è un dict-of-dicts e ha la coppia:
                if all(isinstance(v, dict) for v in connector_kwargs.values()):
                    return connector_kwargs.get(pair, dict(linestyle='--', linewidth=1.5, color='k', alpha=0.9))
                # altrimenti è un unico stile
                return connector_kwargs
            return dict(linestyle='--', linewidth=1.5, color='k', alpha=0.9)

        for pair in pairs:
            # recupera il df della coppia
            df_cur = None
            if sig_df_map is not None:
                df_cur = sig_df_map.get(pair)
            if df_cur is None:
                df_cur = sig_df  # fallback: usa lo stesso df per tutte le coppie

            # posizioni dei box per le due condizioni della coppia
            if (df_cur is None) or (pair[0] not in positions_by_condition) or (pair[1] not in positions_by_condition):
                continue
            posA = positions_by_condition[pair[0]]
            posB = positions_by_condition[pair[1]]

            ckw = get_ckw(pair)

            for _, row in df_cur.iterrows():
                # Draw only if FDR significant
                is_sig = (("sig" in row and bool(row["sig"])) or
                        ("q" in row and float(row["q"]) <= alpha))
                if not is_sig:
                    continue

                k = int(row["state"])
                x1, x2 = posA[k], posB[k]
                y = base + bump
                ax.plot([x1, x2], [y, y], **ckw)

                # stelle in base alla q
                qv = float(row["q"]) if "q" in row else np.nan
                if np.isfinite(qv):
                    if qv <= 0.001: stars = '***'
                    elif qv <= 0.01: stars = '**'
                    elif qv <= alpha: stars = '*'
                    else: stars = None
                    if stars:
                        ax.text((x1 + x2) / 2.0, y + 0.01 * dy, stars,
                                ha='center', va='bottom', fontsize=9)

                bump += 0.06 * dy  # increase to avoid overlap

        ax.set_ylim(y_min, y_max + bump + 0.08 * dy)

    plt.tight_layout()
    plt.show()


# ---------- 5) Transition matrices ----------

# ---------- helper: average row-stochastic transition matrix ----------

def average_transition_matrix(trans_matrices):
    """
    Sum transition COUNTS across subjects and row-normalize.
    trans_matrices: list of (K,K) arrays (can be counts or probabilities).
    Returns averaged row-stochastic (K,K).
    """
    K = trans_matrices[0].shape[0]
    total = np.zeros((K, K), dtype=float)
    for T in trans_matrices:
        total += T
    with np.errstate(invalid='ignore', divide='ignore'):
        avg = total / total.sum(axis=1, keepdims=True)
    avg[np.isnan(avg)] = 0.0
    return avg

# ---------- single matrix ----------

def plot_transition_matrix(T, title="Transition matrix", cmap="Blues", vmin=0.0, vmax=1.0,
                           set_diag_zero=False, annotate=False, fmt=".2f"):
    """
    Plot a single KxK transition matrix (row-stochastic) as a heatmap.
    """
    K = T.shape[0]
    plt.figure()
    if set_diag_zero:
        T = T - np.diag(np.diag(T))
        vmax = np.max(T)
    im = plt.imshow(T, vmin=vmin, vmax=vmax, cmap=cmap, aspect='equal')
    plt.colorbar(im)
    plt.xticks(np.arange(K), [f"S{k+1}" for k in range(K)])
    plt.yticks(np.arange(K), [f"S{k+1}" for k in range(K)])
    plt.xlabel("Next state")
    plt.ylabel("Current state")
    plt.title(title)
    if annotate:
        for i in range(K):
            for j in range(K):
                plt.text(j, i, format(T[i, j], fmt), ha="center", va="center")
    plt.tight_layout()
    plt.show()

# ---------- grid by condition (with order + colors for titles) ----------

def plot_transition_matrices_by_condition(trans_by_condition, order=None, colors=None,
                                          cmap="Blues", vmin=0.0, vmax=1.0, set_diag_zero=False,
                                          annotate=False, fmt=".2f", suptitle=None):
    """
    trans_by_condition: dict {condition_name: list_of_subject_trans_mats}
    order: list of condition names to control panel order (and title coloring)
    colors: optional dict {condition_name: color}; used for title text
    """

    conds = order if order is not None else list(trans_by_condition.keys())
    avgs = [average_transition_matrix(trans_by_condition[c]) for c in conds]
    if set_diag_zero:
        for kk, avg in enumerate(avgs):
            avgs[kk] = avg - np.diag(np.diag(avg))
    K = avgs[0].shape[0]
    vmax = np.max([np.max(avg) for avg in avgs])

    # default colors if not provided
    if colors is None:
        prop_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
        colors = {cond: prop_cycle[i % len(prop_cycle)]
                  for i, cond in enumerate(conds)}

    cols = min(len(conds), 2)
    rows = int(np.ceil(len(conds) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.5*rows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.ravel()

    last_im = None
    for ax, cond, Tavg in zip(axes, conds, avgs):
        im = ax.imshow(Tavg, vmin=vmin, vmax=vmax, cmap=cmap, aspect='equal')
        last_im = im
        ax.set_xticks(np.arange(K))
        ax.set_xticklabels([f"S{k+1}" for k in range(K)])
        ax.set_yticks(np.arange(K))
        ax.set_yticklabels([f"S{k+1}" for k in range(K)])
        ax.set_xlabel("Next state")
        ax.set_ylabel("Current state")
        title_color = (colors.get(
            cond) if colors is not None and cond in colors else None)
        ax.set_title(cond, color=title_color)
        if annotate:
            for i in range(K):
                for j in range(K):
                    ax.text(j, i, format(Tavg[i, j], fmt),
                            ha="center", va="center")

    # remove unused axes (if any)
    for ax in axes[len(conds):]:
        ax.axis("off")

    # shared colorbar
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(last_im, cax=cax)
    if suptitle:
        fig.suptitle(suptitle, y=0.98)
    fig.tight_layout() # rect=[0, 0, 0.9, 0.96]
    plt.show()

# ---------- difference map (A - B) with diverging cmap ----------

def plot_transition_matrix_diff(TA, TB, title="A - B (transition difference)",
                                cmap="coolwarm", clim=None, annotate=False, fmt=".2f"):
    """
    Visualize the difference between two transition matrices (same K).
    TA, TB: (K, K) row-stochastic matrices (e.g., condition averages).
    clim: (vmin, vmax) for symmetric range, e.g., (-0.3, 0.3). If None, auto.
    """
    D = TA - TB
    K = D.shape[0]
    vmin, vmax = clim if clim is not None else (np.min(D), np.max(D))
    plt.figure()
    im = plt.imshow(D, vmin=vmin, vmax=vmax, cmap=cmap, aspect='equal')
    plt.colorbar(im)
    plt.xticks(np.arange(K), [f"S{k+1}" for k in range(K)])
    plt.yticks(np.arange(K), [f"S{k+1}" for k in range(K)])
    plt.xlabel("Next state")
    plt.ylabel("Current state")
    plt.title(title)
    if annotate:
        for i in range(K):
            for j in range(K):
                plt.text(j, i, format(D[i, j], fmt), ha="center", va="center")
    plt.tight_layout()
    plt.show()
    
 
# ---------- plot average static FC ----------

def plot_mean_static_FC(static_FC_by_condition, colors=None, cmap="viridis",
                        roi_to_networks=None, save=False,
                        title='Average static FC per condition'):
    conds = list(static_FC_by_condition.keys())
    avgs = [np.mean(static_FC_by_condition[c], axis=0) for c in conds]
    
    if colors is None:
        prop_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
        colors = {cond: prop_cycle[i % len(prop_cycle)]
                  for i, cond in enumerate(conds)}
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.3)
    axes = np.array(axes).ravel()
    
    im = None
    for ax, cond, avg in zip(axes, conds, avgs):
        im = ax.matshow(np.tanh(avg), vmin=-1, vmax=1, cmap=cmap)
        title_color = (colors.get(cond) if cond in colors else None)
        ax.set_title(' '.join(cond.split('_')), color=title_color)
        xx = np.linspace(min(10, avg.shape[0]-avg.shape[0]%10),
                         min(210, avg.shape[0]-avg.shape[1]%10), 5)
        ax.set_xticks(xx)
        ax.set_yticks(xx)
        ax.tick_params(axis='both', direction='out')
        ax.grid(False)
    
    lefts = [ax.get_position().xmin for ax in axes]
    rights = [ax.get_position().xmax for ax in axes]
    left = min(lefts)
    right = max(rights)

    cbar_height = 0.01
    cbar_bottom = 0.05
    cbar_ax = fig.add_axes([left, cbar_bottom, right - left, cbar_height]) # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax,
                        extend='both', extendfrac=0.03, 
                        orientation='horizontal')
    # cbar.set_label(r'$FC_{ij}$', labelpad=10, fontsize=font)
    cbar.ax.set_title(r'$FC_{ij}$', pad=10)
    cbar.ax.tick_params(axis='x', direction='out')
    cbar.set_ticks(np.arange(-1, 1.25, 0.25))
    cbar.ax.grid(False)
    
    # Add networks names to axes
    if roi_to_networks is not None:
        networks = create_net_name(roi_to_networks)

        for kk, ax in enumerate(axes):
            for label, start, end in networks:
                if kk % 2 == 0:  # labels only on left plots
                    center = (start + end) / 2
                    ax.annotate(label, xy=(-0.18, center), xycoords=('axes fraction', 'data'),
                                ha='right', va='center', rotation=0, fontsize=9)

                ax.axhline(start, color='black', linewidth=0.7, ls='--')
                ax.axhline(end, color='black', linewidth=0.7, ls='--')
    
    if title is not None:
        plt.suptitle(title, y=0.95)
    if save:
        plt.savefig('imgs/Average_static_FC_per_condition.png', dpi=300)
    plt.show()
    

def plot_mean_static_FC_single(static_FC, cond_name='BASELINE', 
                               cmap="viridis", save=False,
                               colors=None, roi_to_networks=None):
    avg = np.mean(static_FC, axis=0)
    
    if colors is None:
        prop_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
        color = { 'FC': prop_cycle[-1] }
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    
    im = ax.matshow(np.tanh(avg), vmin=-1, vmax=1, cmap=cmap)
    title_color = color['FC']
    ax.set_title(cond_name, color=title_color)
    
    left = ax.get_position().xmin
    right = ax.get_position().xmax

    cbar_height = 0.01
    cbar_bottom = 0.02
    cbar_ax = fig.add_axes([left, cbar_bottom, right - left, cbar_height]) # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax,
                        extend='both', extendfrac=0.03, 
                        orientation='horizontal')
    cbar.ax.set_title(r'$FC_{ij}$', pad=10)
    cbar.ax.tick_params(axis='x', direction='out')
    cbar.set_ticks(np.arange(-1, 1.25, 0.25))
    cbar.ax.grid(False)
    
    # Add networks names to axes
    if roi_to_networks is not None:
        networks = create_net_name(roi_to_networks)

        for label, start, end in networks:
            center = (start + end) / 2
            ax.annotate(label, xy=(-0.18, center), xycoords=('axes fraction', 'data'),
                        ha='right', va='center', rotation=0, fontsize=9)

            # ax.axhline(start, color='black', linewidth=0.7, ls='--')
            ax.axhline(end, color='black', linewidth=0.7, ls='--')
    if save:
        plt.savefig(f'imgs/Average_static_FC_per_{cond_name}.png', dpi=300)
    plt.show()


# Dump function to plot networks names without overlap
def create_net_name(roi_to_network):
    # identify where subcorticals start:
    len_names = np.array([len(n) for n in roi_to_network])
    try:
        sub_start = int(np.where(len_names < 4)[0][0])
    except IndexError:
        sub_start = None
    
    # Account for left and right hemisphere names
    res = []
    for net in np.unique(roi_to_network[:sub_start]):
        pos = []
        wh = np.where(net == roi_to_network)[0]
        pos.append(wh[0])
        diff = np.diff(wh)
        splits = np.where(diff > 1)[0]
        if len(splits) > 0:
            pos.append(wh[splits[0]])
            pos.append(wh[splits[0]+1])
            pos.append(wh[-1])
            
            res.append((str(net), int(pos[0]), int(pos[1])))
            res.append((str(net), int(pos[2]), int(pos[-1])))
        else:
            pos.append(wh[-1])
            res.append((str(net), int(pos[0]), int(pos[-1])))
    net_name = res + ([('Subcortical', sub_start, len(roi_to_network)-1)] if sub_start is not None else [])
    return net_name


# ---------- plot 4 FC matrices ----------

def plot_4_conditions(matrices, colors=None, 
                      roi_to_networks=None,
                      title='Median FC per condition'):
    conds = list(matrices.keys())
    
    if colors is None:
        prop_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
        colors = {cond: prop_cycle[i % len(prop_cycle)]
                  for i, cond in enumerate(conds)}
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    axes = np.array(axes).ravel()
    
    im = None
    for ax, cond, mat in zip(axes, conds, matrices.values()):
        im = ax.matshow(mat, vmin=-1, vmax=1, cmap="viridis")
        title_color = (colors.get(cond) if cond in colors else None)
        ax.set_title(cond, color=title_color)
    
    lefts = [ax.get_position().xmin for ax in axes]
    rights = [ax.get_position().xmax for ax in axes]
    left = min(lefts)
    right = max(rights)

    cbar_height = 0.02
    cbar_bottom = 0.05
    cbar_ax = fig.add_axes([left, cbar_bottom, right - left, cbar_height]) # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    
    # Add networks names to axes
    if roi_to_networks is not None:
        networks = create_net_name(roi_to_networks)

        for kk, ax in enumerate(axes):
            for label, start, end in networks:
                if kk % 2 == 0:  # labels only on left plots
                    center = (start + end) / 2
                    ax.annotate(label, xy=(-0.08, center), xycoords=('axes fraction', 'data'),
                                ha='right', va='center', rotation=30, fontsize=7)

                ax.axhline(start, color='black', linewidth=0.5, ls='--')
                ax.axhline(end, color='black', linewidth=0.5, ls='--')
                    
    plt.suptitle(title, y=0.95)
    plt.show()

# ---------- Violin plot ----------

def violin_plot(data_to_plot, plot_labels, p_values, comparisons=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    vp = ax.violinplot(data_to_plot, showmeans=True, showextrema=True, showmedians=False)

    colors = ["#7692ff", "#091540", "#abd2fa", 
              "#3d518c", "#1b2cc1"][::-1]

    for jj, body in enumerate(vp['bodies']):
        body.set_facecolor(colors[(jj*2) % 6])
        body.set_edgecolor('black')
        body.set_alpha(0.7)

    positions = []
    for i, d in enumerate(data_to_plot, start=1):
        x = np.random.normal(i, 0.04, size=len(d)) # avoid superpositions
        ax.scatter(x, d, alpha=0.6, color='black', s=10, label=None)
        positions.append(x)

    # Draw lines between corresponding points
    for index, (y1, y2, x1, x2) in enumerate(zip(data_to_plot[0::], data_to_plot[1::], 
                                        positions[0::], positions[1::])):
        if index == 2: continue
        ax.plot([x1, x2], [y1, y2], color='lightgray', alpha=0.5, linewidth=0.7)

    # Index pairs for comparisons
    comparisons = comparisons if comparisons is not None else [(ii,ii+3) for ii in range(0, 3)]
    y_offset1 = 0.02  # Vertical offset for the lines
    y_offset2 = 0.08  # Vertical offset for the lines
    ls = ['-', '--', '-.']
    for ii, ((start, end), p_val) in enumerate(zip(comparisons, p_values)):
        if ii == 0:
            line_height = max(max(d) for d in data_to_plot) + y_offset1
        elif ii == 1:
            line_height = min(min(d) for d in data_to_plot) - y_offset1
        elif ii == 2:
            line_height = min(min(d) for d in data_to_plot) - y_offset2
        
        x_start, x_end = start + 1, end + 1  # Convert to 1-based index for violin plot positions
        y = line_height
        ax.plot([x_start, x_end], [y, y], ls=ls[ii], color=colors[(ii*2) % 6], lw=1.5)
        ax.text((x_start + x_end) / 2, y + y_offset1/5, 
                'n.s.' if p_val > 0.05 else '*' * (1 if 0.01 < p_val <= 0.05 else 2 if 0.001 < p_val <= 0.01 else 3),
                ha='center', va='bottom', fontsize=15)
        # line_height += y_offset  # Increment for next line


    ax.set_xticks(list(range(1, len(data_to_plot)+1)))
    ax.set_xticklabels(plot_labels, rotation=20, fontsize=15)

    # ax.set_title(f'Training data', fontsize=14)
    ax.set_ylabel(r'$\rho_{\mathrm{tot}}$')
    plt.grid()
    # plt.legend(loc='upper right')
    plt.tight_layout()
    # plt.savefig(f'{imgs_folder}/corr_training_violin.png', dpi=300)
    plt.show()
    

def plot_dispersion_violin(dispersion_dict, order=None, colors=None, title="State dispersion by condition"):
    """
    Plot violin plots of dispersion values (distance of windows from their state centroid).

    Parameters
    ----------
    dispersion_dict : dict
        {condition: {k: distances_array}} 
        where distances_array are distances of windows from centroid for state k.
    order : list, optional
        Order of conditions to plot. Default = sorted keys.
    colors : dict, optional
        {condition: color} mapping. If None, matplotlib default cycle is used.
    title : str
        Plot title.
    """
    # Flatten data into long-form
    rows = []
    for cond, states in dispersion_dict.items():
        for k, dists in states.items():
            for val in dists:
                rows.append({"Condition": cond, "State": f"State {k+1}", "Distance": val})

    df = pd.DataFrame(rows)

    # Set plotting order
    conditions = order if order is not None else sorted(dispersion_dict.keys())

    # Color mapping
    if colors is None:
        palette = None  # seaborn default
    else:
        palette = {cond: colors[cond] for cond in conditions if cond in colors}

    plt.figure(figsize=(1.6 + 2*len(df["State"].unique()), 5))
    sns.violinplot(
        data=df, x="State", y="Distance",
        hue="Condition", order=[f"State {i+1}" for i in range(len(dispersion_dict[conditions[0]]))],
        hue_order=conditions,
        split=False, inner="box", palette=palette
    )

    plt.title(title)
    plt.ylabel("Distance from centroid")
    plt.xlabel("State")
    plt.legend(title="Condition", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_state_distance_distributions(d_within_A, d_within_B, d_cross,
                                      labels=("Within A", "Within B", "Cross A-B"),
                                      title="Window-to-window distance distributions for one state",
                                      p_val=None, logy=False):
    """
    Violin (con box interno) delle tre distribuzioni di distanza per UN singolo stato.
    d_within_A, d_within_B, d_cross: array 1D di distanze (possono essere vuoti).
    labels: etichette per le tre condizioni.
    logy: se True, asse y in scala log (utile se code lunghe).
    """
    # assicurati che siano 1D float
    arrays = [np.asarray(d).ravel().astype(float) for d in (d_within_A, d_within_B, d_cross)]
    names  = list(labels)

    # build long-form DF
    rows = []
    for name, arr in zip(names, arrays):
        for v in arr:
            rows.append({"Group": name, "Distance": v})
    if not rows:
        raise

    df = pd.DataFrame(rows)
    plt.figure(figsize=(7, 5))
    # sns.violinplot(data=df, x="Group", y="Distance", inner="box", cut=0)
    data_to_plot = [list(df[df['Group'] == key].values[:, 1]) for key in df['Group'].unique()]
    plt.violinplot(data_to_plot, showmeans=True, showextrema=True, showmedians=False)
    plt.xticks(list(range(1, 4)), labels)
    
    p_val = cohens_d(data_to_plot[0], data_to_plot[1]) if p_val is None else p_val
        
    y = max(max(d) for d in data_to_plot) + 0.01
    plt.plot([1, 2], [y, y], ls='--', lw=1.5)
    plt.text((1 + 2) / 2, y + 0.001, 
            'n.s.' if p_val > 0.05 else '*' * (1 if 0.01 < p_val <= 0.05 else 2 if 0.001 < p_val <= 0.01 else 3),
            ha='center', va='bottom', fontsize=10)
    
    if logy:
        plt.yscale("log")
    plt.title(title)
    plt.xlabel("")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.show()
    
    
# ---------------------- Edge-wise analysis plot ----------------------    

def plot_global_series_mean_sem(series_by_cs, conditions, colors=None, 
                                step=None, win_len=None, TR=None, save=False,
                                title="Global FC over windows (mean ± SEM)"):
    """
    series_by_cs: {(cond, subj): g_t (W,)}
    Plot for each condition mean ± SEM over time.
    """
    if colors is None:
        prop_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
        colors = {cond: prop_cycle[i % len(prop_cycle)]
                  for i, cond in enumerate(conditions)}
    plt.figure(figsize=(9,4))
    for cond in conditions:
        S = list(series_by_cs[cond].values())
        Wmin = min(map(len, S))
        S_trim = np.stack([g[:Wmin] for g in S], axis=0)  # (S, Wmin)
        mean = S_trim.mean(axis=0)
        sem  = S_trim.std(axis=0, ddof=1) / np.sqrt(S_trim.shape[0])
        t = (np.arange(Wmin) * step + (win_len / 2)) * TR if step is not None and win_len is not None and TR is not None else np.arange(Wmin)
        plt.plot(t, mean, label=' '.join(cond.split('_')), color=colors[cond])
        plt.fill_between(t, mean-sem, mean+sem, color=colors[cond], alpha=0.2)
    plt.xlabel("Window" if step is None and win_len is None and TR is None else f"Time (s)"); plt.ylabel("Global FC (Fisher-z)", fontsize=25)
    plt.title(title); plt.legend(frameon=False, loc='upper right', bbox_to_anchor=(1.35,0.7), fontsize=18); 
    if save:
        plt.savefig(f'imgs/{"_".join(title.split(" "))}.png', dpi=300)
    plt.show() # plt.tight_layout(); 

def plot_topk_edge_heatmap(dfc_by_condition, cond, roi_roi_names, k=50, metric="dcv",
                           cmap="viridis", save=False, vmin=-1, vmax=1):
    """
    Show heatmap (edge x window) of the most dynamic top-k per condition
    metric: 'dcv' (variance), 'median',  'vol' (volatility: median |Δ|)
    Edge selection made upon average across subjects.
    """
    subj_list = np.array(dfc_by_condition[cond])
    Ms = []
    for vec_windows in subj_list:
        if metric.lower() == "dcv": # DCV = Dynamic Connectivity Variance
            m = vec_windows.var(axis=0, ddof=1)
        elif metric.lower() == "median":
            m = np.median(vec_windows, axis=0)
        elif metric.lower() == "vol":
            m = np.median(np.abs(np.diff(vec_windows, axis=0)), axis=0)
        else:
            if type(metric) == str:
                raise ValueError(f"Unknown metric '{metric}'. Use 'dcv' or 'median'.")
            elif type(metric) == function:
                m = metric(vec_windows)
            else:
                raise ValueError(f"metric must be 'dcv', 'median' or a function.")
        Ms.append(m)
    Mmean = np.mean(np.vstack(Ms), axis=0)  # (E,)
    # select top-k edge
    idx = np.argsort(Mmean)[::-1][:k]
    Z0 = subj_list[:, :, idx].T  # (k, W0, subj)
    print(Z0.mean(axis=-1).min(), Z0.mean(axis=-1).max())
    plt.figure(figsize=(16, 0.5*k + 2))
    plt.imshow(Z0.mean(axis=-1), vmin=vmin, vmax=vmax, aspect="auto", 
               cmap=cmap, extent=[0.5, 0.5+Z0.shape[1], -0.5, k-0.5])
    plt.colorbar(label="Edge FC mean (Fisher-z)")
    plt.yticks(np.arange(k), [f"{roi_roi_names[e]}" for e in idx])
    plt.xticks(range(1, Z0.shape[1]+1), labels=[i if i in [1 if x == 0 else x for x in range(0, Z0.shape[1]+1, 5)] else '' for i in range(1, Z0.shape[1]+1)])
    plt.xlabel(r"\# Window"); plt.ylabel("Top-k edges", fontsize=20)
    title = f"{' '.join(cond.split('_'))}  heatmap top-{k} edges by {metric.upper()}"
    plt.grid(False)
    plt.title(title, fontsize=20)
    plt.tight_layout(); 
    if save:
        plt.savefig(f'imgs/{"_".join(title.split(" "))}.png', dpi=300)
    plt.show()

def plot_topk_edge_heatmap_delta(dfc_by_condition, cond_pair, roi_roi_names, k=50, metric="dcv", 
                                 cmap="viridis", save=False, vmin=-1, vmax=1):
    """
    Show heatmap (edge x window) of the most dynamic top-k per condition
    metric: 'dcv' (variance) o 'vol' (median |Δ|)
    Edge selection made upon average across subjects.
    """
    cond1, cond2 = cond_pair
    subj_list1 = np.array(dfc_by_condition[cond1])
    subj_list2 = np.array(dfc_by_condition[cond2])
    Ms1, Ms2 = [], []
    for vec_windows1, vec_windows2 in zip(subj_list1, subj_list2):
        if metric.lower() == "dcv":
            m1 = vec_windows1.var(axis=0, ddof=1)
            m2 = vec_windows2.var(axis=0, ddof=1)
        elif metric.lower() == "median":
            m1 = np.median(vec_windows1, axis=0)
            m2 = np.median(vec_windows2, axis=0)
        elif metric.lower() == "vol":
            m1 = np.median(np.abs(np.diff(vec_windows1, axis=0)), axis=0)
            m2 = np.median(np.abs(np.diff(vec_windows2, axis=0)), axis=0)
        else:
            if type(metric) == str:
                raise ValueError(f"Unknown metric '{metric}'. Use 'dcv' or 'median'.")
            elif type(metric) == function:
                m1 = metric(vec_windows1)
                m2 = metric(vec_windows2)
            else:
                raise ValueError(f"metric must be 'dcv', 'median' or a function.")
        Ms1.append(m1)
        Ms2.append(m2)
    Mmean1 = np.mean(np.vstack(Ms1), axis=0)  # (E,)
    Mmean2 = np.mean(np.vstack(Ms2), axis=0)  # (E,)
    Mmean = Mmean2 - Mmean1
    # select top-k edge
    idx = np.argsort(Mmean)[::-1][:k]
    # chain windows across all subjects (to display common trend): use the first subject for simplicity
    Z0 = subj_list2[:, :, idx].T  # (k, W0, subj)
    print(Z0.mean(axis=-1).min(), Z0.mean(axis=-1).max())
    plt.figure(figsize=(16, 0.5*k + 2))
    plt.imshow(Z0.mean(axis=-1), aspect="auto", vmin=vmin, vmax=vmax,
               cmap=cmap, extent=[0.5, 0.5+Z0.shape[1], -0.5, k-0.5])
    plt.colorbar(label="Edge FC mean (Fisher-z)")
    plt.yticks(np.arange(k), [f"{roi_roi_names[e]}" for e in idx])
    plt.xticks(range(1, Z0.shape[1]+1), labels=[i if i in [1 if x == 0 else x for x in range(0, Z0.shape[1]+1, 5)] else '' for i in range(1, Z0.shape[1]+1)], fontsize=20)
    plt.xlabel(r"\# Window", fontsize=25); plt.ylabel("Top-k edges", fontsize=20)
    title = f"{' vs '.join([' '.join(name.split("_")) for name in cond_pair])} - heatmap top-{k} edges by {metric.upper()}"
    plt.title(title, fontsize=20)
    plt.grid(False)
    plt.tight_layout()
    if save:
        plt.savefig(f'imgs/{"_".join(title.split(" "))}.png', dpi=300)
    plt.show()    
    


def plot_violin_metric_by_netpair(df_metrics_netpair, metric="mu", condA=None, condB=None,
                                  netpairs=None, ax_array=None,
                                  compare_pairs=None,        # if None -> default [(REAL_PRE,REAL_POST),(SHAM_PRE,SHAM_POST)]
                                  all_comparison=True,
                                  paired_pairs=None,         # if None -> default pairs PRE↔POST
                                  alpha=0.05,
                                  fdr=False,                 # FDR for net_pair panel
                                  metric_name=None,
                                  connector_kwargs=None):
    """ df_metrics_netpair: output of netpair_metrics_per_subject (long-form).
    Violin plot of the distributions of 'metric' for net_pair and condition (opz. only A/B). 
    Lines and asterisks on significant pairs condition (t-test; paired for PRE↔POST)
    """
    df = df_metrics_netpair.copy()
    if condA and condB:
        df = df[df["condition"].isin([condA, condB])]
    if netpairs:
        df = df[df["net_pair"].isin(netpairs)]

    pairs = sorted(df["net_pair"].unique())
    conds = sorted(df["condition"].unique())[::-1]
    color_map = {c: plt.rcParams['axes.prop_cycle'].by_key()['color'][i % 10] for i, c in enumerate(conds)}

    # figure/axes
    if ax_array is None:
        fig, axes = plt.subplots(1, len(pairs), figsize=(4*len(pairs), 8), sharey=True)
    else:
        axes = ax_array
        fig = axes[0].figure if isinstance(axes, (list, np.ndarray)) else axes.figure
    if len(pairs) == 1:
        axes = [axes]

    # --- local helper ---
    def _as_unordered_pair(a, b):
        return tuple(sorted((a, b)))

    def get_ckw(pair_key):
        default_kw = dict(linestyle='--', linewidth=1.5, color='k', alpha=0.9)
        if connector_kwargs is None:
            return default_kw
        if isinstance(connector_kwargs, dict):
            if all(isinstance(v, dict) for v in connector_kwargs.values()) and pair_key in connector_kwargs:
                return connector_kwargs[pair_key]
            return connector_kwargs
        return default_kw

    pos = {c: i for i, c in enumerate(conds, start=1)}
    if compare_pairs is None:
        candidate_defaults = [("REAL_PRE","REAL_POST"), ("SHAM_PRE","SHAM_POST")]
        compare_pairs = [(a,b) for (a,b) in candidate_defaults if (a in pos and b in pos)]
        if all_comparison:
            compare_pairs = list(combinations(conds, 2))

    # paired_pairs: by default the pairs PRE↔POST
    if paired_pairs is None:
        paired_pairs = [("REAL_PRE","REAL_POST"), ("SHAM_PRE","SHAM_POST")]

    paired_set = set(_as_unordered_pair(*p) for p in paired_pairs if isinstance(p, (list, tuple)) and len(p) == 2)

    ordered = []
    for (a, b) in compare_pairs:
        if a in pos and b in pos:
            ordered.append((a, b))
    compare_pairs = ordered

    for ax, pair in zip(axes, pairs):
        sub = df[df["net_pair"] == pair]
        data = [sub[sub["condition"] == c][metric].values for c in conds]

        parts = ax.violinplot(data, showmeans=True, showextrema=True, showmedians=False)
        for b, pc in enumerate(parts['bodies']):
            pc.set_facecolor(color_map[conds[b]])
            pc.set_alpha(0.3)

        ax.set_xticks(np.arange(1, len(conds)+1))
        ax.set_xticklabels([' '.join(cond.split('_')) for cond in conds], rotation=15)
        ax.set_title(rf'{metric if metric_name is None else metric_name} by networks pair' if condA is None else pair, fontsize=25)

        positions = []
        for i, d in enumerate(data, start=1):
            x = np.random.normal(i, 0.04, size=len(d)) # avoid superpositions
            ax.scatter(x, d, alpha=0.6, color='black', s=50, label=None)
            positions.append(x)

        # Draw lines between corresponding points
        for y1, y2, x1, x2 in zip(data[0::2], data[1::2],
                                  positions[0::2], positions[1::2]):
            ax.plot([x1, x2], [y1, y2], color='lightgray', alpha=0.5, linewidth=1.5)

        entries = []
        x_pos = {c: i for i, c in enumerate(conds, start=1)}
        for (c1, c2) in compare_pairs:
            v1 = sub[sub["condition"] == c1]
            v2 = sub[sub["condition"] == c2]
            use_paired = _as_unordered_pair(c1, c2) in paired_set

            if use_paired and ("subj" in v1.columns) and ("subj" in v2.columns):
                mrg = pd.merge(v1[["subj", metric]], v2[["subj", metric]],
                               on="subj", suffixes=("_A", "_B"))
                x = mrg[f"{metric}_A"].values
                y = mrg[f"{metric}_B"].values
                if len(x) >= 2:
                    stat, p = ttest_rel(x, y, nan_policy='omit')
                else:
                    p = np.nan
            else:
                x = v1[metric].values
                y = v2[metric].values
                if len(x) >= 2 and len(y) >= 2:
                    stat, p = ttest_ind(x, y, equal_var=False, nan_policy='omit')  # Welch
                else:
                    p = np.nan

            entries.append((c1, c2, p, (x_pos[c1], x_pos[c2])))

        # FDR per pannel (on the m p-value of the net_pair)
        if fdr:
            valid_p = [e[2] for e in entries if np.isfinite(e[2])]
            qvals, _ = bh_fdr(np.array(valid_p))
            p_iter = iter(valid_p)
            q_iter = iter(qvals)
            p_by_pair = {}
            q_by_pair = {}
            sig_by_pair = {}
            for (c1, c2, p, _) in entries:
                if np.isfinite(p):
                    q = float(next(q_iter)); p = float(next(p_iter))
                    q_by_pair[(c1, c2)] = q; p_by_pair[(c1, c2)] = p
                    sig_by_pair[(c1, c2)] = (q <= alpha)
                else:
                    q_by_pair[(c1, c2)] = np.nan; p_by_pair[((c1, c2))] = np.nan
                    sig_by_pair[(c1, c2)] = False
        else:
            q_by_pair = {(c1, c2): (p if np.isfinite(p) else np.nan) for (c1, c2, p, _) in entries}
            p_by_pair = {(c1, c2): (p if np.isfinite(p) else np.nan) for (c1, c2, p, _) in entries}
            sig_by_pair = {(c1, c2): (np.isfinite(p) and p <= alpha) for (c1, c2, p, _) in entries}

        # ---- Draw significant connectors ----
        y_min, y_max = ax.get_ylim()
        dy = (y_max - y_min) if y_max > y_min else 1.0
        base = y_max + 0.02 * dy
        bump = 0.0

        for (c1, c2, p, (x1, x2)) in entries:
            # if not sig_by_pair[(c1, c2)]:
            #     continue
            y = base + bump
            ckw = get_ckw((c1, c2))
            ax.plot([x1, x2], [y, y], **ckw)

            qv = q_by_pair[(c1, c2)]; pv = p_by_pair[(c1, c2)]
            if np.isfinite(qv):
                if qv <= 0.001: stars = '***'
                elif qv <= 0.01: stars = '**'
                elif qv <= alpha: stars = '*'
                else: stars = 'n.s.'
                if stars:
                    ax.text((x1 + x2) / 2.0, y + 0.01 * dy, stars,
                            ha='center', va='bottom', fontsize=18)
            bump += 0.06 * dy

        ax.set_ylim(y_min, y_max + bump + 0.08 * dy)

    axes[0].set_ylabel(rf'{metric if metric_name is None else metric_name}')
    plt.tight_layout()
    if ax_array is None:
        plt.show()
    else:
        return fig, axes