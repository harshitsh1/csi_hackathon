import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .labels import ROOM_LABELS


def _get(df, **kw):
    m = np.ones(len(df), bool)
    for k, v in kw.items():
        m &= (df[k] == v).to_numpy()
    return df[m].sort_values("K")


def label_efficiency(df, out):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, T in zip(axs, "AB"):
        S = "B" if T == "A" else "A"
        for metric, st, lab in (("macro_f1", "-o", "overall macro-F1"),
                                ("target_exclusive_macro_f1", "-s", "target-labeled (exclusive)"),
                                ("shared_macro_f1", "-^", "transferred shared (SIT/STAND/WALK)")):
            g = _get(df, experiment="task3_adaptation", target_room=T, metric=metric, method="ssl")
            sd = _get(df, experiment="task3_adaptation", target_room=T, metric=metric + "_std", method="ssl")
            ax.errorbar(g.K, g.value, yerr=sd.value.to_numpy(), fmt=st, capsize=3, label="SSL: " + lab)
        for rep, ls in (("random", "--"), ("hand", ":")):
            g = _get(df, experiment="task3_adaptation", target_room=T, metric="macro_f1", method=rep)
            ax.plot(g.K, g.value, ls, color="gray", label=f"{rep}: overall macro-F1")
        ax.set_xscale("log"); ax.set_xticks([1, 5, 10, 20]); ax.set_xticklabels([1, 5, 10, 20])
        ax.set_title(f"Target Room {T} (source Room {S})"); ax.set_xlabel("K labeled target captures / exclusive class")
        ax.grid(alpha=.3)
    axs[0].set_ylabel("validation macro-F1"); axs[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fingerprint(df, out):
    g = df[(df.experiment == "task1_room_id") & (df.target_room == "all") & df.metric.str.startswith("room_accuracy_L")]
    L = g.metric.str[len("room_accuracy_L"):].astype(int)
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.plot(L, g.value.to_numpy(), "-o"); ax.set_xlabel("window length (observations)")
    ax.set_ylabel("room accuracy (validation)"); ax.set_ylim(0, 1.02); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def routing(df, out):
    g = df[(df.experiment == "task3_routing") & (df.metric == "macro_f1")]
    modes = ["no_routing", "predicted_room_routing", "oracle_room"]
    Ks = sorted(g.K.unique()); w = 0.27
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for i, m in enumerate(modes):
        v = [g[(g.K == k) & (g.method == m)].value.mean() for k in Ks]
        ax.bar(np.arange(len(Ks)) + (i - 1) * w, v, w, label=m)
    ax.set_xticks(range(len(Ks))); ax.set_xticklabels(Ks); ax.set_xlabel("K"); ax.set_ylabel("pooled macro-F1 (8 labels)")
    ax.legend(fontsize=7); ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def confusion(res_dir, out):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, T in zip(axs, "AB"):
        cm = pd.read_csv(res_dir / f"confusion_K5_target{T}.csv", index_col=0)
        pc = pd.read_csv(res_dir / f"per_class_K5_target{T}.csv")
        ax.imshow(cm.values, cmap="Blues")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, int(cm.values[i, j]), ha="center", va="center", fontsize=8)
        ax.set_xticks(range(len(cm))); ax.set_xticklabels(cm.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(cm)))
        ax.set_yticklabels([f"{a} (P{p:.2f} R{r:.2f} F{f:.2f})" for a, p, r, f in
                            zip(pc.activity, pc.precision, pc.recall, pc.f1)], fontsize=7)
        ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"Target Room {T}, K=5")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def embedding_plot(E, meta, out, max_n=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(E), min(max_n, len(E)), replace=False)
    Z = (E[idx] - E[idx].mean(0)) / (E[idx].std(0) + 1e-6)
    try:
        import umap
        Y = umap.UMAP(n_components=2, random_state=seed).fit_transform(Z); name = "UMAP"
    except Exception:
        from sklearn.manifold import TSNE
        Y = TSNE(2, random_state=seed, init="pca", perplexity=min(30, len(Z) - 1)).fit_transform(Z); name = "t-SNE"
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, col in zip(axs, ("room", "label")):
        v = meta[col].to_numpy()[idx]
        for u in sorted(set(v)):
            ax.scatter(*Y[v == u].T, s=6, label=u)
        ax.set_title(f"{name} of frozen SSL embeddings (validation) - by {col}"); ax.legend(fontsize=7, markerscale=2)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
