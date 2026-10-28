"""Deployment-rule selection, figures, LaTeX tables and a plain-text summary.

Run:  python -m tcfd.report -c config.yaml
Needs results/metrics.csv + results/cp.csv (tcfd.evaluate) and, optionally,
jetson_out/bench/bench.csv (latency/energy). Writes:
  results/selection.csv, results/summary.md
  paper/figures/*.pdf   paper/tables/*.tex
"""
from __future__ import annotations

import json
from pathlib import Path

import os

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")  # reproducible PDF metadata (no creation date)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import base_parser, load_config, path  # noqa: E402
from .metrics import reliability, softmax  # noqa: E402

PREC_ORDER = ["fp32", "fp16", "int8_src", "int8_all", "ws_fp32"]
PREC_LABEL = {"fp32": "FP32", "fp16": "FP16", "int8_src": "INT8 (src cal.)", "int8_all": "INT8 (all-cond. cal.)",
              "ws_fp32": "FP32 (PyTorch)"}
VAR_LABEL = {"src": "Source-calibrated CP", "pooled": "Pooled CP", "mondrian": "Operating-point (Mondrian) CP",
             "lobo_src": "Bearing-level CP (source)", "lobo_pooled": "Bearing-level CP (pooled)",
             "lobo_mondrian": "Bearing-level operating-point CP"}
VAR_SHORT = {"src": "Source-cal. CP", "pooled": "Pooled CP", "mondrian": "Op.-point CP",
             "lobo_src": "Bearing-lvl CP (src)", "lobo_pooled": "Bearing-lvl CP (pooled)",
             "lobo_mondrian": "Bearing-lvl op.-point CP"}


def has_lobo(C) -> bool:
    return "lobo_mondrian" in set(C.variant)


def main_variants(C):
    return ["src", "mondrian"] + (["lobo_mondrian"] if has_lobo(C) else [])


def legend_below(fig, ax, ncol=4):
    h, lab = ax.get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=ncol, bbox_to_anchor=(0.5, -0.01), frameon=False)
BYTES = {"fp32": 4, "fp16": 2, "int8_src": 1, "int8_all": 1, "ws_fp32": 4}

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "figure.dpi": 150, "savefig.bbox": "tight",
                     "pdf.fonttype": 42, "axes.grid": True, "grid.alpha": 0.3})
COL1, COL2 = 3.5, 7.16  # IEEE single / double column width [in]


def cond_label(cfg, c):
    lab = {"N15_M07_F10": "1500 rpm, 0.7 Nm, 1000 N", "N09_M07_F10": "900 rpm", "N15_M01_F10": "0.1 Nm",
           "N15_M07_F04": "400 N"}
    code = cfg["conditions"][c]
    tag = " (source)" if c == cfg["source_condition"] else ""
    return f"{c}: {lab.get(code, code)}{tag}"


def load_bench(cfg) -> pd.DataFrame | None:
    f = path(cfg, cfg["jetson"]["out_dir"]) / "bench" / "bench.csv"
    if not f.exists():
        return None
    b = pd.read_csv(f)
    if "trt_gpu_median_ms" in b:
        b["latency_ms"] = b["trt_gpu_median_ms"].fillna(b["py_lat_p50_ms"])
    else:
        b["latency_ms"] = b["py_lat_p50_ms"]
    keep = [c for c in ["arch", "ratio", "precision", "latency_ms", "py_lat_p50_ms", "trt_gpu_p99_ms",
                        "windows_per_s_b32", "energy_mj_per_inf", "dyn_energy_mj_per_inf", "engine_bytes"]
            if c in b]
    return b[keep].groupby(["arch", "ratio", "precision"], as_index=False).mean(numeric_only=True)


def cost_table(M: pd.DataFrame, bench: pd.DataFrame | None) -> pd.DataFrame:
    c = M[["arch", "ratio", "precision", "macs", "params"]].drop_duplicates(["arch", "ratio", "precision"])
    if bench is not None:
        c = c.merge(bench, on=["arch", "ratio", "precision"], how="left")
    else:
        c["latency_ms"] = np.nan
    # cost = measured latency; if ANY configuration lacks a measurement, use a MACs x bytes proxy for ALL
    # (never mix milliseconds with proxy units)
    proxy = c["macs"].astype(float) * c["precision"].map(BYTES).fillna(4) / 4.0
    rows = c[c.precision != "ws_fp32"] if (c.precision != "ws_fp32").any() else c
    complete = len(rows) > 0 and bool(rows["latency_ms"].notna().all())
    c["cost"] = c["latency_ms"] if complete else proxy / proxy.max()
    c["cost_is_proxy"] = not complete
    if not complete:
        print("[INFO] latency missing for some configurations -> cost = relative MACs x bytes proxy "
              "('Lat.' column in tab_selection.tex is then NOT milliseconds)")
    return c


# ------------------------------------------------------------------------- selection
def select(cfg, M, C, cost) -> pd.DataFrame:
    an = cfg["analysis"]
    alpha, eps, smax, drop = an["alpha"], an["sel_eps"], an["sel_max_set"], an["acc_only_drop"]
    src = cfg["source_condition"]
    shifted = [c for c in cfg["conditions"] if c != src]
    Mr = M[(M.calib == "raw")]
    Mt = M[(M.calib == "ts")]
    Cs = C[(C.score == an["score"])]
    rows = []
    for (fold, arch), g in Mr.groupby(["fold", "arch"]):
        vf =g[(g.split == "valfit") & (g.cond == src)].set_index(["ratio", "precision"])["bacc"]
        dense_key = next((k for k in [(0.0, "fp32"), (0.0, "ws_fp32")] if k in vf.index), vf.index[0])
        dense = vf.loc[dense_key]

        def test_summary(ratio, prec):
            t = Mr[(Mr.fold == fold) & (Mr.arch == arch) & (Mr.ratio == ratio) & (Mr.precision == prec)
                   & (Mr.split == "test")].set_index("cond")
            tt = Mt[(Mt.fold == fold) & (Mt.arch == arch) & (Mt.ratio == ratio) & (Mt.precision == prec)
                    & (Mt.split == "test")].set_index("cond")
            out = {"bacc_src": t.loc[src, "bacc"], "bacc_shift": t.loc[shifted, "bacc"].mean(),
                   "ece_shift_raw": t.loc[shifted, "ece"].mean(), "ece_shift_ts": tt.loc[shifted, "ece"].mean()}
            for v in main_variants(Cs):
                cc = Cs[(Cs.fold == fold) & (Cs.arch == arch) & (Cs.ratio == ratio) & (Cs.precision == prec)
                        & (Cs.split == "test") & (Cs.variant == v)]
                out[f"cov_worst_{v}"] = cc.coverage.min()
                out[f"set_size_{v}"] = cc.set_size.mean()
            return out

        cc_arch = cost[cost.arch == arch].set_index(["ratio", "precision"])["cost"]
        # Rule A: accuracy-only (common practice)
        okA = [k for k in vf.index if vf.loc[k] >= dense - drop]
        kA = min(okA, key=lambda k: cc_arch.get(k, np.inf))
        rows.append({"fold": fold, "arch": arch, "rule": "accuracy-only", "ratio": kA[0], "precision": kA[1],
                     "cost": cc_arch.get(kA, np.nan), **test_summary(*kA)})
        # Rule B: trust-aware, per CP variant
        for v in main_variants(Cs):
            cv = Cs[(Cs.fold == fold) & (Cs.arch == arch) & (Cs.split == "valfit") & (Cs.variant == v)]
            agg = cv.groupby(["ratio", "precision"]).agg(cov=("coverage", "min"), size=("set_size", "max"))
            # Absolute target (coverage >= 1-alpha-eps, size <= smax) when the dense model meets it; under
            # bearing shift it usually does not, so the rule becomes "compress without losing trust":
            # coverage >= dense coverage - eps and set size <= dense set size + 0.2 (validation estimates only).
            d_cov = agg.loc[dense_key, "cov"] if dense_key in agg.index else 1 - alpha
            d_size = agg.loc[dense_key, "size"] if dense_key in agg.index else smax
            if d_cov >= 1 - alpha - eps and d_size <= smax:  # dense model meets the absolute target
                cov_req, size_req = 1 - alpha - eps, smax
            else:  # relative: compression may not lose coverage or inflate sets w.r.t. the dense FP32 model
                cov_req, size_req = min(1 - alpha - eps, d_cov - eps), max(smax, d_size + 0.2)
            okB = [k for k in agg.index if agg.loc[k, "cov"] >= cov_req and agg.loc[k, "size"] <= size_req]
            fallback = okB == [dense_key] or not okB
            kB = min(okB, key=lambda k: cc_arch.get(k, np.inf)) if okB else dense_key
            rows.append({"fold": fold, "arch": arch, "rule": f"trust-aware ({v} CP)", "ratio": kB[0],
                         "precision": kB[1], "cost": cc_arch.get(kB, np.nan), "fallback_dense": fallback,
                         "cov_req": cov_req, "size_req": size_req,
                         **test_summary(*kB)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- figures
def agg(df, keys, col):
    g = df.groupby(keys)[col]
    return g.mean().rename("mean").to_frame().join(g.std().rename("std")).reset_index()


def fig_acc_ece(cfg, M, arch, prec, out):
    d = M[(M.arch == arch) & (M.precision == prec) & (M.split == "test") & (M.calib == "raw")]
    if d.empty:
        return
    fig, ax = plt.subplots(1, 2, figsize=(COL2, 2.2))
    for c in cfg["conditions"]:
        for j, col in enumerate(["bacc", "ece"]):
            a = agg(d[d.cond == c], ["ratio"], col)
            ax[j].errorbar(a.ratio * 100, a["mean"], yerr=a["std"].fillna(0), marker="o", ms=3, capsize=2,
                           label=cond_label(cfg, c), ls="-" if c == cfg["source_condition"] else "--")
    ax[0].set(xlabel="Channels pruned [%]", ylabel="Balanced accuracy (test bearings)")
    ax[1].set(xlabel="Channels pruned [%]", ylabel="ECE (uncalibrated)")
    fig.tight_layout()
    legend_below(fig, ax[1])
    fig.savefig(out / f"fig_acc_ece_{arch}.pdf")
    plt.close(fig)


def fig_coverage(cfg, C, arch, out, score):
    alpha = cfg["analysis"]["alpha"]
    precs = [p for p in ["fp32", "int8_src"] if p in set(C.precision)] or [C.precision.iloc[0]]
    d = C[(C.arch == arch) & (C.score == score) & (C.split == "test")]
    if d.empty:
        return
    variants = ["src", "pooled", "mondrian"] + (["lobo_mondrian"] if has_lobo(C) else [])
    fig, ax = plt.subplots(len(precs), len(variants), figsize=(COL2, 1.9 * len(precs)), squeeze=False, sharey=True)
    for i, p in enumerate(precs):
        for j, v in enumerate(variants):
            for c in cfg["conditions"]:
                a = agg(d[(d.precision == p) & (d.variant == v) & (d.cond == c)], ["ratio"], "coverage")
                ax[i, j].errorbar(a.ratio * 100, a["mean"], yerr=a["std"].fillna(0), marker="o", ms=3,
                                  capsize=2, label=cond_label(cfg, c),
                                  ls="-" if c == cfg["source_condition"] else "--")
            ax[i, j].axhline(1 - alpha, color="k", lw=0.8, ls=":")
            if i == 0:
                ax[i, j].set_title(VAR_SHORT[v])
            if i == len(precs) - 1:
                ax[i, j].set_xlabel("Channels pruned [%]")
        ax[i, 0].set_ylabel(f"{PREC_LABEL.get(p, p)}\nempirical coverage")
    fig.tight_layout()
    legend_below(fig, ax[0, 0])
    fig.savefig(out / f"fig_coverage_{arch}.pdf")
    plt.close(fig)


def fig_reliability(cfg, man, arch, out):
    exp = path(cfg, cfg["export"]["dir"])
    jd = path(cfg, cfg["jetson"]["out_dir"]) / "logits"
    shifted = [c for c in cfg["conditions"] if c != cfg["source_condition"]]
    ratios = sorted({m["ratio"] for f in man["folds"].values() for m in f["models"] if m["arch"] == arch})
    if not ratios:
        return
    r_cmp = max([r for r in ratios if r <= 0.7] or ratios)  # strong but not degenerate compression
    pairs = [(0.0, "fp32"), (r_cmp, "int8_src")]
    fig, ax = plt.subplots(1, 2, figsize=(COL1, 1.8), sharey=True)
    for k, (r, prec) in enumerate(pairs):
        P, Y = [], []
        for fold in man["folds"]:
            mid = f"{arch}_p{int(round(r * 100)):02d}"
            d = jd / fold / mid / prec
            if not d.exists():
                d = path(cfg, cfg["train"].get("runs_dir", "runs")) / fold / mid / "logits_ws"
                if not d.exists() or prec != "fp32":
                    continue
            for c in shifted:
                P.append(softmax(np.load(d / f"test_{c}.npy")))
                Y.append(np.load(exp / fold / "data" / f"test_{c}_y.npy"))
        if not P:
            ax[k].set_title(f"{PREC_LABEL.get(prec, prec)} p{int(r * 100)}: n/a")
            continue
        conf, acc, n, edges = reliability(np.concatenate(P), np.concatenate(Y), cfg["analysis"]["ece_bins"])
        ctr = (edges[:-1] + edges[1:]) / 2
        ax[k].bar(ctr, np.nan_to_num(acc), width=edges[1] - edges[0], alpha=0.7, edgecolor="k", lw=0.3)
        ax[k].plot([0, 1], [0, 1], "k:", lw=0.8)
        ax[k].set(xlabel="Confidence", title=f"{int(r * 100)}% pruned, {PREC_LABEL.get(prec, prec)}")
    ax[0].set_ylabel("Accuracy (shifted conditions)")
    fig.savefig(out / f"fig_reliability_{arch}.pdf")
    plt.close(fig)


def fig_pareto(cfg, M, C, cost, arch, out, score):
    if cost["cost_is_proxy"].all():
        return
    alpha = cfg["analysis"]["alpha"]
    t = C[(C.arch == arch) & (C.score == score) & (C.split == "test")]
    vs = main_variants(C)
    fig, ax = plt.subplots(1, len(vs), figsize=(COL2, 2.2), sharex=True)
    for j, v in enumerate(vs):
        g = (t[t.variant == v].groupby(["fold", "ratio", "precision"]).coverage.min()
             .groupby(["ratio", "precision"]).mean().reset_index())
        g = g.merge(cost[cost.arch == arch], on=["ratio", "precision"])
        g = g[g.latency_ms.notna()]
        for p in [x for x in PREC_ORDER if x in set(g.precision)]:
            gg = g[g.precision == p]
            ax[j].scatter(gg.latency_ms, (1 - alpha) - gg.coverage, label=PREC_LABEL[p], s=14)
            for _, r in gg.iterrows():
                ax[j].annotate(f"{int(r.ratio * 100)}", (r.latency_ms, (1 - alpha) - r.coverage), fontsize=5,
                               xytext=(2, 2), textcoords="offset points")
        ax[j].axhline(0, color="k", lw=0.8, ls=":")
        ax[j].set(xlabel="Latency, batch 1 [ms] (Jetson Orin Nano)", title=VAR_SHORT[v])
    ax[0].set_ylabel("Worst-condition coverage gap")
    fig.tight_layout()
    legend_below(fig, ax[-1])
    fig.savefig(out / f"fig_pareto_{arch}.pdf")
    plt.close(fig)


def fig_int8_calib(cfg, M, C, out, score):
    if not {"int8_src", "int8_all"} <= set(M.precision):
        return
    d = M[(M.split == "test") & (M.calib == "raw") & M.precision.isin(["int8_src", "int8_all"])]
    piv = d.pivot_table(index=["fold", "arch", "ratio", "cond"], columns="precision", values="ece").dropna()
    piv["delta"] = piv["int8_all"] - piv["int8_src"]
    cc = C[(C.split == "test") & (C.score == score) & (C.variant == "src") & C.precision.isin(["int8_src", "int8_all"])]
    pc = cc.pivot_table(index=["fold", "arch", "ratio", "cond"], columns="precision", values="coverage").dropna()
    pc["delta"] = pc["int8_all"] - pc["int8_src"]
    fig, ax = plt.subplots(1, 2, figsize=(COL1, 1.8))
    for j, (df, lab) in enumerate([(piv, "Δ ECE (all − src)"), (pc, "Δ coverage (all − src)")]):
        a = df.reset_index().groupby("cond")["delta"].agg(["mean", "std"])
        ax[j].bar(a.index, a["mean"], yerr=a["std"], capsize=2)
        ax[j].axhline(0, color="k", lw=0.6)
        ax[j].set(ylabel=lab)
    fig.tight_layout()
    fig.savefig(out / "fig_int8_calib.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------------- tables
def pm(m, s, fmt="{:.3f}"):
    return fmt.format(m) if not np.isfinite(s) else (fmt + r"\,$\pm$\," + fmt).format(m, s)


def main_table(cfg, M, C, cost, arch, score):
    src = cfg["source_condition"]
    shifted = [c for c in cfg["conditions"] if c != src]
    Mr = M[(M.arch == arch) & (M.split == "test") & (M.calib == "raw")]
    Mt = M[(M.arch == arch) & (M.split == "test") & (M.calib == "ts")]
    Cc = C[(C.arch == arch) & (C.split == "test") & (C.score == score)]
    per_fold = []
    for (fold, r, p), g in Mr.groupby(["fold", "ratio", "precision"]):
        g = g.set_index("cond")
        gt = Mt[(Mt.fold == fold) & (Mt.ratio == r) & (Mt.precision == p)].set_index("cond")
        cs = Cc[(Cc.fold == fold) & (Cc.ratio == r) & (Cc.precision == p)]
        per_fold.append({"ratio": r, "precision": p, "bacc_src": g.loc[src, "bacc"],
                         "bacc_shift": g.loc[shifted, "bacc"].mean(), "ece_shift": g.loc[shifted, "ece"].mean(),
                         "ece_shift_ts": gt.loc[shifted, "ece"].mean(),
                         "agree": g.loc[shifted, "agree_dense"].mean(),
                         "cov_src": cs[cs.variant == "src"].coverage.min(),
                         "cov_mon": cs[cs.variant == "mondrian"].coverage.min(),
                         "size_mon": cs[cs.variant == "mondrian"].set_size.mean(),
                         "cov_src_mean": cs[cs.variant == "src"].coverage.mean(),
                         "cov_mon_mean": cs[cs.variant == "mondrian"].coverage.mean(),
                         "cov_lobo_mean": cs[cs.variant == "lobo_mondrian"].coverage.mean() if has_lobo(cs) else np.nan,
                         "cov_lobo": cs[cs.variant == "lobo_mondrian"].coverage.min() if has_lobo(cs) else np.nan,
                         "size_lobo": cs[cs.variant == "lobo_mondrian"].set_size.mean() if has_lobo(cs) else np.nan})
    pf = pd.DataFrame(per_fold)
    cols = ["bacc_src", "bacc_shift", "ece_shift", "ece_shift_ts", "agree", "cov_src", "cov_mon", "size_mon"]
    if has_lobo(Cc):
        cols += ["cov_lobo", "size_lobo"]
    mean = pf.groupby(["ratio", "precision"])[cols].mean()
    std = pf.groupby(["ratio", "precision"])[cols].std()
    cst = cost[cost.arch == arch].set_index(["ratio", "precision"])
    head = (r"Pruned & Prec. & Params & Lat.\,[ms] & bAcc$_{\mathrm{src}}$ & bAcc$_{\mathrm{shift}}$ & "
            r"ECE$_{\mathrm{shift}}$ & ECE$^{\mathrm{TS}}_{\mathrm{shift}}$ & Agree & Cov$^{\min}_{\mathrm{src\text{-}CP}}$ & "
            r"Cov$^{\min}_{\mathrm{OP\text{-}CP}}$ & $|C|_{\mathrm{OP\text{-}CP}}$")
    if "cov_lobo" in cols:
        head += r" & Cov$^{\min}_{\mathrm{BL\text{-}CP}}$ & $|C|_{\mathrm{BL\text{-}CP}}$"
    lines = [r"\begin{tabular}{rl r r " + "c" * len(cols) + "}", r"\toprule", head + r" \\", r"\midrule"]
    order = sorted(mean.index, key=lambda k: (k[0], PREC_ORDER.index(k[1]) if k[1] in PREC_ORDER else 99))
    for k in order:
        r, p = k
        params = cst.loc[k, "params"] if k in cst.index else np.nan
        lat = cst.loc[k, "latency_ms"] if k in cst.index and "latency_ms" in cst else np.nan
        cells = [f"{int(r * 100)}\\%", PREC_LABEL.get(p, p).replace("%", r"\%"),
                 f"{params / 1e3:.1f}k" if np.isfinite(params) else "--",
                 f"{lat:.3f}" if np.isfinite(lat) else "--"]
        cells += [pm(mean.loc[k, c], std.loc[k, c], "{:.2f}" if c.startswith("size") else "{:.3f}") for c in cols]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines), pf


SHORT = {"fp32": "FP32", "fp16": "FP16", "int8_src": "I8s", "int8_all": "I8a"}


def selection_table(sel: pd.DataFrame) -> str:
    """Compact LaTeX: per arch x rule, the per-fold choices (pruned%/precision) and mean+-std test metrics."""
    cols = [("cost", "Lat.\\,[ms]", "{:.3f}"), ("bacc_shift", "bAcc$_{\\mathrm{shift}}$", "{:.3f}"),
            ("ece_shift_raw", "ECE$_{\\mathrm{shift}}$", "{:.3f}"),
            ("ece_shift_ts", "ECE$^{\\mathrm{TS}}_{\\mathrm{shift}}$", "{:.3f}"),
            ("cov_worst_src", "Cov$^{\\min}_{\\mathrm{src}}$", "{:.3f}"),
            ("cov_worst_mondrian", "Cov$^{\\min}_{\\mathrm{OP}}$", "{:.3f}"),
            ("set_size_mondrian", "$|C|_{\\mathrm{OP}}$", "{:.2f}")]
    if "cov_worst_lobo_mondrian" in sel:
        cols += [("cov_worst_lobo_mondrian", "Cov$^{\\min}_{\\mathrm{BL}}$", "{:.3f}"),
                 ("set_size_lobo_mondrian", "$|C|_{\\mathrm{BL}}$", "{:.2f}")]
    lines = [r"\begin{tabular}{ll l " + "c" * len(cols) + "}", r"\toprule",
             "Model & Rule & Choice per fold & " + " & ".join(c[1] for c in cols) + r" \\", r"\midrule"]
    for (arch, rule), g in sel.groupby(["arch", "rule"], sort=False):
        g = g.sort_values("fold")
        choice = ", ".join(f"{int(round(r * 100))}/{SHORT.get(p, p)}" for r, p in zip(g.ratio, g.precision))
        if "fallback_dense" in g and g["fallback_dense"].fillna(False).astype(bool).any():
            choice += r"$^\dagger$"
        nice = {"accuracy-only": "accuracy-only", "trust-aware (src CP)": "trust (src-CP)",
                "trust-aware (mondrian CP)": "trust (OP-CP)", "trust-aware (lobo_mondrian CP)": "trust (BL-CP)"}
        cells = [{"resnet1d": "ResNet-1D", "wdcnn": "WDCNN"}.get(arch, arch), nice.get(rule, rule.replace("_", r"\_")),
                 choice]
        for c, _, fmt in cols:
            m, sd = g[c].mean(), g[c].std()
            cells.append(pm(m, sd if len(g) > 1 else np.nan, fmt))
        lines.append(" & ".join(cells) + r" \\")
    any_fb = "fallback_dense" in sel and sel["fallback_dense"].fillna(False).astype(bool).any()
    note = (r"Choice = pruned\%/precision per fold (I8s/I8a: INT8 calibrated on source/all conditions)"
            + (r"; $^\dagger$ no candidate met the constraint in some fold, dense FP32 kept." if any_fb else "."))
    lines += [r"\bottomrule", r"\multicolumn{%d}{l}{\footnotesize %s}\\" % (3 + len(cols), note), r"\end{tabular}"]
    return "\n".join(lines)


def jetson_table(cfg) -> str | None:
    """Table 'Jetson Orin Nano, batch 1': selected engines, GPU latency, speed-up, end-to-end latency, energy."""
    f = path(cfg, cfg["jetson"]["out_dir"]) / "bench" / "bench.csv"
    if not f.exists():
        return None
    b = pd.read_csv(f)
    need = {"trt_gpu_median_ms", "py_lat_p50_ms", "dyn_energy_mj_per_inf", "params"}
    if not need <= set(b.columns):  # e.g. ONNX Runtime dry run (self-test): no TensorRT timings
        return None
    lab = {"fp32": "FP32", "fp16": "FP16", "int8_src": "INT8 (src)", "int8_all": "INT8 (all)"}
    name = {"resnet1d": "ResNet-1D", "wdcnn": "WDCNN"}
    sel = [("resnet1d", 0.0, "fp32"), ("resnet1d", 0.0, "fp16"), ("resnet1d", 0.0, "int8_all"),
           ("resnet1d", 0.5, "fp32"), ("resnet1d", 0.85, "fp32"), ("resnet1d", 0.85, "fp16"),
           ("resnet1d", 0.85, "int8_all"),
           ("wdcnn", 0.0, "fp32"), ("wdcnn", 0.0, "int8_all"), ("wdcnn", 0.5, "fp32"), ("wdcnn", 0.85, "fp32"),
           ("wdcnn", 0.85, "int8_all")]
    lines = [r"\begin{tabular}{l r l r r r r r}", r"\toprule",
             r"Model & Pruned & Prec. & Params & GPU lat. [ms] & Speed-up & End-to-end [ms] & $E_{\mathrm{dyn}}$ [mJ] \\",
             r"\midrule"]
    last = None
    for a, r, p in sel:
        rows = b[(b.arch == a) & (b.ratio == r) & (b.precision == p)]
        base = b[(b.arch == a) & (b.ratio == 0.0) & (b.precision == "fp32")]
        if rows.empty or base.empty:
            continue
        x, b0 = rows.iloc[0], base.iloc[0]
        if last is not None and a != last:
            lines.append(r"\midrule")
        last = a
        lines.append(f"{name.get(a, a)} & {int(round(r * 100))}\\% & {lab[p]} & {x.params / 1e3:.1f}k & "
                     f"{x.trt_gpu_median_ms:.3f} & {b0.trt_gpu_median_ms / x.trt_gpu_median_ms:.1f}$\\times$ & "
                     f"{x.py_lat_p50_ms:.2f} & {x.dyn_energy_mj_per_inf:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main():
    p = base_parser("Selection, figures, tables")
    a = p.parse_args()
    cfg = load_config(a.config, a.set)
    an = cfg["analysis"]
    res = path(cfg, an["results_dir"])
    M = pd.read_csv(res / "metrics.csv")
    C = pd.read_csv(res / "cp.csv")
    man = json.load(open(path(cfg, cfg["export"]["dir"]) / "manifest.json"))
    figs, tabs = path(cfg, an.get("figures_dir", "paper/figures")), path(cfg, an.get("tables_dir", "paper/tables"))
    figs.mkdir(parents=True, exist_ok=True)
    tabs.mkdir(parents=True, exist_ok=True)
    if (M.precision != "ws_fp32").any():  # Jetson results present -> report deployed engines only
        M, C = M[M.precision != "ws_fp32"], C[C.precision != "ws_fp32"]
    else:
        print("[INFO] only workstation (PyTorch FP32) logits found - reporting those")
    bench = load_bench(cfg)
    cost = cost_table(M, bench)
    if bench is None:
        print("[INFO] no bench.csv yet - selection uses a MACs x bytes proxy for cost; Pareto figure skipped")
    sel = select(cfg, M, C, cost)
    sel.to_csv(res / "selection.csv", index=False)
    main_prec = "fp32" if "fp32" in set(M.precision) else "ws_fp32"
    summary = ["# Results summary (auto-generated)\n",
               f"alpha={an['alpha']}, score={an['score']}, precisions present: {sorted(set(M.precision))}\n",
               "Columns (test bearings, mean over folds): bacc_src = balanced acc. at source condition; "
               "bacc_shift / ece_shift = mean over shifted conditions (ece uncalibrated, ece_shift_ts after "
               "temperature scaling); agree = label agreement with dense FP32 on shifted conditions; "
               "cov_src / cov_mon = worst-condition coverage of source-calibrated / operating-point (Mondrian) "
               f"conformal sets (target {1 - an['alpha']:.2f}); size_mon = mean Mondrian set size; cov_lobo / size_lobo = "
               "bearing-level operating-point conformal (calibrated on out-of-bearing scores, tcfd.lobo); *_mean = coverage "
               "averaged over the four conditions instead of the worst one.\n"]
    for arch in sorted(set(M.arch)):
        fig_acc_ece(cfg, M, arch, main_prec, figs)
        fig_coverage(cfg, C, arch, figs, an["score"])
        fig_reliability(cfg, man, arch, figs)
        fig_pareto(cfg, M, C, cost, arch, figs, an["score"])
        tex, pf = main_table(cfg, M, C, cost, arch, an["score"])
        (tabs / f"tab_main_{arch}.tex").write_text(tex)
        summary.append(f"\n## {arch}\n")
        summary.append(pf.groupby(["ratio", "precision"]).mean(numeric_only=True).round(3).to_markdown())
    fig_int8_calib(cfg, M, C, figs, an["score"])
    s = sel.groupby(["arch", "rule"]).agg(
        ratio=("ratio", "mean"), cost=("cost", "mean"), bacc_shift=("bacc_shift", "mean"),
        ece_shift_ts=("ece_shift_ts", "mean"), cov_worst_src=("cov_worst_src", "mean"),
        cov_worst_mondrian=("cov_worst_mondrian", "mean"), set_size_mondrian=("set_size_mondrian", "mean"),
        **({"cov_worst_lobo": ("cov_worst_lobo_mondrian", "mean"), "set_size_lobo": ("set_size_lobo_mondrian", "mean")}
           if "cov_worst_lobo_mondrian" in sel else {}))
    (tabs / "tab_selection.tex").write_text(selection_table(sel))
    jt = jetson_table(cfg)
    if jt:
        (tabs / "tab_jetson.tex").write_text(jt)
    summary.append("\n## Deployment rule (mean over folds)\n")
    summary.append(s.round(3).to_markdown())
    summary.append("\n\nPer-fold choices:\n")
    summary.append(sel.round(3).to_markdown(index=False))
    if bench is not None:
        summary.append("\n\n## Jetson benchmark\n")
        summary.append(bench.round(4).to_markdown(index=False))
    (res / "summary.md").write_text("\n".join(summary) + "\n")
    print(f"Wrote figures to {figs}, tables to {tabs}, {res/'selection.csv'}, {res/'summary.md'}")


if __name__ == "__main__":
    main()
