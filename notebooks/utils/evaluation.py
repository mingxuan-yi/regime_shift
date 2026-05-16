"""Evaluation suite: edge metrics, regime metrics, per-Saggioro-type recall.

The ``METRIC_DESCRIPTIONS`` and ``TYPE_DESCRIPTIONS`` dictionaries provide a
one-line plain-English description of every metric, suitable for printing
alongside results tables.
"""

from typing import Dict, List, Tuple

import numpy as np


EdgeKey = Tuple[str, str, int]


# ── plain-English descriptions of every metric reported by evaluate_full ──

METRIC_DESCRIPTIONS = {
    "regime_accuracy":
        "Fraction of timesteps where the predicted regime label matches GT. "
        "Label-permutation invariant: takes the better of the two label assignments.",
    "changepoint_err":
        "Absolute distance (in time steps) between the predicted single changepoint "
        "and the true one. Lower is better.",
    "Rk_f1":
        "Harmonic mean of precision and recall on regime k's directed lag-1 edges. "
        "Treats each edge as the tuple (parent, child, lag); a wrong direction is "
        "double-counted (1 FP + 1 FN).",
    "Rk_precision":
        "TP / (TP + FP) over regime k's directed edges. High precision = few spurious edges.",
    "Rk_recall":
        "TP / (TP + FN) over regime k's directed edges. High recall = few missed edges.",
    "Rk_shd":
        "Structural Hamming Distance: number of single-edge edits "
        "(add / delete / reverse) needed to turn the predicted DAG into the true one. "
        "A reversal counts as 1 edit (vs 2 in F1).",
    "Rk_phi_rmse":
        "RMSE between predicted and true lag-1 coefficient matrices Phi_k "
        "(over all N x N entries). Captures coefficient-magnitude accuracy "
        "in addition to structural identity.",
}

TYPE_DESCRIPTIONS = {
    "appear":
        "Edges present in R2 only (and whose reverse is not in R1). "
        "Recall = correctly identified appearances / total true appearances. "
        "An edge is recovered when the method places it in pred R2 but not in pred R1.",
    "remove":
        "Edges present in R1 only (and whose reverse is not in R2). "
        "Recall = correctly identified removals / total true removals. "
        "An edge is recovered when present in pred R1 but absent in pred R2.",
    "reversal":
        "Variable pairs (a, b) with a -> b in R1 and b -> a in R2 (or vice versa). "
        "Recall = pairs where the method gets BOTH directions right (R1 direction in pred R1 only, "
        "R2 direction in pred R2 only). One reversal contributes 2 directed-edge changes.",
    "magnitude_shift":
        "Edges in both regimes whose coefficient changes by more than the magnitude_threshold "
        "(default 0.20), with the same sign. Recall = present in both pred R1 and pred R2 "
        "(this metric does NOT verify coefficient accuracy; use Phi_RMSE for that).",
    "sign_flip":
        "Edges in both regimes whose coefficient changes sign across regimes. "
        "Treated as a special case of magnitude_shift.",
    "unchanged":
        "Edges present in both regimes with similar coefficients. "
        "Recall = present in both pred R1 and pred R2 (sanity check that the method "
        "does not spuriously perturb structurally stable edges).",
}


# ── core metric functions ──────────────────────────────────────────────────

def edges_to_set(edges, exclude_self_loops=True):
    out = set()
    for parent, child, lag, _ in edges:
        if exclude_self_loops and parent == child:
            continue
        out.add((parent, child, lag))
    return out


def regime_label_accuracy(pred_seq, true_seq):
    pred_seq = np.asarray(pred_seq); true_seq = np.asarray(true_seq)
    return max(
        (pred_seq == true_seq).mean(),
        (pred_seq == (3 - true_seq)).mean(),
    )


def changepoint_error(pred_seq, true_seq):
    def _cp(s):
        d = np.diff(s); idx = np.where(d != 0)[0]
        return idx[0] + 1 if len(idx) else None
    p, t = _cp(pred_seq), _cp(true_seq)
    if p is None or t is None:
        return np.inf
    return abs(p - t)


def shd(pred_edges, true_edges):
    return len(edges_to_set(pred_edges) ^ edges_to_set(true_edges))


def edge_prf1(pred_edges, true_edges):
    p_set = edges_to_set(pred_edges); t_set = edges_to_set(true_edges)
    tp = len(p_set & t_set); fp = len(p_set - t_set); fn = len(t_set - p_set)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def coefficient_rmse(pred_Phi, true_Phi):
    return float(np.sqrt(np.mean((pred_Phi - true_Phi) ** 2)))


def per_type_recall(pred_edges_dict, ds):
    """Per-Saggioro-type recall."""
    pred_R1 = edges_to_set(pred_edges_dict[1])
    pred_R2 = edges_to_set(pred_edges_dict[2])
    true_R1 = edges_to_set(ds.edges_dict[1])
    by_tag: Dict[str, List[EdgeKey]] = {}
    for k, t in ds.tags.items():
        by_tag.setdefault(t, []).append(k)
    out = {}
    for tag, keys in by_tag.items():
        ks = set(keys)
        if tag == "appear":
            tp = len((pred_R2 & ks) - pred_R1); fn = len(ks) - tp
        elif tag == "remove":
            tp = len((pred_R1 & ks) - pred_R2); fn = len(ks) - tp
        elif tag == "reversal":
            seen, pairs = set(), []
            for (p, c, l) in keys:
                key = (frozenset((p, c)), l)
                if key in seen: continue
                seen.add(key)
                if (p, c, l) in true_R1:
                    r1d, r2d = (p, c, l), (c, p, l)
                else:
                    r1d, r2d = (c, p, l), (p, c, l)
                pairs.append((r1d, r2d))
            tp = sum(
                1 for r1d, r2d in pairs
                if r1d in pred_R1 and r1d not in pred_R2
                and r2d in pred_R2 and r2d not in pred_R1
            )
            fn = len(pairs) - tp
            out[tag] = {"recall": tp / max(len(pairs), 1), "n": len(pairs), "tp": tp, "fn": fn}
            continue
        else:  # magnitude_shift / sign_flip / unchanged
            tp = len(ks & pred_R1 & pred_R2); fn = len(ks) - tp
        out[tag] = {"recall": tp / max(tp + fn, 1), "n": len(ks), "tp": tp, "fn": fn}
    return out


def evaluate_full(pred_seq, pred_edges_dict, pred_Phi_dict, ds):
    """Full evaluation: regime + per-regime + per-Saggioro-type metrics."""
    res = {
        "regime_accuracy":  regime_label_accuracy(pred_seq, ds.regime_seq),
        "changepoint_err":  int(changepoint_error(pred_seq, ds.regime_seq)),
    }
    for k in (1, 2):
        prf = edge_prf1(pred_edges_dict[k], ds.edges_dict[k])
        res[f"R{k}_shd"]       = shd(pred_edges_dict[k], ds.edges_dict[k])
        res[f"R{k}_precision"] = prf["precision"]
        res[f"R{k}_recall"]    = prf["recall"]
        res[f"R{k}_f1"]        = prf["f1"]
        if pred_Phi_dict is not None and k in pred_Phi_dict:
            res[f"R{k}_phi_rmse"] = coefficient_rmse(pred_Phi_dict[k], ds.Phi_dict[k])
    pt = per_type_recall(pred_edges_dict, ds)
    for tag, m in pt.items():
        res[f"type_{tag}_recall"] = m["recall"]
        res[f"type_{tag}_n"]      = m["n"]
    return res
