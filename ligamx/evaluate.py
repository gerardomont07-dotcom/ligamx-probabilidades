"""Métricas, comparación contra el mercado y gráficas."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .models import outcome_idx

TEST_START = "2023-01-01"  # los hiperparámetros solo ven partidos anteriores a esta fecha
NAMES = {"elo": "Elo", "dc": "Dixon-Coles", "lr": "Logística", "pin": "Pinnacle", "avg": "Promedio del mercado"}


def match_logloss(P, y):
    return -np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1))


def match_brier(P, y):
    return ((P - np.eye(3)[y]) ** 2).sum(axis=1)


def bootstrap_ci(diff, n_boot=10000, seed=0):
    """IC95% percentil de la media de diferencias pareadas por partido."""
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), (n_boot, len(diff)))].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def calibration_table(P, y, bins=10):
    """Agrupa las 3 probabilidades de cada partido: predicho vs. frecuencia observada."""
    p, hit = P.ravel(), np.eye(3)[y].ravel()
    b = np.minimum((p * bins).astype(int), bins - 1)
    rows = [{"bin": f"{k / bins:.1f}-{(k + 1) / bins:.1f}", "n": int((b == k).sum()),
             "predicho": float(p[b == k].mean()), "observado": float(hit[b == k].mean())}
            for k in range(bins) if (b == k).any()]
    ece = sum(r["n"] * abs(r["predicho"] - r["observado"]) for r in rows) / len(p)
    return rows, float(ece)


def _versus(P, M, y, seed):
    d = match_logloss(P, y) - match_logloss(M, y)
    return {"dif_logloss": float(d.mean()), "ic95": bootstrap_ci(d, seed=seed)}


def evaluate(df, preds, start=TEST_START, seed=0):
    """df: partidos jugados; preds: {modelo: P alineada con df}. Evalúa desde `start`.
    La comparación con Pinnacle usa solo partidos con cuota de Pinnacle (los mismos
    para modelo y mercado); la del promedio del mercado usa todos."""
    test = (df.date >= start).to_numpy()
    y = outcome_idx(df)
    mkt = {m: df[[f"{m}_h", f"{m}_d", f"{m}_a"]].to_numpy() for m in ("pin", "avg")}
    has_pin = test & ~np.isnan(mkt["pin"][:, 0])
    out = {"inicio_prueba": start, "n_prueba": int(test.sum()), "n_con_pinnacle": int(has_pin.sum()),
           "modelos": {}, "mercado": {}, "por_temporada": []}
    for m, mask in (("pin", has_pin), ("avg", test)):
        out["mercado"][m] = {"n": int(mask.sum()), "logloss": float(match_logloss(mkt[m][mask], y[mask]).mean()),
                             "brier": float(match_brier(mkt[m][mask], y[mask]).mean()),
                             "calibracion": calibration_table(mkt[m][mask], y[mask])[0]}
    for name, P in preds.items():
        cal, ece = calibration_table(P[test], y[test])
        out["modelos"][name] = {
            "logloss": float(match_logloss(P[test], y[test]).mean()),
            "brier": float(match_brier(P[test], y[test]).mean()),
            "ece": ece, "calibracion": cal,
            "logloss_en_pinnacle": float(match_logloss(P[has_pin], y[has_pin]).mean()),
            "brier_en_pinnacle": float(match_brier(P[has_pin], y[has_pin]).mean()),
            "vs_pinnacle": _versus(P[has_pin], mkt["pin"][has_pin], y[has_pin], seed),
            "vs_promedio_mercado": _versus(P[test], mkt["avg"][test], y[test], seed),
        }
    for season in df.season[test].unique():
        s = test & (df.season == season).to_numpy()
        sp = s & has_pin
        row = {"temporada": season, "n": int(s.sum()), "n_pinnacle": int(sp.sum()),
               "avg": float(match_logloss(mkt["avg"][s], y[s]).mean()),
               "pin": float(match_logloss(mkt["pin"][sp], y[sp]).mean()) if sp.any() else None}
        row.update({name: float(match_logloss(P[s], y[s]).mean()) for name, P in preds.items()})
        out["por_temporada"].append(row)
    return out


def plots(metrics, outdir):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Calibración perfecta")
    curves = {**{k: v["calibracion"] for k, v in metrics["modelos"].items()},
              "pin": metrics["mercado"]["pin"]["calibracion"]}
    for name, cal in curves.items():
        cal = [r for r in cal if r["n"] >= 20]
        ax.plot([r["predicho"] for r in cal], [r["observado"] for r in cal], "o-", ms=4, label=NAMES[name])
    ax.set(xlabel="Probabilidad predicha", ylabel="Frecuencia observada",
           title=f"Calibración, partidos desde {metrics['inicio_prueba'][:4]}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{outdir}/calibracion.png", dpi=130)
    plt.close(fig)

    rows = metrics["por_temporada"]
    x = range(len(rows))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name in [*metrics["modelos"], "avg", "pin"]:
        ax.plot(x, [r[name] if r[name] is not None else np.nan for r in rows], "o-", ms=4, label=NAMES[name])
    ax.set_xticks(list(x), [f"{r['temporada']}\n(n={r['n']})" for r in rows], fontsize=8)
    ax.set(ylabel="Log-loss (menor es mejor)", title="Log-loss por temporada")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{outdir}/logloss_temporada.png", dpi=130)
    plt.close(fig)
