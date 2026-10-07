"""Regenera datos, modelos, evaluación y sitio:  python run.py
   --tune       reajusta hiperparámetros (solo con datos anteriores a 2023)
   --offline    no descarga MEX.csv ni el calendario
   --no-commit  no hace commit de git al terminar"""
import argparse
import json
import os
import subprocess

import numpy as np
from scipy.optimize import minimize

from ligamx import data, evaluate, models, predlog, site

VERSION = "0.1.0"  # subir cuando cambie el código de un modelo o sus hiperparámetros
RAW, FIXTURES, PARAMS = "data/raw/MEX.csv", "data/fixtures.csv", "params.json"
LOG, METRICS = "data/predictions_log.csv", "results/metrics.json"
VAL_START = "2014-07-01"  # las dos primeras temporadas solo calientan los ratings


def run_models(df, params):
    elo_P, eh, ea, ratings = models.elo_walk(df, **params["elo"])
    preds = {"elo": elo_P, "dc": models.dc_walk(df, **params["dc"]),
             "lr": models.lr_walk(df, elo_P, eh, ea, **params["lr"])}
    return preds, ratings


def tune(df):
    """Elige hiperparámetros por log-loss en [VAL_START, TEST_START). Nunca ve la prueba."""
    past = df[df.date < evaluate.TEST_START].reset_index(drop=True)
    val = (past.date >= VAL_START).to_numpy()
    y = models.outcome_idx(past)

    def score(P):
        return float(evaluate.match_logloss(P[val], y[val]).mean())

    keys = ["k", "home", "nu", "regress"]
    fit = minimize(lambda x: score(models.elo_walk(past, **dict(zip(keys, x)))[0]),
                   [20, 60, 0.9, 0.1], method="Nelder-Mead", options={"xatol": 0.01, "fatol": 1e-5})
    elo = dict(zip(keys, map(float, fit.x)))
    elo_P, eh, ea, _ = models.elo_walk(past, **elo)
    dc = {xi: score(models.dc_walk(past, xi=xi)) for xi in (0.0005, 0.001, 0.0015, 0.002, 0.003)}
    lr = {C: score(models.lr_walk(past, elo_P, eh, ea, C=C)) for C in (0.01, 0.1, 1.0, 10.0)}
    best_dc, best_lr = min(dc, key=dc.get), min(lr, key=lr.get)
    val_scores = {"elo": float(fit.fun), "dc": dc[best_dc], "lr": lr[best_lr]}
    return {"elo": elo, "dc": {"xi": best_dc}, "lr": {"C": best_lr},
            "validacion_logloss": val_scores, "modelo_sitio": min(val_scores, key=val_scores.get)}


def team_table(df, ratings):
    current = df[df.season == df.season.iloc[-1]]
    rows = []
    for t in sorted(set(current.home) | set(current.away), key=lambda t: -ratings[t]):
        g = df[(df.home == t) | (df.away == t)].tail(5)
        diff = np.where(g.home == t, g.hg - g.ag, g.ag - g.hg)
        rows.append({"equipo": t, "elo": round(ratings[t]), "forma": "".join("G" if d > 0 else "E" if d == 0 else "P" for d in diff)})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in ("--tune", "--offline", "--no-commit"):
        ap.add_argument(flag, action="store_true")
    args = ap.parse_args()

    if not args.offline:
        try:
            data.download(RAW)
        except OSError as e:
            print(f"AVISO: no se pudo descargar MEX.csv ({e}); se usa la copia local.")
        try:
            print(f"Calendario: {data.fetch_fixtures(FIXTURES)} próximos partidos (ESPN)")
        except (OSError, ValueError, KeyError) as e:
            print(f"AVISO: no se pudo actualizar el calendario ({e}); se usa fixtures.csv tal como está.")
    df = data.load_matches(RAW)
    print(f"{len(df)} partidos, {df.date.min():%Y-%m-%d} a {df.date.max():%Y-%m-%d}; "
          f"{df.pin_h.isna().mean():.1%} sin cuota de Pinnacle")

    if args.tune or not os.path.exists(PARAMS):
        with open(PARAMS, "w") as f:
            json.dump(tune(df), f, indent=2)
    with open(PARAMS) as f:
        params = json.load(f)

    fx = data.load_fixtures(FIXTURES, set(df.home) | set(df.away))
    full = data.with_fixtures(df, fx)
    preds, _ = run_models(full, params)
    _, _, _, ratings = models.elo_walk(df, **params["elo"])  # ratings tras el último partido jugado

    n = len(df)
    metrics = evaluate.evaluate(df, {k: P[:n] for k, P in preds.items()})
    metrics.update(version=VERSION, params=params, datos={
        "partidos": n, "desde": f"{df.date.min():%Y-%m-%d}", "hasta": f"{df.date.max():%Y-%m-%d}",
        "sin_pinnacle": float(df.pin_h.isna().mean()), "cambios_de_equipos": data.team_changes(df)})
    # Línea base histórica: Elo walk-forward en todos los partidos con cuota de Pinnacle
    # (incluye el periodo usado para ajustar; solo sirve para comparar con la medición previa).
    pin = df.pin_h.notna().to_numpy()
    y = models.outcome_idx(df)
    metrics["elo_historico_completo"] = {
        "n": int(pin.sum()), "elo": float(evaluate.match_logloss(preds["elo"][:n][pin], y[pin]).mean()),
        "pinnacle": float(evaluate.match_logloss(df[["pin_h", "pin_d", "pin_a"]].to_numpy()[pin], y[pin]).mean())}
    os.makedirs("results", exist_ok=True)
    with open(METRICS, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    evaluate.plots(metrics, "results")

    model = params["modelo_sitio"]
    upcoming = [{"kickoff": r.date.to_pydatetime(), "home": r.home, "away": r.away, "model": model,
                 "version": VERSION, "p": preds[model][n + i].tolist(),
                 "mkt": None if np.isnan(r.mkt_h) else [r.mkt_h, r.mkt_d, r.mkt_a]}
                for i, r in enumerate(fx.itertuples())]
    added = predlog.append(LOG, upcoming)
    site.build("docs", upcoming, team_table(df, ratings), metrics, "results")
    print(f"Sitio regenerado en docs/. Predicciones nuevas registradas: {added}")

    if not args.no_commit:
        subprocess.run(["git", "add", "data", "results", "docs", PARAMS], check=True)
        subprocess.run(["git", "commit", "-q", "-m", f"Regeneración automática (modelo v{VERSION})"])


if __name__ == "__main__":
    main()
