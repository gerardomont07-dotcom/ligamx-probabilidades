"""Registro de predicciones de solo-agregar con hash encadenado.
El hash detecta ediciones accidentales o parciales; la garantía frente a una
reescritura completa es el historial de git (un commit por ejecución)."""
import csv
import hashlib
import os
from datetime import datetime, timezone

FIELDS = ["logged_at_utc", "kickoff_utc", "home", "away", "model", "version",
          "p_home", "p_draw", "p_away", "prev_hash", "hash"]
GENESIS = "0" * 64


def _hash(row):
    return hashlib.sha256("|".join(row[f] for f in FIELDS[:-1]).encode()).hexdigest()


def verify(path):
    """Devuelve las filas; lanza ValueError si alguna fila previa fue alterada."""
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    prev = GENESIS
    for n, r in enumerate(rows, start=2):
        if r["prev_hash"] != prev or r["hash"] != _hash(r):
            raise ValueError(f"Registro de predicciones alterado en la línea {n} de {path}")
        if r["logged_at_utc"] >= r["kickoff_utc"]:
            raise ValueError(f"Predicción registrada después del partido en la línea {n}")
        prev = r["hash"]
    return rows


def append(path, preds, now=None):
    """preds: dicts con kickoff (datetime UTC), home, away, model, version, p (3 floats).
    Solo registra partidos futuros que aún no estén registrados con esa versión."""
    now = now or datetime.now(timezone.utc).replace(tzinfo=None)
    rows = verify(path)
    seen = {(r["kickoff_utc"], r["home"], r["away"], r["version"]) for r in rows}
    prev = rows[-1]["hash"] if rows else GENESIS
    new = []
    for p in preds:
        ko = p["kickoff"].strftime("%Y-%m-%dT%H:%M:%S")
        if p["kickoff"] <= now or (ko, p["home"], p["away"], p["version"]) in seen:
            continue
        row = {"logged_at_utc": now.strftime("%Y-%m-%dT%H:%M:%S"), "kickoff_utc": ko,
               "home": p["home"], "away": p["away"], "model": p["model"], "version": p["version"],
               "p_home": f"{p['p'][0]:.4f}", "p_draw": f"{p['p'][1]:.4f}", "p_away": f"{p['p'][2]:.4f}",
               "prev_hash": prev}
        row["hash"] = prev = _hash(row)
        new.append(row)
    if new:
        fresh = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, FIELDS)
            if fresh:
                w.writeheader()
            w.writerows(new)
    return len(new)
