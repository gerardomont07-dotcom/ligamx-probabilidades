"""Carga y limpieza de MEX.csv (football-data.co.uk) y de data/fixtures.csv."""
import urllib.request

import numpy as np
import pandas as pd

URL = "https://www.football-data.co.uk/new/MEX.csv"

# Franquicias que se mudaron: el plantel y la plaza siguen, así que heredan historial.
# Monarcas Morelia -> Mazatlán FC (2020). Los demás cambios (Chiapas, Veracruz, Atlante
# 2026, ascensos) son clubes distintos y arrancan como equipo nuevo.
ALIASES = {"Monarcas": "Mazatlan FC"}


def download(path):
    urllib.request.urlretrieve(URL, path)


def torneo(date):
    """Apertura = jul-dic, Clausura = ene-jun (la liguilla cae dentro de su torneo)."""
    return ("A" if date.month >= 7 else "C") + str(date.year)


def devig(odds):
    """Probabilidades sin margen, método proporcional. odds: array (n, 3)."""
    inv = 1.0 / np.asarray(odds, dtype=float)
    return inv / inv.sum(axis=1, keepdims=True)


def load_matches(path):
    raw = pd.read_csv(path, encoding="utf-8-sig")
    df = pd.DataFrame({
        "season": raw["Season"],
        "date": pd.to_datetime(raw["Date"] + " " + raw["Time"].fillna("00:00"), format="%d/%m/%Y %H:%M"),
        "home": raw["Home"].str.strip().replace(ALIASES),
        "away": raw["Away"].str.strip().replace(ALIASES),
        "hg": raw["HG"].astype(float),
        "ag": raw["AG"].astype(float),
        "res": raw["Res"],
    })
    for name, cols in {"pin": ["PSCH", "PSCD", "PSCA"], "avg": ["AvgCH", "AvgCD", "AvgCA"]}.items():
        p = devig(raw[cols].to_numpy())
        for j, o in enumerate("hda"):
            df[f"{name}_{o}"] = p[:, j]

    df = df.dropna(subset=["hg", "ag"]).drop_duplicates(["date", "home", "away"])
    expected = np.where(df.hg > df.ag, "H", np.where(df.hg < df.ag, "A", "D"))
    bad = df[df.res != expected]
    if len(bad):
        raise ValueError(f"Resultado inconsistente con el marcador en {len(bad)} filas:\n{bad.head()}")
    df = df.sort_values("date", kind="stable").reset_index(drop=True)
    df["torneo"] = df.date.map(torneo)
    return df


def team_changes(df):
    """Por temporada: equipos que entran y salen respecto a la anterior."""
    out, prev = [], None
    for season, g in df.groupby("season", sort=True):
        teams = set(g.home) | set(g.away)
        if prev is not None and teams != prev:
            out.append({"season": season, "entran": sorted(teams - prev), "salen": sorted(prev - teams)})
        prev = teams
    return out


def load_fixtures(path, known_teams):
    """Próximos partidos. Date dd/mm/yyyy, Time HH:MM en UTC. Cuotas opcionales."""
    raw = pd.read_csv(path, dtype={"Date": str, "Time": str, "Home": str, "Away": str})
    if raw.empty:
        return pd.DataFrame(columns=["date", "home", "away", "mkt_h", "mkt_d", "mkt_a"])
    fx = pd.DataFrame({
        "date": pd.to_datetime(raw["Date"] + " " + raw["Time"].fillna("00:00"), format="%d/%m/%Y %H:%M"),
        "home": raw["Home"].str.strip().replace(ALIASES),
        "away": raw["Away"].str.strip().replace(ALIASES),
    })
    unknown = (set(fx.home) | set(fx.away)) - set(known_teams)
    if unknown:
        raise ValueError(f"Equipos desconocidos en fixtures.csv: {sorted(unknown)}. Válidos: {sorted(known_teams)}")
    p = devig(raw[["OddsH", "OddsD", "OddsA"]].to_numpy())
    fx["mkt_h"], fx["mkt_d"], fx["mkt_a"] = p[:, 0], p[:, 1], p[:, 2]
    return fx.sort_values("date").reset_index(drop=True)


def with_fixtures(df, fx):
    """Agrega los próximos partidos al final (sin resultado) para predecirlos walk-forward."""
    if fx.empty:
        return df
    extra = fx[["date", "home", "away"]].copy()
    extra["torneo"] = extra.date.map(torneo)
    return pd.concat([df, extra], ignore_index=True)
