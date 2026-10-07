"""Carga y limpieza de MEX.csv (football-data.co.uk) y de data/fixtures.csv."""
import gzip
import json
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


# --- Calendario automático (ESPN, sin clave; API no oficial, puede cambiar sin aviso) ---

ESPN = "https://site.api.espn.com/apis/site/v2/sports/soccer/mex.1/scoreboard"
ESPN_NAMES = {
    "América": "Club America", "Atlante": "Atlante", "Atlas": "Atlas", "Atlético de San Luis": "Atl. San Luis",
    "Cruz Azul": "Cruz Azul", "FC Juárez": "Juarez", "Guadalajara": "Guadalajara Chivas", "León": "Club Leon",
    "Monterrey": "Monterrey", "Necaxa": "Necaxa", "Pachuca": "Pachuca", "Puebla": "Puebla",
    "Pumas UNAM": "UNAM Pumas", "Querétaro": "Queretaro", "Santos": "Santos Laguna", "Tigres UANL": "Tigres UANL",
    "Tijuana": "Club Tijuana", "Toluca": "Toluca", "Mazatlán FC": "Mazatlan FC",
}


def _get_json(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        body = r.read()
    if body[:2] == b"\x1f\x8b":  # ESPN a veces responde comprimido aunque no se pida
        body = gzip.decompress(body)
    return json.loads(body)


def parse_espn(payload):
    """Partidos aún no iniciados: lista de (kickoff UTC, local, visita) con nombres de MEX.csv."""
    out = []
    for e in payload.get("events", []):
        c = e["competitions"][0]
        if c["status"]["type"]["state"] != "pre":
            continue
        t = {x["homeAway"]: x["team"]["displayName"] for x in c["competitors"]}
        unknown = set(t.values()) - set(ESPN_NAMES)
        if unknown:
            raise ValueError(f"Equipo de ESPN sin equivalencia en ESPN_NAMES: {sorted(unknown)}")
        out.append((pd.Timestamp(e["date"]).tz_convert(None), ESPN_NAMES[t["home"]], ESPN_NAMES[t["away"]]))
    return out


def fetch_fixtures(path, days=7, today=None):
    """Reescribe fixtures.csv con los partidos de los próximos `days` días. Sin cuotas."""
    today = today or pd.Timestamp.now("UTC").tz_convert(None).normalize()
    cal = {pd.Timestamp(d).tz_convert(None).normalize() for d in _get_json(ESPN)["leagues"][0]["calendar"]}
    dates = sorted({today} | {d for d in cal if today <= d <= today + pd.Timedelta(days=days)})
    games = sorted({g for d in dates for g in parse_espn(_get_json(f"{ESPN}?dates={d:%Y%m%d}"))})
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("Date,Time,Home,Away,OddsH,OddsD,OddsA\n")
        f.writelines(f"{k:%d/%m/%Y},{k:%H:%M},{h},{a},,,\n" for k, h, a in games)
    return len(games)
