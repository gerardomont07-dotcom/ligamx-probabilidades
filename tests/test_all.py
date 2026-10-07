import os
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from ligamx import data, evaluate, models, predlog, site

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "MEX.csv")


@pytest.fixture(scope="module")
def df():
    return data.load_matches(RAW)


# --- datos ---

def test_carga(df):
    assert len(df) > 4700
    assert df.date.is_monotonic_increasing
    assert not df.duplicated(["date", "home", "away"]).any()
    assert set(df.res) == {"H", "D", "A"}
    assert "Monarcas" not in set(df.home) | set(df.away)  # alias aplicado
    assert 0.03 < df.pin_h.isna().mean() < 0.15


def test_torneo():
    assert data.torneo(pd.Timestamp("2024-07-05")) == "A2024"
    assert data.torneo(pd.Timestamp("2025-01-10")) == "C2025"
    assert data.torneo(pd.Timestamp("2015-06-01")) == "C2015"  # final de liguilla


def test_devig():
    p = data.devig([[2.0, 3.5, 4.0], [1.5, 4.0, np.nan]])
    assert p[0].sum() == pytest.approx(1)
    assert p[0, 0] == pytest.approx(0.5 / (0.5 + 1 / 3.5 + 0.25))
    assert np.isnan(p[1]).all()


def test_cambios_de_equipos(df):
    ch = {c["season"]: c for c in data.team_changes(df)}
    assert "Necaxa" in ch["2016/2017"]["entran"]
    assert "Veracruz" in ch["2020/2021"]["salen"]


def test_fixtures(tmp_path):
    p = tmp_path / "fx.csv"
    p.write_text("Date,Time,Home,Away,OddsH,OddsD,OddsA\n")
    assert data.load_fixtures(p, {"Toluca"}).empty
    p.write_text("Date,Time,Home,Away,OddsH,OddsD,OddsA\n10/10/2030,02:00,Toluca,Monarcas,,,\n")
    fx = data.load_fixtures(p, {"Toluca", "Mazatlan FC"})
    assert fx.away[0] == "Mazatlan FC" and np.isnan(fx.mkt_h[0])
    p.write_text("Date,Time,Home,Away,OddsH,OddsD,OddsA\n10/10/2030,02:00,Toluca,Inventado FC,,,\n")
    with pytest.raises(ValueError, match="Inventado"):
        data.load_fixtures(p, {"Toluca"})


def test_calendario_espn(df):
    current = df[df.season == df.season.iloc[-1]]
    assert set(current.home) <= set(data.ESPN_NAMES.values())  # todo equipo actual tiene equivalencia
    ev = lambda state, home: {"date": "2026-10-10T01:00Z", "competitions": [{"status": {"type": {"state": state}},
        "competitors": [{"homeAway": "home", "team": {"displayName": home}}, {"homeAway": "away", "team": {"displayName": "León"}}]}]}
    got = data.parse_espn({"events": [ev("pre", "Puebla"), ev("post", "Toluca")]})
    assert got == [(pd.Timestamp("2026-10-10 01:00"), "Puebla", "Club Leon")]  # el ya jugado se omite
    with pytest.raises(ValueError, match="Inventado"):
        data.parse_espn({"events": [ev("pre", "Inventado")]})


# --- modelos ---

def _all_models(d):
    elo_P, eh, ea, _ = models.elo_walk(d)
    return {"elo": elo_P, "dc": models.dc_walk(d), "lr": models.lr_walk(d, elo_P, eh, ea)}


def test_probabilidades_validas_y_sin_fuga(df):
    """Alterar resultados futuros no debe cambiar ninguna predicción pasada."""
    d = df.iloc[:1500].reset_index(drop=True)
    k = 1100
    base = _all_models(d)
    alt = d.copy()
    alt.loc[k:, ["hg", "ag"]] = alt.loc[k:, ["ag", "hg"]].to_numpy() + [[3, 0]]
    changed = _all_models(alt)
    for name, P in base.items():
        assert np.allclose(P.sum(axis=1), 1), name
        assert (P > 0).all(), name
        assert np.array_equal(P[:k + 1], changed[name][:k + 1]), name
        assert not np.array_equal(P[k + 1:], changed[name][k + 1:]), name  # el futuro sí reacciona


def test_fixtures_no_actualizan(df):
    d = df.iloc[:800].reset_index(drop=True)
    fx = pd.DataFrame({"date": [d.date.iloc[-1] + pd.Timedelta(days=3)], "home": [d.home[0]], "away": [d.away[0]]})
    full = data.with_fixtures(d, fx)
    P = _all_models(full)
    for name, base in _all_models(d).items():
        assert np.array_equal(P[name][:800], base), name
        assert P[name][800].sum() == pytest.approx(1), name


def test_elo_premia_al_ganador():
    d = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2020-01-08"]), "torneo": "C2020",
                      "home": ["A", "A"], "away": ["B", "B"], "hg": [3.0, 1.0], "ag": [0.0, 1.0]})
    P, eh, ea, _ = models.elo_walk(d)
    assert eh[1] > 1500 > ea[1]
    assert P[1, 0] > P[0, 0]


def test_dixon_coles_distingue_fuerzas():
    lh, la = np.array([2.0, 1.0]), np.array([1.0, 1.0])
    P = models._dc_probs(lh, la, -0.1)
    assert P[0, 0] > P[1, 0] and P[1, 0] == pytest.approx(P[1, 2])


# --- evaluación ---

def test_metricas():
    P = np.array([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]])
    y = np.array([0, 1])
    assert evaluate.match_logloss(P, y) == pytest.approx([-np.log(0.5), -np.log(0.3)])
    assert evaluate.match_brier(P, y)[0] == pytest.approx(0.25 + 0.09 + 0.04)
    d = np.random.default_rng(1).normal(0.02, 0.1, 500)
    lo, hi = evaluate.bootstrap_ci(d)
    assert lo < d.mean() < hi and evaluate.bootstrap_ci(d) == [lo, hi]  # semilla fija
    rows, ece = evaluate.calibration_table(np.tile([0.5, 0.3, 0.2], (100, 1)), np.repeat([0, 1, 2], [50, 30, 20]))
    assert ece == pytest.approx(0, abs=1e-9) and sum(r["n"] for r in rows) == 300


def test_evaluate_usa_los_mismos_partidos_que_pinnacle(df):
    P = np.tile(models.PRIOR, (len(df), 1))
    m = evaluate.evaluate(df, {"elo": P})
    assert m["n_con_pinnacle"] < m["n_prueba"]
    assert m["mercado"]["pin"]["n"] == m["n_con_pinnacle"]
    assert m["modelos"]["elo"]["vs_pinnacle"]["dif_logloss"] > 0  # una constante no le gana al mercado


# --- registro de predicciones ---

def _pred(day):
    return {"kickoff": datetime(2030, 1, day, 2), "home": "A", "away": "B", "model": "elo", "version": "1", "p": [0.5, 0.3, 0.2]}


def test_registro_solo_agrega_y_detecta_alteraciones(tmp_path):
    path = str(tmp_path / "log.csv")
    now = datetime(2030, 1, 2)
    assert predlog.append(path, [_pred(1), _pred(5)], now=now) == 1  # el partido pasado no se registra
    assert predlog.append(path, [_pred(5), _pred(6)], now=now) == 1  # no se duplica
    rows = predlog.verify(path)
    assert [r["kickoff_utc"][8:10] for r in rows] == ["05", "06"]
    assert rows[1]["prev_hash"] == rows[0]["hash"]

    text = open(path, encoding="utf-8").read()
    open(path, "w", encoding="utf-8", newline="").write(text.replace("0.5000", "0.9000", 1))
    with pytest.raises(ValueError, match="alterado"):
        predlog.verify(path)
    with pytest.raises(ValueError):
        predlog.append(path, [_pred(7)], now=now)


# --- sitio ---

def test_sitio_con_y_sin_partidos(df, tmp_path):
    res = tmp_path / "res"
    res.mkdir()
    P = np.tile(models.PRIOR, (len(df), 1))
    m = evaluate.evaluate(df, {"elo": P})
    m.update(version="t", params={"modelo_sitio": "elo"}, datos={"partidos": len(df), "desde": "a", "hasta": "b"})
    evaluate.plots(m, str(res))
    teams = [{"equipo": "Toluca", "elo": 1600, "forma": "GGEPG"}]
    up = [{**_pred(5), "home": "<b>A</b>", "mkt": None}]
    for upcoming in ([], up):
        site.build(str(tmp_path), upcoming, teams, m, str(res))
        for page in ("index.html", "equipos.html", "rendimiento.html"):
            html = (tmp_path / page).read_text(encoding="utf-8")
            assert site.AVISO in html
            assert "<script" not in html
    assert "&lt;b&gt;A&lt;/b&gt;" in (tmp_path / "index.html").read_text(encoding="utf-8")
