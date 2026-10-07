"""Genera el sitio estático (HTML + CSS, sin JS ni backend) en español."""
import shutil
from html import escape

from .evaluate import NAMES

AVISO = "Solo informativo. Sin dinero real. Mayores de 18. Si el juego te afecta, busca ayuda."

CSS = """
:root{--bg:#fff;--fg:#1a1a1a;--mut:#666;--line:#ddd;--acc:#0a6b3d}
@media(prefers-color-scheme:dark){:root{--bg:#121212;--fg:#eee;--mut:#aaa;--line:#333;--acc:#5fd39a}}
*{box-sizing:border-box}
body{margin:0;font:16px/1.5 system-ui,sans-serif;background:var(--bg);color:var(--fg)}
header,main,footer{max-width:760px;margin:0 auto;padding:16px}
nav a{margin-right:14px;color:var(--acc)}
h1{font-size:1.4rem;margin:.2em 0}h2{font-size:1.1rem;margin-top:1.6em}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:.95rem}
th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
img{max-width:100%;height:auto;background:#fff}
.mut{color:var(--mut);font-size:.9rem}
footer{border-top:1px solid var(--line);margin-top:32px;color:var(--mut);font-size:.9rem}
"""


def _page(title, body):
    return f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} · Liga MX en probabilidades</title><link rel="stylesheet" href="style.css"></head>
<body><header><h1>Liga MX en probabilidades</h1>
<nav><a href="index.html">Próximos partidos</a><a href="equipos.html">Equipos</a><a href="rendimiento.html">Qué tan bien funciona</a></nav></header>
<main>{body}</main>
<footer><p><strong>{AVISO}</strong></p>
<p>Proyecto de análisis estadístico. No es un servicio de pronósticos ni de apuestas, no recomienda apostar y no tiene relación con casas de apuestas.</p></footer>
</body></html>"""


def _table(head, rows, num_from=1):
    cls = lambda j: ' class="n"' if j >= num_from else ""
    th = "".join(f"<th{cls(j)}>{h}</th>" for j, h in enumerate(head))
    trs = "".join("<tr>" + "".join(f"<td{cls(j)}>{c}</td>" for j, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="scroll"><table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table></div>'


def _pct(p):
    return f"{100 * p:.0f}%"


def _index(upcoming, model):
    if not upcoming:
        return "<h2>Próximos partidos</h2><p>No hay partidos cargados por ahora.</p>"
    rows = []
    for u in upcoming:
        mkt = " / ".join(map(_pct, u["mkt"])) if u["mkt"] else "—"
        rows.append([f"{u['kickoff']:%d/%m %H:%M}", f"{escape(u['home'])} – {escape(u['away'])}",
                     *map(_pct, u["p"]), mkt])
    return (f"<h2>Próximos partidos</h2><p class='mut'>Probabilidades del modelo {NAMES[model]}. Horas en UTC. "
            "La columna de mercado son las probabilidades implícitas sin margen, como referencia.</p>"
            + _table(["Fecha", "Partido", "Local", "Empate", "Visita", "Mercado (L/E/V)"], rows, num_from=2))


def _verdict(v):
    lo, hi = v["ic95"]
    return "peor que el mercado" if lo > 0 else "mejor que el mercado" if hi < 0 else "sin diferencia distinguible"


def _rendimiento(m):
    pin, avg = m["mercado"]["pin"], m["mercado"]["avg"]
    rows = [[NAMES[k], f"{v['logloss_en_pinnacle']:.4f}", f"{v['brier_en_pinnacle']:.4f}",
             f"{v['vs_pinnacle']['dif_logloss']:+.4f}",
             f"{v['vs_pinnacle']['ic95'][0]:+.4f} a {v['vs_pinnacle']['ic95'][1]:+.4f}", _verdict(v["vs_pinnacle"])]
            for k, v in m["modelos"].items()]
    rows.append([NAMES["pin"], f"{pin['logloss']:.4f}", f"{pin['brier']:.4f}", "—", "—", "referencia"])
    rows2 = [[NAMES[k], f"{v['logloss']:.4f}", f"{v['brier']:.4f}", f"{v['vs_promedio_mercado']['dif_logloss']:+.4f}",
              f"{v['vs_promedio_mercado']['ic95'][0]:+.4f} a {v['vs_promedio_mercado']['ic95'][1]:+.4f}",
              _verdict(v["vs_promedio_mercado"])] for k, v in m["modelos"].items()]
    rows2.append([NAMES["avg"], f"{avg['logloss']:.4f}", f"{avg['brier']:.4f}", "—", "—", "referencia"])
    head = ["Modelo", "Log-loss", "Brier", "Diferencia", "IC 95%", "Lectura"]
    site_model = m["params"]["modelo_sitio"]
    cal = [[r["bin"], r["n"], _pct(r["predicho"]), _pct(r["observado"])] for r in m["modelos"][site_model]["calibracion"]]
    return f"""<h2>Qué tan bien funciona</h2>
<p>Los modelos se ajustaron solo con partidos anteriores a {m['inicio_prueba'][:4]} y se evalúan con los
{m['n_prueba']} partidos jugados desde entonces. Mostramos todos los resultados, también cuando el modelo pierde.
En log-loss y Brier, <strong>menor es mejor</strong>; una diferencia positiva significa que el modelo es peor que el mercado.</p>
<h2>Contra Pinnacle ({m['n_con_pinnacle']} partidos con cuota de cierre)</h2>
{_table(head, rows)}
<h2>Contra el promedio del mercado (los {m['n_prueba']} partidos)</h2>
<p class="mut">La fuente dejó de publicar cuotas de Pinnacle en la segunda mitad de 2025/26, así que esta tabla cubre también los partidos más recientes.</p>
{_table(head, rows2)}
<h2>Calibración</h2>
<p>Si el modelo dice 40%, ¿ocurre cerca del 40% de las veces? Cuanto más pegada a la diagonal, mejor.</p>
<img src="calibracion.png" alt="Curva de calibración de los modelos y de Pinnacle" width="780" height="780">
<p class="mut">Tabla del modelo {NAMES[site_model]} (error de calibración esperado: {m['modelos'][site_model]['ece']:.3f}).</p>
{_table(["Probabilidad predicha", "Casos", "Promedio predicho", "Frecuencia observada"], cal)}
<h2>Log-loss por temporada</h2>
<img src="logloss_temporada.png" alt="Log-loss por temporada de cada modelo y del mercado" width="910" height="585">
<p class="mut">2022/2023 solo incluye el Clausura 2023. El punto de Pinnacle en 2025/2026 cubre únicamente los partidos con cuota de Pinnacle, por lo que no es directamente comparable con las demás líneas de esa temporada.</p>
<h2>Limitaciones</h2>
<ul><li>Solo usa resultados y marcadores; no sabe de lesiones, alineaciones ni xG.</li>
<li>Se compara contra cuotas de cierre, que ya incorporan toda la información disponible antes del partido.</li>
<li>La liguilla se trata como cualquier otro partido.</li></ul>
<p class="mut">Datos: football-data.co.uk, {m['datos']['partidos']} partidos de {m['datos']['desde']} a {m['datos']['hasta']}. Versión del modelo {m['version']}.</p>"""


def build(outdir, upcoming, teams, metrics, results_dir):
    model = metrics["params"]["modelo_sitio"]
    pages = {
        "index.html": ("Próximos partidos", _index(upcoming, model)),
        "equipos.html": ("Equipos", "<h2>Equipos</h2><p class='mut'>Elo: 1500 es el promedio. Forma: últimos 5 partidos, "
                         "del más antiguo al más reciente (G ganó, E empató, P perdió).</p>"
                         + _table(["Equipo", "Elo", "Forma"], [[escape(t["equipo"]), t["elo"], t["forma"]] for t in teams])),
        "rendimiento.html": ("Qué tan bien funciona", _rendimiento(metrics)),
    }
    for name, (title, body) in pages.items():
        with open(f"{outdir}/{name}", "w", encoding="utf-8") as f:
            f.write(_page(title, body))
    with open(f"{outdir}/style.css", "w", encoding="utf-8") as f:
        f.write(CSS)
    for img in ("calibracion.png", "logloss_temporada.png"):
        shutil.copy(f"{results_dir}/{img}", f"{outdir}/{img}")
