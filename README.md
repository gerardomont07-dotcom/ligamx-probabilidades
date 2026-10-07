# Liga MX en probabilidades

Prototipo que estima probabilidades de local / empate / visita para partidos de Liga MX,
mide qué tan bien calibradas están y las publica en un sitio estático.

**Solo informativo. Sin dinero real. Mayores de 18. Si el juego te afecta, busca ayuda.**
No es un servicio de picks ni de apuestas, no promete ganancias y no enlaza a casas de apuestas.

## Uso

```bash
python -m venv .venv
.venv\Scripts\activate          # en macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pytest                          # 14 pruebas
python run.py                   # descarga datos, corre modelos, evalúa y regenera docs/
```

Opciones de `run.py`: `--tune` (reajusta hiperparámetros, solo con datos anteriores a 2023),
`--offline` (no descarga), `--no-commit` (no hace commit de git).

El sitio queda en `docs/` (HTML + CSS, sin JavaScript ni backend). Para publicarlo en
GitHub Pages se sirve la carpeta `docs/` de la rama principal.

### Próximos partidos

`MEX.csv` solo trae partidos ya jugados. En cada ejecución, `run.py` baja el calendario de los
próximos 7 días desde la API pública de ESPN (sin clave, no oficial: puede cambiar sin aviso) y
reescribe `data/fixtures.csv`. Si la descarga falla, usa el archivo tal como esté, así que
también se puede llenar a mano:

```
Date,Time,Home,Away,OddsH,OddsD,OddsA
20/10/2026,02:00,Toluca,Club America,2.40,3.30,3.00
```

Hora en UTC, nombres de equipo como en `MEX.csv`, cuotas opcionales (solo se muestran como
probabilidad implícita de referencia; la descarga automática las deja vacías y borra las que
se hayan puesto a mano). Si un nombre no existe, `run.py` falla y lista los válidos.

Cada partido se registra una sola vez por versión del modelo, la primera vez que aparece en el
calendario. `MEX.csv` se actualiza con algunos días de retraso, así que una predicción puede
no incluir los resultados de la jornada inmediata anterior.

### Registro auditable

Cada ejecución agrega a `data/predictions_log.csv` las predicciones de partidos que aún no
empiezan, con fecha y hora UTC, versión del modelo y un hash encadenado con la fila anterior.
Nunca se reescriben filas; si alguien altera una, `run.py` y las pruebas fallan. El hash
detecta ediciones parciales; la garantía frente a una reescritura completa es el historial de
git (un commit por ejecución), así que conviene subirlo a un remoto público.

## Datos

- Fuente: <https://www.football-data.co.uk/new/MEX.csv>. 4,743 partidos del 21/07/2012 al 28/09/2026.
- 6.5% sin cuota de cierre de Pinnacle. Casi todos los faltantes son recientes: 201 de 336
  partidos de 2025/26 y los 88 de 2026/27. Por eso también se compara contra el promedio del
  mercado (`AvgC*`), que sí está completo.
- Sin duplicados ni marcadores inconsistentes con el resultado.
- Equipos: Monarcas Morelia y Mazatlán FC se tratan como la misma franquicia (mudanza de 2020).
  Los demás cambios (Chiapas, Veracruz, Dorados, Leones Negros, Lobos BUAP, Atlante en 2026)
  son clubes distintos; un club que entra o regresa tras más de 400 días arranca con Elo 1450.
- Torneos: Apertura = julio a diciembre, Clausura = enero a junio. El CSV no marca la liguilla,
  así que se trata como cualquier otro partido.

## Modelos

Todos son walk-forward: la predicción de un partido solo usa partidos anteriores. Una prueba
automática altera resultados futuros y verifica que ninguna predicción pasada cambie.

| Modelo | Qué hace | Hiperparámetros elegidos (datos < 2023) |
|---|---|---|
| Elo | Ventaja de local, regresión a la media por torneo, multiplicador por diferencia de goles, empate con el modelo de Davidson | K = 7.6, localía = 84, ν = 0.80, regresión ≈ 0 |
| Dixon-Coles | Poisson con ataque/defensa por equipo y decaimiento temporal; se reajusta cada mes | ξ = 0.0015 por día |
| Logística multinomial | Diferencia de Elo, forma (últimos 5), días de descanso; se reentrena cada torneo | C = 0.01 |

Validación (jul 2014 a dic 2022), log-loss: Elo 1.0458, Dixon-Coles 1.0451, Logística 1.0504.
El sitio usa Dixon-Coles por ser el mejor en validación.

## Resultados (prueba: 1,275 partidos desde enero de 2023)

El conjunto de prueba se evaluó una sola vez, después de fijar los hiperparámetros.

Contra Pinnacle, en los 982 partidos de prueba que tienen cuota de cierre:

| | Log-loss | Brier | Diferencia vs. Pinnacle | IC 95% (bootstrap) |
|---|---|---|---|---|
| Pinnacle (sin margen) | 0.9820 | 0.5846 | — | — |
| Dixon-Coles | 1.0027 | 0.5991 | +0.0208 | +0.0093 a +0.0321 |
| Logística | 1.0039 | 0.6000 | +0.0220 | +0.0100 a +0.0337 |
| Elo | 1.0055 | 0.6009 | +0.0235 | +0.0110 a +0.0357 |

Contra el promedio del mercado, en los 1,275 partidos: Dixon-Coles +0.0227 (IC +0.0133 a
+0.0321), Logística +0.0243, Elo +0.0266. Promedio del mercado: log-loss 0.9890.

**Conclusión: ningún modelo le gana al mercado.** Los tres son peores que Pinnacle por unas
0.02 unidades de log-loss y el intervalo de confianza excluye el cero. Dixon-Coles y la
logística mejoran al Elo simple en 0.002 a 0.003, una diferencia menor que el ruido. La
calibración sí es razonable (error de calibración esperado de 0.008 a 0.016): cuando el modelo
dice 40%, ocurre cerca del 40% de las veces. Es decir, el modelo es honesto pero menos
informado que el mercado.

Línea base histórica: Elo en los 4,437 partidos con Pinnacle da 1.0406 contra 1.0232
(diferencia +0.017). La medición previa era 1.0455 contra 1.0222; no coincide exactamente
porque aquí los hiperparámetros están ajustados y la ventana no es idéntica.

Detalle completo en `results/metrics.json`; gráficas en `results/`.

## Limitaciones

- Solo usa resultados y marcadores. No sabe de lesiones, alineaciones, rotaciones ni xG.
- Se compara contra cuotas de cierre, que ya incorporan toda la información previa al partido.
- La comparación con Pinnacle no cubre la mayor parte de 2025/26 ni 2026/27 por falta de cuotas.
- La liguilla no se distingue de la fase regular.
- Dixon-Coles se reajusta una vez al mes y ρ se estima en un segundo paso.
- No se probó un ensamble de los tres modelos.
- Las horas de `MEX.csv` no vienen documentadas como UTC; solo se usan para ordenar partidos.

## Datos extra que valdría la pena conseguir

Sin verificar disponibilidad ni términos de uso actuales; revisar antes de usar.

| Dato | Para qué | Dónde buscar |
|---|---|---|
| Calendario de próximos partidos | Dejar de llenar `fixtures.csv` a mano | API-Football (plan gratuito limitado), TheSportsDB |
| xG por partido | Es la mejora con más probabilidad de acercarse al mercado | FBref, FotMob, Sofascore (sus términos restringen el scraping) |
| Alineaciones y lesiones | Rotaciones entre liga y copas, bajas | API-Football (de pago para histórico), Transfermarkt |
| Cuotas de apertura | Medir contra una línea menos informada que el cierre | The Odds API, guardando capturas propias |
