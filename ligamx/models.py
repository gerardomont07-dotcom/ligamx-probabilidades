"""Modelos walk-forward. Cada función devuelve P (n, 3) = [local, empate, visita],
donde la fila i solo usa partidos anteriores a i. Filas sin resultado (próximos
partidos) se predicen pero no actualizan nada."""
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression

PRIOR = np.array([0.45, 0.27, 0.28])  # frecuencias aproximadas, solo para el arranque
NEW_TEAM = 1450.0  # ponytail: Elo fijo para recién llegados; estimarlo si importa
ABSENT_DAYS = 400  # un club que vuelve tras más de un año cuenta como nuevo


def outcome_idx(df):
    return np.where(df.hg > df.ag, 0, np.where(df.hg < df.ag, 2, 1))


def davidson(d, nu):
    """Probabilidades 1X2 a partir de la diferencia de Elo (modelo de Davidson)."""
    a = 10 ** (d / 400)
    den = a + 1 + nu * np.sqrt(a)
    return np.array([a / den, nu * np.sqrt(a) / den, 1 / den])


def elo_walk(df, k=20.0, home=60.0, nu=0.9, regress=0.1):
    """Elo con ventaja de local, regresión a la media por torneo y multiplicador por goles.
    Devuelve (P, elo_local_previo, elo_visita_previo, ratings_finales)."""
    R, last_t, last_d = {}, {}, {}
    start = df.date.iloc[0]
    n = len(df)
    P, eh, ea = np.empty((n, 3)), np.empty(n), np.empty(n)
    for i, (date, tor, h, a, hg, ag) in enumerate(zip(df.date, df.torneo, df.home, df.away, df.hg, df.ag)):
        for t in (h, a):
            if t not in R or (date - last_d[t]).days > ABSENT_DAYS:
                R[t] = 1500.0 if (date - start).days < 365 else NEW_TEAM
            elif last_t[t] != tor:
                R[t] = 1500 + (R[t] - 1500) * (1 - regress)
            last_t[t], last_d[t] = tor, date
        eh[i], ea[i] = R[h], R[a]
        P[i] = p = davidson(R[h] - R[a] + home, nu)
        if hg == hg:  # no NaN: partido jugado
            gd = abs(hg - ag)
            g = 1.0 if gd <= 1 else 1.5 if gd == 2 else (11 + gd) / 8
            delta = k * g * ((1.0 if hg > ag else 0.5 if hg == ag else 0.0) - (p[0] + 0.5 * p[1]))
            R[h] += delta
            R[a] -= delta
    return P, eh, ea, R


def _fit_poisson(hi, ai, hg, ag, w, nt, x0, ridge):
    def f(x):
        att, dfn = x[2:2 + nt], x[2 + nt:]
        eh = x[0] + x[1] + att[hi] - dfn[ai]
        ea = x[0] + att[ai] - dfn[hi]
        lh, la = np.exp(eh), np.exp(ea)
        nll = -(w * (hg * eh - lh + ag * ea - la)).sum() + ridge * (att @ att + dfn @ dfn)
        rh, ra = w * (lh - hg), w * (la - ag)
        g = np.empty_like(x)
        g[0], g[1] = rh.sum() + ra.sum(), rh.sum()
        g[2:2 + nt] = np.bincount(hi, rh, nt) + np.bincount(ai, ra, nt) + 2 * ridge * att
        g[2 + nt:] = -np.bincount(ai, rh, nt) - np.bincount(hi, ra, nt) + 2 * ridge * dfn
        return nll, g
    return minimize(f, x0, jac=True, method="L-BFGS-B").x


def _tau(hg, ag, lh, la, rho):
    t = np.ones_like(lh)
    t = np.where((hg == 0) & (ag == 0), 1 - lh * la * rho, t)
    t = np.where((hg == 0) & (ag == 1), 1 + lh * rho, t)
    t = np.where((hg == 1) & (ag == 0), 1 + la * rho, t)
    return np.where((hg == 1) & (ag == 1), 1 - rho, t)


def _dc_probs(lh, la, rho, max_goals=10):
    g = np.arange(max_goals + 1)
    M = poisson.pmf(g[None, :], lh[:, None])[:, :, None] * poisson.pmf(g[None, :], la[:, None])[:, None, :]
    M[:, 0, 0] *= 1 - lh * la * rho
    M[:, 0, 1] *= 1 + lh * rho
    M[:, 1, 0] *= 1 + la * rho
    M[:, 1, 1] *= 1 - rho
    M /= M.sum(axis=(1, 2), keepdims=True)
    ph = np.tril(M, -1).sum(axis=(1, 2))  # filas = goles del local
    pd_ = np.trace(M, axis1=1, axis2=2)
    return np.column_stack([ph, pd_, 1 - ph - pd_])


def dc_walk(df, xi=0.002, ridge=1.0, min_train=200):
    """Dixon-Coles con decaimiento exp(-xi * días).
    ponytail: se reajusta una vez por mes calendario (no por jornada) y rho se estima
    en un segundo paso con las lambdas fijas; ajuste conjunto por jornada si hace falta."""
    teams = {t: j for j, t in enumerate(sorted(set(df.home) | set(df.away)))}
    nt = len(teams)
    hi, ai = df.home.map(teams).to_numpy(), df.away.map(teams).to_numpy()
    hg, ag = df.hg.to_numpy(), df.ag.to_numpy()
    day = df.date.dt.normalize().to_numpy()
    played = ~np.isnan(hg)
    P = np.tile(PRIOR, (len(df), 1))
    x = np.zeros(2 + 2 * nt)
    x[0], x[1] = 0.1, 0.25
    for _, idx in sorted(df.groupby([df.date.dt.year, df.date.dt.month]).indices.items()):
        t0 = day[idx[0]]
        tr = played & (day < t0)
        if tr.sum() < min_train:
            continue
        w = np.exp(-xi * ((t0 - day[tr]) / np.timedelta64(1, "D")))
        x = _fit_poisson(hi[tr], ai[tr], hg[tr], ag[tr], w, nt, x, ridge)
        att, dfn = x[2:2 + nt], x[2 + nt:]

        def lam(H, A):
            return np.exp(x[0] + x[1] + att[H] - dfn[A]), np.exp(x[0] + att[A] - dfn[H])

        lh, la = lam(hi[tr], ai[tr])
        rho = minimize_scalar(lambda r: -(w * np.log(_tau(hg[tr], ag[tr], lh, la, r))).sum(),
                              bounds=(-0.2, 0.2), method="bounded").x
        P[idx] = _dc_probs(*lam(hi[idx], ai[idx]), rho)
    return P


def features(df, eh, ea):
    """Variables previas al partido: dif. de Elo, dif. de forma (puntos por partido en
    los últimos 5) y dif. de días de descanso (tope 14). El intercepto es la localía."""
    hist, last = {}, {}
    X = np.empty((len(df), 3))
    for i, (date, h, a, hg, ag) in enumerate(zip(df.date, df.home, df.away, df.hg, df.ag)):
        form = [np.mean(hist[t][-5:]) if hist.get(t) else 1.37 for t in (h, a)]
        rest = [min((date - last[t]).days, 14) if t in last else 14 for t in (h, a)]
        X[i] = [(eh[i] - ea[i]) / 100, form[0] - form[1], (rest[0] - rest[1]) / 7]
        if hg == hg:
            hist.setdefault(h, []).append(3 if hg > ag else 1 if hg == ag else 0)
            hist.setdefault(a, []).append(3 if ag > hg else 1 if hg == ag else 0)
            last[h] = last[a] = date
    return X


def lr_walk(df, elo_P, eh, ea, C=1.0, burn=300, min_train=500):
    """Logística multinomial reentrenada al inicio de cada torneo con todo lo anterior.
    Antes de tener datos suficientes devuelve las probabilidades del Elo."""
    X, y = features(df, eh, ea), outcome_idx(df)
    played = df.hg.notna().to_numpy()
    P = elo_P.copy()
    for _, idx in df.groupby("torneo", sort=False).indices.items():
        tr = np.arange(burn, idx[0])
        tr = tr[played[tr]]
        if len(tr) < min_train:
            continue
        P[idx] = LogisticRegression(C=C, max_iter=1000).fit(X[tr], y[tr]).predict_proba(X[idx])
    return P
