"""從 data/county_panel.csv 獨立重跑縣市面板迴歸，核對 index.html 內寫死的常數。

預期值直接讀 index.html 的 COUNTY_FE / LADDER / ERA / WR2 / WCB / ITS / ROBUST，頁面改了數字而沒重算就會報錯。
模型：ln(收購量[/產量]) ~ 輔導價差 + 固定效果，標準誤依縣市群聚（CR1）。
ITS：面板縣市合計的計畫＋輔導收購占比，對 100 年調價做中斷時間序列（Newey-West SE）與假政策年檢定。
執行：python3 scripts/verify_fe.py（需 pandas、numpy）。
"""
import itertools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
html = (ROOT / 'index.html').read_text(encoding='utf8')
page = {k: json.loads(re.search(rf'^const {k} = (.+?);\s*$', html, re.M).group(1))
        for k in ('COUNTY_FE', 'LADDER', 'ERA', 'WR2', 'WCB', 'ITS', 'ROBUST')}

d = pd.read_csv(ROOT / 'data' / 'county_panel.csv')
d.columns = ['y', 'p', 'c', 'mk', 'gp', 'gap', 'pa', 'sur', 'prod', 'share']
d['per'] = d.y.astype(str) + '-' + d.p.astype(str)
d['new'] = (d.y >= 99).astype(float)
d['gxn'] = d.gap * d.new                      # 交互作用：新制 × 價差
d['cxe'] = d.c + '_' + d.new.astype(str)      # 縣市 × 時期 FE（99 年縣市改制）
d['gr'] = d.gap / d.mk * 100                  # 價差率（%）：價差 ÷ 收穫期市價

new = d[(d.y >= 99) & (d.y <= 113)]
old = d[d.y <= 98]
s20 = new[~new.per.isin(page['COUNTY_FE']['added'])]   # 階梯 ①–④ 的舊 20 期作子集
both = d[d.y <= 113]


def ln_y(df, qty, share=True):
    with np.errstate(divide='ignore'):  # 餘糧有 0 值，ln(0) 視為缺值排除
        v = np.log(df[qty] / df['prod'] if share else df[qty])
    return df.assign(Y=v.replace(-np.inf, np.nan)).dropna(subset=['Y'])


def fe_matrix(df, fes):
    return pd.get_dummies(df[fes], drop_first=True).astype(float).values if fes else np.empty((len(df), 0))


def reg(df, fes, xs=('gap',)):
    """回傳 n、群數、各 x 的 (β, SE)。"""
    X = np.column_stack([np.ones(len(df)), df[list(xs)].values, fe_matrix(df, fes)])
    Y, g = df.Y.values, df.c.values
    n, k = X.shape
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ X.T @ Y
    e = Y - X @ b
    G = len(np.unique(g))
    meat = sum(np.outer(s, s) for s in (X[g == c].T @ e[g == c] for c in np.unique(g)))
    V = XtXi @ meat @ XtXi * G / (G - 1) * (n - 1) / (n - k)
    return n, G, [(b[i], np.sqrt(V[i, i])) for i in range(1, 1 + len(xs))]


def within_r2(df, fes):
    """y 與價差都先扣掉固定效果，殘差對殘差的 R²。"""
    D = np.column_stack([np.ones(len(df)), fe_matrix(df, fes)])
    resid = lambda v: v - D @ np.linalg.lstsq(D, v, rcond=None)[0]
    return np.corrcoef(resid(df.Y.values), resid(df.gap.values.astype(float)))[0, 1] ** 2


def wild_boot_p(df, fes, xs=('gap',), j=0):
    """限制型 wild cluster bootstrap（WCR-C）的雙尾 p 值，H0：第 j 個 x 的係數 = 0。

    群數只有 15～16，直接窮舉全部 2^G 組 Rademacher 權重（±1），結果精確、不含隨機性。
    """
    X = np.column_stack([np.ones(len(df)), df[list(xs)].values, fe_matrix(df, fes)])
    Y = df.Y.values
    g = pd.factorize(df.c)[0]
    G = g.max() + 1
    n, k = X.shape
    XtXi = np.linalg.pinv(X.T @ X)
    A, col = XtXi @ X.T, 1 + j
    q = X @ XtXi[:, col]                      # β_col 的 CR1 變異數 = adj · Σ_g (Σ_{i∈g} q_i e_i)²
    adj = G / (G - 1) * (n - 1) / (n - k)

    def t_of(Ys):                             # Ys：n × R，一次算 R 組
        E = Ys - X @ (A @ Ys)
        S = np.zeros((G, Ys.shape[1]))
        np.add.at(S, g, q[:, None] * E)
        return (A[col] @ Ys) / np.sqrt(adj * (S ** 2).sum(0))

    t0 = t_of(Y[:, None])[0]
    Xr = np.delete(X, col, axis=1)            # 在 H0 下估計，取擬合值與殘差
    fit = Xr @ np.linalg.lstsq(Xr, Y, rcond=None)[0]
    u = Y - fit
    signs = np.array(list(itertools.product([-1.0, 1.0], repeat=G)))
    hits = sum((np.abs(t_of(fit[:, None] + signs[s:s + 4096].T[g] * u[:, None])) >= abs(t0) - 1e-12).sum()
               for s in range(0, len(signs), 4096))
    return hits / len(signs)



def its(df, K, lag=3, ctrl=()):
    """中斷時間序列：占比 ~ 常數 + 2 期作 + (年−K) + 政策後 + 政策後×(年−K)。

    回傳 n、自由度、係數向量，以及「政策後」水準跳升與斜率變化的 Newey-West SE（Bartlett，lag 3）。
    """
    df = df.sort_values(['y', 'p'])
    t, post = (df.y - K).values, (df.y >= K).values.astype(float)
    X = np.column_stack([np.ones(len(df)), (df.p == 2).values, t, post, post * t] + [df[c].values for c in ctrl])
    Y = df.sh.values
    n, k = X.shape
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ X.T @ Y
    u = X * (Y - X @ b)[:, None]
    S = u.T @ u
    for l in range(1, lag + 1):
        G = u[l:].T @ u[:-l]
        S += (1 - l / (lag + 1)) * (G + G.T)
    V = XtXi @ S @ XtXi * n / (n - k)
    return n, n - k, b, np.sqrt(V[3, 3]), np.sqrt(V[4, 4])


# 面板縣市合計的期作序列（88、90–113 年；89 年從缺）。市價取各縣市收穫期市價的簡單平均。
agg = d[d.y <= 113].groupby(['y', 'p']).agg(pa=('pa', 'sum'), prod=('prod', 'sum'), mk=('mk', 'mean')).reset_index()
agg['sh'] = agg.pa / agg['prod'] * 100

fails = []


def check(name, got, want, tol=5e-4):
    ok = abs(got - want) < tol
    print(f"{'✓' if ok else '✗'} {name}: 重算 {got:.4f}／頁面 {want}")
    if not ok:
        fails.append(name)


def check_model(name, df, fes, m, G_want):
    n, G, [(b, se)] = reg(df, fes)
    check(f'{name} β', b, m['b'])
    check(f'{name} SE', se, m['se'])
    check(f'{name} n', n, m['n'], 0.5)
    check(f'{name} 群數', G, G_want, 0.5)


# 1. 縣市面板表（COUNTY_FE.specs 順序）
FE2 = ['c', 'per']
spec = [s['m'] for s in page['COUNTY_FE']['specs']]
nG = len(page['COUNTY_FE']['counties'])
check_model('新制 計畫＋輔導', ln_y(new, 'pa'), FE2, spec[0], nG)
check_model('新制 僅 1 期', ln_y(new[new.p == 1], 'pa'), FE2, spec[1], nG)
check_model('新制 僅 2 期', ln_y(new[new.p == 2], 'pa'), FE2, spec[2], nG - 1)   # 有一縣沒有 2 期作
check_model('舊 20 期作子集', ln_y(s20, 'pa'), FE2, spec[3], nG)
check_model('新制 餘糧', ln_y(new, 'sur'), FE2, spec[4], nG)
check_model('新制 餘糧 僅 1 期', ln_y(new[new.p == 1], 'sur'), FE2, spec[5], nG)
check_model('新制 餘糧 僅 2 期', ln_y(new[new.p == 2], 'sur'), FE2, spec[6], nG - 1)

# 2. 係數縮水階梯（LADDER 五步）＋ 組內 R²（WR2）
steps = [(ln_y(s20, 'pa', share=False), []), (ln_y(s20, 'pa', share=False), ['c']),
         (ln_y(s20, 'pa', share=False), FE2), (ln_y(s20, 'pa'), FE2), (ln_y(new, 'pa'), FE2)]
for i, ((df, fes), m, w) in enumerate(zip(steps, page['LADDER'], page['WR2']), 1):
    check_model(f'階梯 {i}', df, fes, m, nG)
    check(f'階梯 {i} 組內 R²', within_r2(df, fes), w)

# 3. 時代對比（ERA）：舊制、新制、合併、交互作用
eG = page['ERA']['counties']
eras = [e['m'] for e in page['ERA']['eras']]
check_model('舊制', ln_y(old, 'pa'), FE2, eras[0], eG)
check_model('新制', ln_y(new, 'pa'), FE2, eras[1], nG)
check_model('合併', ln_y(both, 'pa'), FE2, eras[2], eG)
n, G, [(b, se), (dx, dxse)] = reg(ln_y(both, 'pa'), ['cxe', 'per'], xs=('gap', 'gxn'))
it = page['ERA']['inter']
check('交互作用 舊制 β', b, it['b'])
check('交互作用 新制−舊制', dx, it['dx'])
check('交互作用 SE', dxse, it['dxse'])
_, _, [_, (dx2, dxse2)] = reg(ln_y(both, 'pa'), FE2, xs=('gap', 'gxn'))
check('共用縣市 FE 的交互作用 t（頁面文字）', dx2 / dxse2, -1.17, 5e-3)

# 3b. wild cluster bootstrap p 值（WCB）
wcb = page['WCB']
check('bootstrap p 舊制', wild_boot_p(ln_y(old, 'pa'), FE2), wcb['old'])
check('bootstrap p 新制', wild_boot_p(ln_y(new, 'pa'), FE2), wcb['new'])
check('bootstrap p 合併', wild_boot_p(ln_y(both, 'pa'), FE2), wcb['pooled'])
check('bootstrap p 舊 20 期作子集', wild_boot_p(ln_y(s20, 'pa'), FE2), wcb['s20'])
check('bootstrap p 交互作用', wild_boot_p(ln_y(both, 'pa'), ['cxe', 'per'], xs=('gap', 'gxn'), j=1), wcb['inter'])

# 4. 時代描述統計（ERA.desc）
for df, m in zip([old, new], [x['m'] for x in page['ERA']['desc']]):
    tag = f"{int(df.y.min())}–{int(df.y.max())}"
    check(f'{tag} 占比中位數', df.share.median(), m['shMed'], 0.05)
    check(f'{tag} 占比變異係數', df.share.std() / df.share.mean(), m['cv'], 5e-3)
    check(f'{tag} 價差中位數', df.gap.median(), m['gapMed'], 5e-3)
    check(f'{tag} 價差為負 %', (df.gap < 0).mean() * 100, m['negPct'], 0.05)

# 5. 100 年中斷時間序列＋假政策年（ITS）
I = page['ITS']
for m, ctrl in zip(I['main'], [(), ('mk',)]):
    n, df_, b, se, sse = its(agg, 100, ctrl=ctrl)
    check(f"ITS {m['l']} 跳升", b[3], m['b'], 5e-3)
    check(f"ITS {m['l']} SE", se, m['se'], 5e-3)
    check(f"ITS {m['l']} 斜率變化", b[4], m['s'], 5e-3)
    check(f"ITS {m['l']} n", n, m['n'], 0.5)
seg = {'pre': agg[agg.y <= 99], 'post': agg[agg.y >= 100]}
for m in I['placebo']:
    n, _, b, se, _ = its(seg[m['seg']], m['K'])
    check(f"假政策年 {m['K']} 跳升", b[3], m['b'], 5e-3)
    check(f"假政策年 {m['K']} SE", se, m['se'], 5e-3)
check('ITS 期作點數', len(agg), len(I['pts']), 0.5)

# 6. 留一年度＋價差率（ROBUST）
samples = {'new': new, 'old': old, 'both': both}
for r in page['ROBUST']:
    base = ln_y(samples[r['k']], 'pa')
    for m in r['loo']:
        x = base[base.y != m['y']]
        _, _, [(b, se)] = reg(x, FE2)
        check(f"留一 {r['k']} 剔除 {m['y']} β", b, m['b'])
        check(f"留一 {r['k']} 剔除 {m['y']} SE", se, m['se'])
        check(f"留一 {r['k']} 剔除 {m['y']} bootstrap p", wild_boot_p(x, FE2), m['p'], 5e-3)
    _, _, [(b, se)] = reg(base, FE2, xs=('gr',))
    check(f"價差率 {r['k']} β", b, r['ratio']['b'])
    check(f"價差率 {r['k']} SE", se, r['ratio']['se'])
    check(f"價差率 {r['k']} bootstrap p", wild_boot_p(base, FE2, xs=('gr',)), r['ratio']['p'], 5e-3)

print(f'\n{len(fails)} 項不符：{fails}' if fails else '\n全部與 index.html 相符')
raise SystemExit(1 if fails else 0)
