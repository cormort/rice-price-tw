"""從 data/county_panel.csv 獨立重跑縣市面板迴歸，核對 index.html 內寫死的係數。

模型：ln(收購量 / 產量) ~ 輔導價差 + 縣市 FE + 期作 FE，標準誤依縣市群聚（CR1）。
執行：python3 scripts/verify_fe.py（需 pandas、numpy）；數字不符會 AssertionError。
"""
from pathlib import Path
import numpy as np
import pandas as pd

d = pd.read_csv(Path(__file__).parent.parent / 'data' / 'county_panel.csv')
d.columns = ['y', 'p', 'c', 'mk', 'gp', 'gap', 'pa', 'sur', 'prod', 'share']


def fe(df, qty):
    with np.errstate(divide='ignore'):  # 餘糧有 0 值，ln(0) 視為缺值排除
        df = df.assign(Y=np.log(df[qty] / df['prod']).replace(-np.inf, np.nan)).dropna(subset=['Y'])
    per = df.y.astype(str) + '-' + df.p.astype(str)
    X = pd.get_dummies(pd.DataFrame({'c': df.c, 'per': per}), drop_first=True).astype(float)
    X.insert(0, 'gap', df.gap)
    X.insert(0, 'const', 1.0)
    X, Y, g = X.values, df.Y.values, df.c.values
    n, k = X.shape
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ X.T @ Y
    e = Y - X @ b
    meat = sum(np.outer(s, s) for s in (X[g == c].T @ e[g == c] for c in np.unique(g)))
    G = len(np.unique(g))
    se = np.sqrt((XtXi @ meat @ XtXi)[1, 1] * G / (G - 1) * (n - 1) / (n - k))
    return n, G, round(b[1], 4), round(se, 4)


new, old = d[(d.y >= 99) & (d.y <= 113)], d[d.y <= 98]
# (名稱, 樣本, 應變數欄, 預期 n, 預期群數, 預期 β, 預期 SE) —— 預期值即 index.html 的 COUNTY_FE / ERA
CASES = [
    ('新制 計畫＋輔導', new, 'pa', 414, 15, 0.0486, 0.0295),
    ('新制 僅 1 期', new[new.p == 1], 'pa', 214, 15, 0.0149, 0.0262),
    ('新制 僅 2 期', new[new.p == 2], 'pa', 200, 14, 0.0465, 0.0609),
    ('新制 餘糧', new, 'sur', 383, 15, 0.1822, 0.2248),
    ('舊制 計畫＋輔導', old, 'pa', 301, 16, 0.5016, 0.0988),
]
for name, df, qty, n0, G0, b0, se0 in CASES:
    n, G, b, se = fe(df, qty)
    print(f'{name}: n={n} G={G} β={b} SE={se} t={b / se:.2f}')
    assert (n, G) == (n0, G0) and abs(b - b0) < 5e-4 and abs(se - se0) < 5e-4, name
print('全部與 index.html 相符')
