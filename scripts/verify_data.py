"""核對 data/*.csv 與 index.html 內嵌 DATA／CROP_VER 常數及下載連結。

僅使用 Python 標準函式庫，執行：python3 scripts/verify_data.py
"""
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
html = (ROOT / 'index.html').read_text(encoding='utf8')


def extract_const(name):
    m = re.search(rf'^const {name} = (.+?);\s*$', html, re.M)
    if not m:
        raise ValueError(f'Missing const {name}')
    return json.loads(m.group(1))


def read_csv(rel):
    with (ROOT / rel).open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


DATA = extract_const('DATA')
CROP_VER = extract_const('CROP_VER')
fails = []


def check(name, cond, detail=''):
    if not cond:
        fails.append(f'{name}: {detail}' if detail else name)
        print(f'✗ {name}' + (f' ({detail})' if detail else ''))
    else:
        print(f'✓ {name}')


def close(a, b, tol=1e-4):
    return (a is None and b is None) or (a is not None and b is not None and abs(a - b) <= tol)


def parse_opt(val):
    s = val.strip()
    return None if s == '' else float(s)


# 1. 下載連結與 data/ 檔案雙向核對
disk_csvs = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / 'data').glob('*.csv'))
linked = [f['href'] for f in DATA['files']]
check('下載連結皆存在於磁碟', all((ROOT / h).exists() for h in linked))
check('data/*.csv 皆已列入下載清單', sorted(linked) == disk_csvs, f'diff={set(disk_csvs) ^ set(linked)}')

# 2. monthly_prices.csv ↔ DATA.monthly
mp = read_csv('data/monthly_prices.csv')
check('monthly_prices 列數', len(mp) == len(DATA['monthly']), f'{len(mp)} vs {len(DATA["monthly"])}')
mp_ok = True
for r_csv, r_js in zip(mp, DATA['monthly']):
    if int(r_csv['民國年']) != r_js['y'] or int(r_csv['月份']) != r_js['m']:
        mp_ok = False
        break
    for i, c in enumerate(DATA['cols']):
        if not close(parse_opt(r_csv[c + '(元/公斤)']), r_js['v'][i], 1e-6):
            mp_ok = False
            break
check('monthly_prices.csv ↔ DATA.monthly', mp_ok)

# 3. gov_vs_market.csv ↔ DATA.gov
gm = read_csv('data/gov_vs_market.csv')
check('gov_vs_market 列數', len(gm) == len(DATA['gov']))
gm_ok = all(
    int(r_csv['民國年']) == r_js['y']
    and close(parse_opt(r_csv['市場稻穀價(元/公斤)']), r_js['mk'])
    and close(parse_opt(r_csv['計畫價(元/公斤)']), r_js['plan'])
    and close(parse_opt(r_csv['輔導價(元/公斤)']), r_js['guid'])
    and close(parse_opt(r_csv['餘糧價(元/公斤)']), r_js['surp'])
    and close(parse_opt(r_csv['稻作總產量(公噸)']), r_js['prod'], 1)
    and close(parse_opt(r_csv['公糧收購量(公噸)']), r_js['qty'], 1)
    and close(parse_opt(r_csv['收購占比%']), r_js['share'], 0.01)
    for r_csv, r_js in zip(gm, DATA['gov'])
)
check('gov_vs_market.csv ↔ DATA.gov', gm_ok)

# 4. tab33_official.csv ↔ DATA.tab33
t33 = read_csv('data/tab33_official.csv')
check('tab33_official 列數', len(t33) == len(DATA['tab33']))
t33_ok = all(
    int(r_csv['民國年']) == r_js['y']
    and close(parse_opt(r_csv['計畫收購(公噸)']), r_js['plan'], 1)
    and close(parse_opt(r_csv['輔導收購(公噸)']), r_js['asst'], 1)
    and close(parse_opt(r_csv['餘糧收購(公噸)']), r_js['surplus'], 1)
    and close(parse_opt(r_csv['災害穀(公噸)']), r_js['disaster'], 1)
    and close(parse_opt(r_csv['小計(公噸)']), r_js['sub'], 1)
    and close(parse_opt(r_csv['計畫+輔導(公噸)']), r_js['fullPlanAsst'], 1)
    and close(parse_opt(r_csv['收購支出(億元)']), r_js['cost'], 0.01)
    for r_csv, r_js in zip(t33, DATA['tab33'])
)
check('tab33_official.csv ↔ DATA.tab33', t33_ok)

# 5. purchase_official_99_114.csv ↔ DATA.periodOfficial
po = read_csv('data/purchase_official_99_114.csv')
check('purchase_official_99_114 列數', len(po) == len(DATA['periodOfficial']))
po_ok = all(
    int(r_csv['民國年']) == r_js['y']
    and all(
        close(parse_opt(r_csv[ck]), r_js[jk], 0.1)
        for ck, jk in [
            ('計畫全年(公噸)', 'plan'), ('輔導全年(公噸)', 'asst'), ('餘糧全年(公噸)', 'surplus'),
            ('年合計(公噸)', 'total'), ('一期計畫', 'p1_plan'), ('一期輔導', 'p1_asst'),
            ('一期餘糧', 'p1_surplus'), ('二期計畫', 'p2_plan'), ('二期輔導', 'p2_asst'), ('二期餘糧', 'p2_surplus'),
        ]
    )
    for r_csv, r_js in zip(po, DATA['periodOfficial'])
)
check('purchase_official_99_114.csv ↔ DATA.periodOfficial', po_ok)

# 6. quota_history.csv ↔ DATA.quotaHist
qh = read_csv('data/quota_history.csv')
check('quota_history 列數', len(qh) == len(DATA['quotaHist']))
qh_ok = all(
    close(parse_opt(r_csv['計畫收購(公斤/公頃)']), r_js['plan'], 0.1)
    and close(parse_opt(r_csv['輔導收購(公斤/公頃)']), r_js['asst'], 0.1)
    and close(parse_opt(r_csv['餘糧收購(公斤/公頃)']), r_js['surplus'], 0.1)
    and close(parse_opt(r_csv['輔導收購價(稉種,元/公斤)']), r_js['asstPrice'], 0.01)
    for r_csv, r_js in zip(qh, DATA['quotaHist'])
)
check('quota_history.csv ↔ DATA.quotaHist', qh_ok)

# 7. variety_vs_gov.csv ↔ DATA.variety
vv = read_csv('data/variety_vs_gov.csv')
check('variety_vs_gov 列數', len(vv) == len(DATA['variety']))
vv_ok = all(
    all(
        close(parse_opt(r_csv[f'{y}年市價(元/公斤)']), r_js['mk'][str(y)])
        and close(parse_opt(r_csv[f'{y}年價差']), r_js['gap'][str(y)])
        for y in DATA['varYears']
    )
    for r_csv, r_js in zip(vv, DATA['variety'])
)
check('variety_vs_gov.csv ↔ DATA.variety', vv_ok)

# 8. crop_mix_113p1.csv ↔ DATA.cropMix
cm = read_csv('data/crop_mix_113p1.csv')
check('crop_mix_113p1 列數', len(cm) == len(DATA['cropMix']))
cm_ok = all(
    close(parse_opt(r_csv['113年1期收購量(公噸)']), r_js['qty'], 0.1)
    and close(parse_opt(r_csv['占比%']), r_js['share'], 0.01)
    for r_csv, r_js in zip(cm, DATA['cropMix'])
)
check('crop_mix_113p1.csv ↔ DATA.cropMix', cm_ok)

# 9. crop_mix_verified.csv ↔ CROP_VER
cv = read_csv('data/crop_mix_verified.csv')
cv_rows = [r for r in cv if r['民國年'].strip() != '合計']
check('crop_mix_verified 期作列數', len(cv_rows) == len(CROP_VER['periods']))
cv_ok = all(
    int(r_csv['民國年']) == r_js['y']
    and int(r_csv['期作']) == r_js['p']
    and close(parse_opt(r_csv['期作合計(公噸)']), r_js['t'], 0.1)
    for r_csv, r_js in zip(cv_rows, CROP_VER['periods'])
)
check('crop_mix_verified.csv ↔ CROP_VER', cv_ok)

print(f'\n{len(fails)} 項不符' if fails else '\n全部 CSV 與 index.html 資料及下載連結相符')
raise SystemExit(1 if fails else 0)
