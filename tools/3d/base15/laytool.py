# 얼굴 배치 도구(laytool.html) 데이터 생성: faces15.build_flat을 기준 배치로 한 번 돌려 꼭짓점마다
# 화면(카메라 방향) 좌표·파츠 묶음(눈/볼/입)·표면에서 띄운 높이를 기록한다. 도구는 이 좌표를 새 배치로 옮긴 뒤
# 같은 카메라 방향으로 몸 메시에 다시 쏴서 붙인다(faces15의 proj_view와 같은 방식). 확정 값은 faces15.py LAY/FLAT['lay']에 옮겨 다시 굽는다.
#   python3 laytool.py → ../../../build/laytool.html
import os, sys, json, base64, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import faces15 as F
C = F.C; D0 = os.path.dirname(os.path.abspath(__file__))
orig = F.proj_view

def run(q, lay):
    rec = []
    def pv(P):
        loc, nor = orig(P); rec.append((np.asarray(P, float).copy(), loc, nor)); return loc, nor
    pv.miss = 0; F.proj_view = pv
    q = dict(q, lay=dict(q.get('lay', {}), **lay)); V, Cc, Fc = F.build_flat(q); F.proj_view = orig
    S = np.vstack([r[0] for r in rec]); L = np.vstack([r[1] for r in rec]); N = np.vstack([r[2] for r in rec])
    assert len(S) == len(V), (len(S), len(V))
    return S, ((np.asarray(V) - L) * N).sum(1), np.asarray(V), Cc, Fc

out = {'EYE_X': C.EYE_X, 'EYE_Z': C.EYE_Z, 'CAMP': F.CAMP, 'LAY': F.LAY, 'faces': {}}
A = F.proj(np.array([[0.0, C.EYE_Z]]))[0][0]; out['A'] = [float(A[0]), float(A[2]), float(-A[1])]   # 기준점(three 좌표)
pal = []
for k, q in F.FLAT.items():
    L0 = dict(F.LAY, **q.get('lay', {}))
    S, h, V, Cc, Fc = run(q, {})
    g = np.full(len(S), -1)
    for gi, (key, d) in enumerate((('KE', .1), ('KB', .1), ('MZ', .05))):
        S2 = run(q, {key: L0[key] + d})[0]; mv = np.abs(S2 - S).max(1) > 1e-7
        assert (g[mv] == -1).all(); g[mv] = gi
    assert (g >= 0).all(), (g < 0).sum()
    ci = []
    for c in Cc:
        if c not in pal: pal.append(c)
        ci.append(pal.index(c))
    out['faces'][k] = {'name': q['name'], 'L': L0, 'S': [round(float(a), 6) for a in S.ravel()], 'g': g.tolist(), 'h': [round(float(a), 6) for a in h],
                       'c': ci, 'f': [int(a) for a in np.asarray(Fc).ravel()], 'iris': q['iris'][0], 'eyl': q['eyl'], 'blush': q['blush'][:2],
                       'my': sum(y for _, y in q['mouthp']) / len(q['mouthp'])}
    print(k, q['name'], 'verts', len(S), '묶음', np.bincount(g).tolist())
out['pal'] = [F.lin(c) for c in pal]
hair = lambda n: (lambda h: {'color': h['color'], 'hair': h['hair'], 'cap': h['cap']})(json.load(open(os.path.join(D0, n), encoding='utf-8')))
out['hair'] = [hair('hair01_lite.json'), hair('hair02_lite.json')]
out['glb'] = base64.b64encode(open(os.path.join(D0, 'base15.glb'), 'rb').read()).decode()
dst = os.path.join(D0, '..', '..', '..', 'build', 'laytool.html')
html = open(os.path.join(D0, 'laytool_src.html'), encoding='utf-8').read().replace('/*DATA*/null', json.dumps(out, separators=(',', ':'), ensure_ascii=False))
os.makedirs(os.path.dirname(dst), exist_ok=True); open(dst, 'w', encoding='utf-8').write(html)
print(os.path.abspath(dst), os.path.getsize(dst))
