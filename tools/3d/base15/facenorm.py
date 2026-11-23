# 얼굴 피부 다듬기: base15_raw.glb(render15.py 결과) → base15.glb
# 1) 얼굴 앞쪽 삼각형을 잘게 나눈다(새 꼭짓점은 거리장 표면에 붙이고, 뼈 가중치·색은 양 끝의 평균).
#    줄인 메시(2000면)는 삼각형이 커서 법선 보간 경계가 관자놀이·뺨에 밝은 줄로 보였다.
# 2) 법선: 얼굴 굴곡(눈자리·눈썹뼈·콧대·미간·볼·입 주변·코)의 세기를 FK배로 줄인 거리장의 기울기.
#    머리(목 위, 귀 안쪽)는 거리장 법선을 그대로, 그 밖은 원래 메시 법선에 굴곡 차이만큼만 더한다(뒤통수·몸은 그대로).
#   python3 facenorm.py [FK=0.55] [단계=1]   → 그 뒤 faces15.py를 다시 돌려 얼굴 조각 법선도 맞춘다
import os, sys, json, struct
import numpy as np
D0 = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('PARAMS', os.path.join(D0, 'params.json')); sys.path.insert(0, D0)
import chibi15 as C
k = float(sys.argv[1]) if len(sys.argv) > 1 else 0.55; LV = int(sys.argv[2]) if len(sys.argv) > 2 else 1
b = open(os.path.join(D0, 'base15_raw.glb'), 'rb').read()
n = struct.unpack('<I', b[12:16])[0]; j = json.loads(b[20:20 + n]); B0 = 20 + n + 8
pr = j['meshes'][0]['primitives'][0]; A = pr['attributes']
DT = {5126: np.float32, 5121: np.uint8, 5123: np.uint16, 5125: np.uint32}; NC = {'SCALAR': 1, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}
def get(i):
    a = j['accessors'][i]; v = j['bufferViews'][a['bufferView']]; o = B0 + v.get('byteOffset', 0) + a.get('byteOffset', 0); d = DT[a['componentType']]
    return np.frombuffer(b[o:o + a['count'] * NC[a['type']] * np.dtype(d).itemsize], d).reshape(a['count'], -1).copy()
P = get(A['POSITION']).astype(float); N0 = get(A['NORMAL']).astype(float); CO = get(A['COLOR_0']).astype(float)
JN = get(A['JOINTS_0']).astype(int); WT = get(A['WEIGHTS_0']).astype(float); I = get(pr['indices']).reshape(-1, 3).astype(int)

def surf(p, it=4):
    x, y, z = p[:, 0], -p[:, 2], p[:, 1]; e = 1e-3
    for _ in range(it):
        g = np.stack([C.sdf(x + e, y, z) - C.sdf(x - e, y, z), C.sdf(x, y + e, z) - C.sdf(x, y - e, z), C.sdf(x, y, z + e) - C.sdf(x, y, z - e)], 1) / (2 * e)
        f = C.sdf(x, y, z) / np.maximum((g * g).sum(1), 1e-9); x, y, z = x - f * g[:, 0], y - f * g[:, 1], z - f * g[:, 2]
    return np.stack([x, z, -y], 1)
def inreg(p): return (p[:, 1] > 0.60) & (p[:, 1] < 1.02) & (p[:, 2] > 0.12) & (np.abs(p[:, 0]) < 0.40)   # three: y 위, z 앞

for _ in range(LV):
    cen = P[I].mean(1); R = inreg(cen); E = {}
    def mid(a, c):
        key = (min(a, c), max(a, c))
        if key not in E: E[key] = -1
        return key
    for t in np.nonzero(R)[0]:
        for a, c in ((0, 1), (1, 2), (2, 0)): mid(I[t, a], I[t, c])
    keys = list(E); base = len(P)
    for q_, key in enumerate(keys): E[key] = base + q_
    a_, c_ = np.array([k_[0] for k_ in keys]), np.array([k_[1] for k_ in keys])
    P = np.vstack([P, surf((P[a_] + P[c_]) / 2)]); N0 = np.vstack([N0, N0[a_] + N0[c_]]); CO = np.vstack([CO, (CO[a_] + CO[c_]) / 2])
    nj, nw = [], []
    for a, c in zip(a_, c_):                                     # 뼈 가중치: 두 끝을 합쳐 큰 4개
        d = {}
        for J_, W_ in ((JN[a], WT[a]), (JN[c], WT[c])):
            for jj, ww in zip(J_, W_): d[jj] = d.get(jj, 0) + ww / 2
        it_ = sorted(d.items(), key=lambda x: -x[1])[:4]; s_ = sum(w for _, w in it_)
        it_ += [(0, 0.0)] * (4 - len(it_)); nj.append([x[0] for x in it_]); nw.append([x[1] / s_ for x in it_])
    JN = np.vstack([JN, nj]); WT = np.vstack([WT, nw])
    NI = []
    for t in I:                                                   # 나뉜 변 수에 따라 1→2·3·4 (이웃과 틈 없이)
        v = list(t); m = [E.get((min(v[i], v[(i + 1) % 3]), max(v[i], v[(i + 1) % 3]))) for i in range(3)]; s_ = sum(x is not None for x in m)
        if s_ == 0: NI.append(v)
        elif s_ == 3: NI += [[v[0], m[0], m[2]], [m[0], v[1], m[1]], [m[2], m[1], v[2]], [m[0], m[1], m[2]]]
        elif s_ == 1:
            i = [x is not None for x in m].index(True); a, c, o = v[i], v[(i + 1) % 3], v[(i + 2) % 3]; NI += [[a, m[i], o], [m[i], c, o]]
        else:
            i = [x is None for x in m].index(True); a, c, o = v[i], v[(i + 1) % 3], v[(i + 2) % 3]   # a-c 변만 안 나뉨
            mc, mo = m[(i + 1) % 3], m[(i + 2) % 3]                                                     # c-o, o-a 중점
            NI += [[a, c, mc], [a, mc, mo], [mo, mc, o]]
    I = np.array(NI)
N0 /= np.maximum(np.linalg.norm(N0, axis=1, keepdims=True), 1e-9)

x, y, z = P[:, 0], -P[:, 2], P[:, 1]
def grad(fk):
    C.FK = fk; e = 1e-3
    g = np.stack([C.sdf(x + e, y, z) - C.sdf(x - e, y, z), C.sdf(x, y + e, z) - C.sdf(x, y - e, z), C.sdf(x, y, z + e) - C.sdf(x, y, z - e)], 1)
    g /= np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9); return np.stack([g[:, 0], g[:, 2], -g[:, 1]], 1)
gk = grad(k); d = gk - grad(1.0); C.FK = 1.0
sst = lambda a, b_, v: np.clip((v - a) / (b_ - a), 0, 1) ** 2 * (3 - 2 * np.clip((v - a) / (b_ - a), 0, 1))
w = sst(0.58, 0.66, P[:, 1]) * sst(0.41, 0.37, np.abs(P[:, 0]))
N = (N0 + d) * (1 - w)[:, None] + gk * w[:, None]; N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)

# glb 다시 쓰기(같은 배치, 길이만 바뀜)
cnt = len(P); bufs = [P.astype(np.float32), N.astype(np.float32), CO.astype(np.float32), JN.astype(np.uint8), WT.astype(np.float32),
                     I.astype(np.uint16 if cnt < 65536 else np.uint32).ravel()]
order = [A['POSITION'], A['NORMAL'], A['COLOR_0'], A['JOINTS_0'], A['WEIGHTS_0'], pr['indices']]
ibm = get(j['skins'][0]['inverseBindMatrices']).astype(np.float32)
blob = b''; views = []
for ai, arr in zip(order + [j['skins'][0]['inverseBindMatrices']], bufs + [ibm]):
    by = arr.tobytes(); views.append((ai, len(blob), len(by))); blob += by + b'\0' * ((-len(by)) % 4)
for ai, off, ln in views:
    a = j['accessors'][ai]; v = j['bufferViews'][a['bufferView']]; v['byteOffset'] = off; v['byteLength'] = ln; a.pop('byteOffset', None)
    if ai != j['skins'][0]['inverseBindMatrices']: a['count'] = len(I) * 3 if ai == pr['indices'] else cnt
    if ai == pr['indices']: a['componentType'] = 5123 if cnt < 65536 else 5125
    if ai == A['POSITION']: a['min'] = P.min(0).tolist(); a['max'] = P.max(0).tolist()
j['buffers'][0]['byteLength'] = len(blob)
js = json.dumps(j, separators=(',', ':')).encode(); js += b' ' * ((-len(js)) % 4)
out = struct.pack('<III', 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(blob)) + struct.pack('<II', len(js), 0x4E4F534A) + js + struct.pack('<II', len(blob), 0x004E4942) + blob
open(os.path.join(D0, 'base15.glb'), 'wb').write(out)
print('FK', k, '단계', LV, '꼭짓점', cnt, '면', len(I), 'glb', len(out))
