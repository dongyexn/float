# 주민 얼굴 20종(A~T) → faces15.json
# 원안: 점토 SD 베이스에서 만든 얼굴 시안(동글눈·점눈·웃는눈·빙글빙글·졸린눈·코주부·새침·윙크·놀람·화남·슬픔·하트눈·별눈·부끄·메롱·당황·무표정·감동·뿌듯·뽀뽀).
# 그 정의(옛 머리 좌표: 눈 높이 2.32, 눈 간격 0.255, 입 2.10, 파츠 배율 1.5)를 그대로 옮겨 base15 머리 표면(chibi15 거리장)에 투영한다.
# 파츠마다 중심을 base15의 눈·입 자리에 맞춰 옮기고 크기는 눈 간격 비율로 줄인다. 게임은 이 조각들을 head 뼈 밑에 붙인다(꼭짓점 색).
#   PARAMS=params.json python3 faces15.py   → faces15.json
import os, sys, json, math
import numpy as np
D0 = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('PARAMS', os.path.join(D0, 'params.json'))
sys.path.insert(0, D0)
import chibi15 as C

EY, EX, MY, FSZ = 2.32, 0.255, 2.10, 1.5          # 원안 좌표
K = C.EYE_X / EX                                   # 크기 비율
LIFT = 0.0035                                      # 감량한 몸 메시와 겹치지 않게 살짝 띄움
NAMES = {'A': '동글눈', 'B': '점눈', 'C': '웃는눈', 'D': '빙글빙글', 'E': '졸린눈', 'F': '코주부', 'G': '새침', 'H': '윙크',
         'I': '놀람', 'J': '화남', 'K': '슬픔', 'L': '하트눈', 'M': '별눈', 'N': '부끄', 'O': '메롱', 'P': '당황',
         'Q': '무표정', 'R': '감동', 'S': '뿌듯', 'T': '뽀뽀'}
HEX = {'ink': '#2b211e', 'white': '#ffffff', 'brown': '#6b3f2a', 'cheek': '#f29a8e', 'mouth': '#8a3b33', 'tongue': '#e8707a',
       'tear': '#8fd3f2', 'heart': '#e8475f', 'skin': '#e4b491'}

def lin(h):
    c = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    return [round(x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4, 4) for x in c]

def mapc(u, v):
    """원안 좌표의 한 점 → base15 정면 좌표(x, z). 눈 주변은 눈 높이에, 입 주변은 입 높이에 맞추고 그 사이는 잇는다"""
    a, b = EY - 0.06, MY + 0.08
    za, zb = C.EYE_Z + (a - EY) * K, C.MOUTH_Z + (b - MY) * K
    if v >= a: z = C.EYE_Z + (v - EY) * K
    elif v <= b: z = C.MOUTH_Z + (v - MY) * K
    else: z = zb + (za - zb) * (v - b) / (a - b)
    return np.array([u * K, z])

def conv(pts, k=None):
    P = np.array([(p[0], p[1]) for p in pts], float); c = P.mean(0); k = k or K
    if abs(c[0]) > EX + 0.1 and EY - 0.2 < c[1] < EY - 0.05:      # 볼 자리(볼터치·빗금): 원안 그대로 옮기면 base15의 좁은 볼을 벗어나 옆으로 삐져나오므로 base15의 볼터치 자리에 맞춘다
        sg = 1 if c[0] > 0 else -1
        t = np.array([sg * (C.EYE_X + C.Q['blushDX'] + (abs(c[0]) - (EX + 0.17)) * K * 0.6), C.EYE_Z + C.Q['blushDZ'] + (c[1] - (EY - 0.13)) * K])
    else: t = mapc(*c)
    return t + (P - c) * k

def proj(P):
    """정면(-y)에서 쏜 선이 머리 표면과 만나는 점과 바깥 법선"""
    P = np.asarray(P, float); x, z = P[:, 0], P[:, 1]
    ys = np.linspace(-0.75, 0.0, 300)
    d = C.sdf(np.repeat(x[:, None], 300, 1), np.repeat(ys[None, :], len(x), 0), np.repeat(z[:, None], 300, 1))
    i = np.clip(np.argmax(d < 0, 1), 1, 299); r = np.arange(len(x))
    d0, d1 = d[r, i - 1], d[r, i]; y = ys[i - 1] + (ys[i] - ys[i - 1]) * d0 / np.maximum(d0 - d1, 1e-9)
    for _ in range(4):                                             # 표면으로 조금 더 붙인다
        f = C.sdf(x, y, z); e = 1e-3; g = (C.sdf(x, y + e, z) - C.sdf(x, y - e, z)) / (2 * e)
        y = y - f / np.where(np.abs(g) > 0.2, g, -1.0)
    e = 2e-3
    n = np.stack([C.sdf(x + e, y, z) - C.sdf(x - e, y, z), C.sdf(x, y + e, z) - C.sdf(x, y - e, z), C.sdf(x, y, z + e) - C.sdf(x, y, z - e)], 1)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    return np.stack([x, y, z], 1), n

PROJ = proj          # 표면에 붙이는 방법(기본: 정면에서). 새 얼굴은 proj_view로 바꿔 쓴다

class Mesh:
    def __init__(self): self.v, self.c, self.f = [], [], []
    def add(self, V, col, F):
        o = len(self.v); self.v += [list(p) for p in V]; self.c += ([col] * len(V) if isinstance(col, str) else list(col)); self.f += [[a + o, b + o, c + o] for a, b, c in F]
        return o

def dome(M, out2, h, base, col, rings=3):
    """외곽선 안을 채운 점토 방울: 가장자리는 base, 가운데로 h만큼 솟음. 돌려주는 값 = (꼭짓점 시작 번호, 꼭짓점의 정면 좌표, 틀)"""
    out2 = np.asarray(out2, float); c = out2.mean(0); n = len(out2)
    hx = max(np.abs(out2[:, 0] - c[0]).max(), 1e-6); hy = max(np.abs(out2[:, 1] - c[1]).max(), 1e-6)
    P2, H = [c], [base + h]
    for r in range(1, rings + 1):
        rho = r / rings
        for p in out2: P2.append(c + (p - c) * rho); H.append(base + h * math.sqrt(max(0.0, 1 - rho * rho)))
    P2 = np.array(P2); loc, nor = PROJ(P2); V = loc + nor * (np.array(H)[:, None] + LIFT)
    F = [[0, 1 + k, 1 + (k + 1) % n] for k in range(n)]
    for r in range(1, rings):
        a, b = 1 + (r - 1) * n, 1 + r * n
        for k in range(n): F += [[a + k, b + k, b + (k + 1) % n], [a + k, b + (k + 1) % n, a + (k + 1) % n]]
    o = M.add(V, col, F)
    return o, P2, (c, hx, hy)

def stroke(M, path2, w, h, col, seg=4, sub=2):
    """점토를 굴린 줄: 경로를 따라 반원 단면, 양 끝은 가늘어짐"""
    P = np.asarray(path2, float); dense = []
    for i in range(len(P) - 1):
        for k in range(sub): dense.append(P[i] + (P[i + 1] - P[i]) * k / sub)
    dense.append(P[-1]); dense = np.array(dense); N = len(dense); P2, H = [], []
    for i, p in enumerate(dense):
        t = dense[min(i + 1, N - 1)] - dense[max(i - 1, 0)]; t = t / max(np.linalg.norm(t), 1e-9); nn = np.array([-t[1], t[0]])
        e = min(i, N - 1 - i) / (1.5 * sub / 2 + 0.5); tp = math.sqrt(min(1.0, e)) if e < 1 else 1.0
        for k in range(seg + 1):
            a = math.pi * k / seg; P2.append(p + nn * (math.cos(a) * w * max(tp, 0.25))); H.append(math.sin(a) * h * max(tp, 0.2) - 0.003 * K)
    loc, nor = PROJ(np.array(P2)); V = loc + nor * (np.array(H)[:, None] + LIFT); F = []
    for i in range(N - 1):
        a, b = i * (seg + 1), (i + 1) * (seg + 1)
        for k in range(seg): F += [[a + k, b + k, b + k + 1], [a + k, b + k + 1, a + k + 1]]
    M.add(V, col, F)

def ellipse(cu, cv, ru, rv, rot=0.0, n=14, lo=0.0, hi=2 * math.pi):
    ru, rv = ru * FSZ, rv * FSZ; full = hi - lo >= 2 * math.pi - 1e-6; pts = []
    for k in range(n):
        a = lo + (hi - lo) * k / (n - (0 if full else 1)); x, y = math.cos(a) * ru, math.sin(a) * rv
        pts.append((cu + x * math.cos(rot) - y * math.sin(rot), cv + x * math.sin(rot) + y * math.cos(rot)))
    return pts
def arc(cu, cv, ru, rv, lo, hi, n=9): return [(cu + math.cos(lo + (hi - lo) * k / (n - 1)) * ru, cv + math.sin(lo + (hi - lo) * k / (n - 1)) * rv) for k in range(n)]
def heart(cu, cv, s, n=18):
    s *= FSZ
    return [(cu + s * 16 * math.sin(t) ** 3 / 17, cv + s * (13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)) / 17 + s * 0.15) for t in [2 * math.pi * k / n for k in range(n)]][::-1]
def star(cu, cv, r, n=5):
    r *= FSZ
    return [(cu + math.cos(math.pi / 2 + k * math.pi / n) * (r if k % 2 == 0 else r * 0.5), cv + math.sin(math.pi / 2 + k * math.pi / n) * (r if k % 2 == 0 else r * 0.5)) for k in range(2 * n)]

HL = (0.33, 0.70, 0.19, 0.19, '#ffffff'); HL2 = (0.66, 0.32, 0.09, 0.09, '#ffffff')

def face_parts(st):
    """원안 그대로: [(재질, 종류, 인자)]"""
    P = []
    D = lambda m, o, h, base=-0.004: P.append((m, 'dome', (o, h * FSZ, base * FSZ)))
    E = lambda o, layers=(HL,), base='#2b211e', h=0.018: P.append(('eye', 'eye', (o, h * FSZ, base, tuple(layers))))
    L = lambda m, path, w, h=None, scale=True: P.append((m, 'stroke', (path, w, h, scale)))
    cheeks = lambda r=0.075, y=EY - 0.13, x=EX + 0.17, h=0.006: [D('cheek', ellipse(s * x, y, r * 1.25, r * 0.8), h) for s in (1, -1)]
    smile = lambda ru=0.055, rv=0.035, y=MY + 0.035: L('mouth', arc(0, y, ru, rv, math.radians(200), math.radians(340)), 0.009)
    frown = lambda ru=0.045, rv=0.025, y=MY - 0.01: L('mouth', arc(0, y, ru, rv, math.radians(25), math.radians(155)), 0.009)
    def openm(ru=0.07, rv=0.065, y=MY + 0.03, tongue=True):
        D('mouth', ellipse(0, y, ru, rv, 0, 12, math.pi, 2 * math.pi), 0.008)
        if tongue: D('tongue', ellipse(0, y - rv * 0.62, ru * 0.5, rv * 0.32, n=10), 0.011)
    closed_up = lambda s, w=0.06: L('ink', arc(s * EX, EY - 0.025, w, 0.05, math.radians(15), math.radians(165)), 0.013)
    closed_dn = lambda s, w=0.06: L('ink', arc(s * EX, EY + 0.02, w, 0.04, math.radians(195), math.radians(345)), 0.012)
    brow = lambda s, tilt=0.0, y=EY + 0.11, w=0.05: L('ink', [(s * (EX - w), y + tilt), (s * EX, y + 0.008), (s * (EX + w), y - tilt)], 0.011)
    wink = lambda sx: L('ink', [(sx * (EX + 0.05), EY + 0.045), (sx * (EX - 0.04), EY), (sx * (EX + 0.05), EY - 0.045)], 0.012)
    if st == 'A':
        for s in (1, -1): E(ellipse(s * EX, EY, 0.052, 0.072))
        smile(); cheeks()
    elif st == 'B':
        for s in (1, -1): E(ellipse(s * EX, EY, 0.033, 0.036, n=10), ()); brow(s)
        L('mouth', arc(-0.028, MY + 0.03, 0.028, 0.03, math.radians(190), math.radians(350), 7) + arc(0.028, MY + 0.03, 0.028, 0.03, math.radians(190), math.radians(350), 7)[1:], 0.008)
        cheeks(0.065)
    elif st == 'C':
        for s in (1, -1): closed_up(s)
        openm(); cheeks(0.085)
    elif st == 'D':
        for s in (1, -1):
            L('ink', [(s * EX + math.cos(s * t + 1.0) * (0.006 + 0.0095 * t), EY + math.sin(s * t + 1.0) * (0.006 + 0.0095 * t)) for t in [k * 0.35 for k in range(26)]], 0.009, None, False)
        L('mouth', [(-0.045, MY + 0.02), (-0.022, MY + 0.033), (0.0, MY + 0.02), (0.022, MY + 0.033), (0.045, MY + 0.02)], 0.008)
        D('tear', ellipse(-EX - 0.2, EY + 0.13, 0.022, 0.034, n=10), 0.018)
        cheeks(0.06)
    elif st == 'E':
        for s in (1, -1): closed_dn(s)
        D('mouth', ellipse(0, MY + 0.01, 0.022, 0.026, n=10), 0.01); cheeks(0.06)
    elif st == 'F':
        for s in (1, -1): E(ellipse(s * EX * 0.92, EY + 0.02, 0.03, 0.04, n=10), ())
        D('skin', ellipse(0, (EY + MY) / 2 + 0.005, 0.06, 0.05), 0.05, -0.01)
        L('mouth', arc(0, MY + 0.02, 0.085, 0.05, math.radians(200), math.radians(340)), 0.009)
        cheeks(0.08)
    elif st == 'G':
        for s in (1, -1):
            E(ellipse(s * EX, EY + 0.005, 0.058, 0.055, 0, 11, math.pi, 2 * math.pi), [(0.35, 0.55, 0.2, 0.26, '#ffffff')], h=0.016)
            L('ink', [(s * (EX - 0.07), EY + 0.01), (s * EX, EY + 0.012), (s * (EX + 0.07), EY + 0.01), (s * (EX + 0.09), EY + 0.03)], 0.012)
        D('mouth', ellipse(0, MY + 0.015, 0.02, 0.016, n=10), 0.012); cheeks(0.07)
    elif st == 'H':
        E(ellipse(-EX, EY, 0.05, 0.068)); wink(1)
        openm(0.06, 0.055); cheeks(0.075)
    elif st == 'I':
        for s in (1, -1): E(ellipse(s * EX, EY, 0.062, 0.07), [HL, HL2]); brow(s, 0.0, EY + 0.15, 0.045)
        D('mouth', ellipse(0, MY - 0.005, 0.03, 0.045, n=12), 0.01)
    elif st == 'J':
        for s in (1, -1):
            E(ellipse(s * EX, EY - 0.005, 0.05, 0.05, 0, 11, math.pi, 2 * math.pi), [(0.36, 0.55, 0.2, 0.26, '#ffffff')], h=0.016)
            L('ink', [(s * (EX - 0.065), EY + 0.05), (s * (EX + 0.06), EY + 0.1)], 0.013)
        frown(0.05, 0.03)
    elif st == 'K':
        for s in (1, -1):
            E(ellipse(s * EX, EY, 0.05, 0.062), [HL, HL2, (0.5, 0.12, 0.38, 0.1, '#5b4a46')])
            L('ink', [(s * (EX - 0.06), EY + 0.11), (s * (EX + 0.055), EY + 0.08)], 0.01)
        D('tear', ellipse(EX + 0.035, EY - 0.1, 0.018, 0.028, n=10), 0.014)
        frown(0.04, 0.022)
    elif st == 'L':
        for s in (1, -1): P.append(('eye', 'eye', (heart(s * EX, EY, 0.075), 0.02 * FSZ, '#e8475f', ((0.33, 0.72, 0.16, 0.14, '#ffffff'),))))
        openm(0.065, 0.06); cheeks(0.08)
    elif st == 'M':
        for s in (1, -1): P.append(('eye', 'eye', (star(s * EX, EY, 0.105), 0.022 * FSZ, '#f2bf3a', ((0.4, 0.62, 0.14, 0.12, '#fff6d8'),))))
        openm(0.07, 0.065)
    elif st == 'N':
        wink(1); wink(-1)
        L('mouth', [(-0.05, MY + 0.02), (-0.025, MY + 0.035), (0.0, MY + 0.02), (0.025, MY + 0.035), (0.05, MY + 0.02)], 0.008)
        cheeks(0.09)
        for s in (1, -1):
            for k in (-1, 0, 1): L('mouth', [(s * (EX + 0.17) + k * 0.04 - 0.01, EY - 0.11), (s * (EX + 0.17) + k * 0.04 + 0.01, EY - 0.15)], 0.005)
    elif st == 'O':
        E(ellipse(-EX, EY, 0.05, 0.068)); closed_up(1); smile(0.06, 0.03)
        D('tongue', ellipse(0.02, MY - 0.005, 0.03, 0.04, 0, 11, math.pi, 2 * math.pi), 0.014); cheeks()
    elif st == 'P':
        for s in (1, -1): E(ellipse(s * EX, EY, 0.035, 0.04, n=10), ((0.35, 0.68, 0.22, 0.22, '#ffffff'),))
        D('tear', ellipse(EX + 0.2, EY + 0.13, 0.025, 0.038, n=10), 0.02)
        L('mouth', [(-0.05, MY + 0.025), (-0.025, MY + 0.01), (0.0, MY + 0.025), (0.025, MY + 0.01), (0.05, MY + 0.025)], 0.008)
    elif st == 'Q':
        for s in (1, -1): L('ink', [(s * (EX - 0.045), EY), (s * (EX + 0.045), EY)], 0.012)
        L('mouth', [(-0.035, MY + 0.02), (0.035, MY + 0.02)], 0.008)
    elif st == 'R':
        for s in (1, -1):
            E(ellipse(s * EX, EY, 0.065, 0.08, n=16), [(0.5, 0.45, 0.42, 0.42, '#4a2e22'), HL, HL2, (0.7, 0.66, 0.08, 0.08, '#ffffff'), (0.5, 0.1, 0.42, 0.12, '#9bd2ea')])
            D('tear', ellipse(s * (EX + 0.06), EY - 0.075, 0.016, 0.022, n=8), 0.012)
        L('mouth', [(-0.04, MY + 0.02), (-0.02, MY + 0.03), (0.0, MY + 0.02), (0.02, MY + 0.03), (0.04, MY + 0.02)], 0.008)
        cheeks(0.07)
    elif st == 'S':
        for s in (1, -1): closed_dn(s, 0.055)
        openm(0.08, 0.07); cheeks(0.085)
    elif st == 'T':
        for s in (1, -1): closed_dn(s, 0.05)
        L('mouth', arc(0.0, MY + 0.04, 0.02, 0.018, math.radians(-90), math.radians(90), 7) + arc(0.0, MY + 0.005, 0.02, 0.018, math.radians(-90), math.radians(90), 7), 0.009)
        D('heart', heart(EX + 0.2, EY + 0.17, 0.045, 14), 0.016); cheeks(0.08)
    BR = {'A': (0, 0, .012), 'C': (.005, .005, .022), 'D': (.03, -.01, .01), 'E': (-.01, -.012, .004), 'F': (0, 0, .012),
          'G': (-.005, .02, .014), 'H': (0, 0, .012), 'L': (.005, .005, .02), 'M': (.01, .01, .022), 'N': (.025, -.012, .008),
          'O': (0, .005, .014), 'P': (.03, -.012, .008), 'Q': (0, 0, 0), 'R': (.032, -.012, .01), 'S': (.006, .006, .022), 'T': (.004, .004, .018)}
    if st in BR:
        bi, bo, bm_ = BR[st]
        for s in (1, -1):
            L('ink', [(s * (EX - 0.06), EY + 0.13 + bi), (s * (EX - 0.02), EY + 0.13 + bm_ * 0.55 + bi * 0.66 + bo * 0.33), (s * (EX + 0.02), EY + 0.13 + bm_ * 0.55 + bi * 0.33 + bo * 0.66), (s * (EX + 0.06), EY + 0.13 + bo)], 0.0085)
    return P

def build(st):
    M = Mesh()
    for m, kind, a in face_parts(st):
        if kind == 'dome':
            dome(M, conv(a[0], K * 0.62 if m == 'cheek' else None), a[1] * K, (0.0015 if m == 'cheek' else a[2] * K), HEX[m], rings=3 if m in ('skin', 'cheek') else 2)   # 볼터치는 얇아서 감량한 몸 메시에 파묻히지 않게 가장자리까지 띄운다
        elif kind == 'stroke':
            path, w, h, scale = a; P = np.array(path, float); c = P.mean(0)
            if scale: P = c + (P - c) * FSZ
            w = w * FSZ * K; stroke(M, conv([c]) [0] + (P - c) * K, w, (h * FSZ * K if h is not None else w * 0.8), HEX[m])
        else:                                                      # 눈: 바탕 방울 + 위에 얹는 반짝임. 눈 밖으로 걸치는 층(눈물 고임 등)은 바탕에 칠한다
            out, h, base, layers = a; h *= K; o2 = conv(out)
            part = [l for l in layers if math.hypot((l[0] - .5) * 2, (l[1] - .5) * 2) + 2 * max(l[2], l[3]) > 1.05]
            o, p2, (c, hx, hy) = dome(M, o2, h, -0.004 * FSZ * K, base, rings=4 if part else 3)
            done = [(o, p2)]; up = 0; sc = np.array([hx, hy])
            for l in layers:
                cu, cv, ru, rv, hexc = l
                if l in part:
                    for o_, uv_ in done:
                        for i, (pu, pv) in enumerate((uv_ - c) / sc):
                            if ((pu - (cu - .5) * 2) / (2 * ru)) ** 2 + ((pv - (cv - .5) * 2) / (2 * rv)) ** 2 < 1: M.c[o_ + i] = hexc
                else:
                    up += 1; cc = c + np.array([(cu - .5) * 2 * hx, (cv - .5) * 2 * hy]); rr = math.hypot((cu - .5) * 2, (cv - .5) * 2)
                    ring = [cc + np.array([math.cos(t) * 2 * ru * hx, math.sin(t) * 2 * rv * hy]) for t in np.linspace(0, 2 * math.pi, 11 if ru > 0.3 else 9)[:-1]]
                    o_, p2_, fr = dome(M, ring, 0.0025, h * math.sqrt(max(0.05, 1 - min(rr + ru, 0.97) ** 2)) * 0.9 + 0.0012 * up, hexc, rings=2)
                    done.append((o_, p2_))
    V = np.array(M.v); F = np.array(M.f)
    e = 2e-3; x, y, z = V[:, 0], V[:, 1], V[:, 2]
    n = np.stack([C.sdf(x + e, y, z) - C.sdf(x - e, y, z), C.sdf(x, y + e, z) - C.sdf(x, y - e, z), C.sdf(x, y, z + e) - C.sdf(x, y, z - e)], 1)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]); flip = (fn * n[F].mean(1)).sum(1) < 0     # 면은 바깥을 보게(게임은 뒷면을 안 그림)
    F[flip] = F[flip][:, ::-1]
    return V, M.c, F


# ───────── 새 얼굴(납작하게 찍은 눈) ─────────
# 그림 시안(1774×887 한 장: 위 여자, 아래 남자. 얼굴 중심 x=887)에서 잰 좌표·색을 그대로 옮긴다. 그림의 비율은 그대로 두고 머리에 들어가도록 전체를 줄여 붙인다(PX).
# 그림의 왼쪽 눈(바깥쪽이 그림 왼쪽)만 재서 좌우 대칭으로 쓴다. 하이라이트만 양쪽 다 화면 왼쪽.
# 게임 카메라가 위에서 내려다봐 세로가 눌리고 눈이 기울어 보이므로, 정면이 아니라 카메라 방향으로 머리 표면에 투영한다(proj_view). 게임 화면이 밝게 나와 색은 DK로 보정한다(게임 화면을 그림과 대조해 맞춘 값).
# 볼터치 가장자리는 몸 피부색(HEX['skin'])과 같게 푼다. 층마다 0.0012씩 띄워 겹친다.
SKIN = HEX['skin']
PX, CX = 326.0, 887.0          # 그림 261px(얼굴 중심~눈 가운데)를 EYE_X의 0.8배로: 1:1이면 볼터치 바깥쪽이 머리 윤곽 밖으로 넘어간다
# 배치(파츠 모양은 그림 그대로, 자리·크기만 머리에 맞춘다). PXF: 파츠 축척(그림 px당 EYE_X/PXF), KE: 눈 가운데 x(EYE_X 배), MZ: 입 높이, KB·DB·SB: 볼터치 x(EYE_X 배)·눈 아래 깊이(EYE_X 배)·크기 배율, EY: 눈 높이 이동(EYE_X 배). 얼굴마다 FLAT의 'lay'로 덮어쓸 수 있다(배치 도구 laytool.html)
LAY = dict(PXF=275.0, KE=0.98, EY=0.0, MZ=0.71, KB=1.1, DB=0.5, SB=0.85)   # 0.8배 통째 축소(PX)는 파츠가 가운데로 몰리고 입이 올라가 보여 따로 놓는다
if os.environ.get('LAY'): LAY.update(json.loads(os.environ['LAY']))
CAMP = math.atan2(125, 169.7)          # 게임 카메라가 내려다보는 각(play_game.js setCam: 수평 169.7, 높이 125)

def proj_view(P):
    """게임 카메라 방향으로 쏜 선이 머리 표면과 만나는 점과 바깥 법선. 입력은 화면 좌표(x, 눈높이 기준 위쪽)를 (x, EYE_Z+위쪽)으로 준 것.
    이렇게 붙이면 주민이 카메라를 볼 때 화면에 그림 그대로 보인다(정면 투영은 위에서 내려다보면 세로가 눌리고 눈이 기운다)"""
    P = np.asarray(P, float); X, Y = P[:, 0], P[:, 1] - C.EYE_Z; cs, sn = math.cos(CAMP), math.sin(CAMP)
    A = proj(np.array([[0.0, C.EYE_Z]]))[0][0]                                     # 기준점: 눈높이 정중앙의 표면
    ox, oy, oz = A[0] + X, A[1] + Y * sn, A[2] + Y * cs                            # 화면 위쪽 = (0, sin, cos), 보는 방향 = (0, cos, -sin)
    ts = np.linspace(-0.9, 0.7, 500)
    d = C.sdf(np.repeat(ox[:, None], 500, 1), oy[:, None] + ts[None, :] * cs, oz[:, None] - ts[None, :] * sn)
    hit = (d < 0).any(1); i = np.clip(np.argmax(d < 0, 1), 1, 499); r = np.arange(len(X))
    d0, d1 = d[r, i - 1], d[r, i]; t = ts[i - 1] + (ts[i] - ts[i - 1]) * d0 / np.maximum(d0 - d1, 1e-9)
    for _ in range(4):
        f = C.sdf(ox, oy + t * cs, oz - t * sn); e = 1e-3; g = (C.sdf(ox, oy + (t + e) * cs, oz - (t + e) * sn) - C.sdf(ox, oy + (t - e) * cs, oz - (t - e) * sn)) / (2 * e)
        t = t - f / np.where(np.abs(g) > 0.2, g, -1.0)
    x, y, z = ox, oy + t * cs, oz - t * sn; e = 2e-3
    n = np.stack([C.sdf(x + e, y, z) - C.sdf(x - e, y, z), C.sdf(x, y + e, z) - C.sdf(x, y - e, z), C.sdf(x, y, z + e) - C.sdf(x, y, z - e)], 1)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    proj_view.miss += int((~hit).sum())
    return np.stack([x, y, z], 1), n
proj_view.miss = 0
R = math.radians
def hx(h):return [int(h[i:i+2],16) for i in(1,3,5)]
def mix(a,b,t):A,Bb=hx(a),hx(b);return '#%02x%02x%02x'%tuple(int(round(A[i]+(Bb[i]-A[i])*t)) for i in range(3))
def flat(M,out,lift,col,grad=None,rings=2):
    """표면에 납작하게 붙는 판. grad=(위색,아래색)이면 세로 그러데이션(6단)"""
    o,P2,_=dome(M,np.asarray(out,float),0.0004,lift,col if isinstance(col,str) else '#000000',rings=rings)
    if grad:
        z=P2[:,1];z0,z1=z.min(),z.max()
        for i in range(len(P2)):t=round((z1-z[i])/max(z1-z0,1e-9)*5)/5;M.c[o+i]=mix(grad[0],grad[1],t)
def soft(M,out,lift,stops=None,radial=None,rings=5):
    """부드러운 색 변화. stops=[(t,색)...] 세로(위 t=0 → 아래 t=1), radial=(가운데색,가장자리색,시작 비율)"""
    out=np.asarray(out,float);o,P2,_=dome(M,out,0.0004,lift,'#000000',rings=rings);n=len(out)
    def q(h):return '#%02x%02x%02x'%tuple(min(255,int(round(v/4)*4)) for v in hx(h))
    if stops:
        z=P2[:,1];z0,z1=z.min(),z.max()
        for i in range(len(P2)):
            t=(z1-z[i])/max(z1-z0,1e-9)
            for k in range(len(stops)-1):
                if t<=stops[k+1][0] or k==len(stops)-2:
                    u=min(1,max(0,(t-stops[k][0])/max(stops[k+1][0]-stops[k][0],1e-9)));M.c[o+i]=q(mix(stops[k][1],stops[k+1][1],u));break
    if radial:
        c0,c1,st=radial;M.c[o]=q(c0)
        for r in range(1,rings+1):
            rho=r/rings;u=0 if rho<=st else ((rho-st)/(1-st))**1.3
            for k in range(n):M.c[o+1+(r-1)*n+k]=q(mix(c0,c1,u))
def ribbon(M,path,w,lift,col,taper=(.25,.25)):
    """납작한 선. w=반폭(숫자 또는 경로 따라 리스트), 양 끝은 가늘게"""
    P=np.asarray(path,float);d=[]
    for i in range(len(P)-1):
        for k in range(3):d.append(P[i]+(P[i+1]-P[i])*k/3)
    d.append(P[-1]);d=np.array(d);N=len(d);ws=np.interp(np.linspace(0,1,N),np.linspace(0,1,len(w)),w) if not np.isscalar(w) else np.full(N,w)
    P2=[]
    for i,p in enumerate(d):
        t=d[min(i+1,N-1)]-d[max(i-1,0)];t=t/max(np.linalg.norm(t),1e-9);n=np.array([-t[1],t[0]]);u=i/(N-1)
        tp=min(1,(u/.18)**.6*(1-taper[0])+taper[0]) if u<.18 else min(1,((1-u)/.18)**.6*(1-taper[1])+taper[1]) if u>.82 else 1
        for sg in(1,0,-1):P2.append(p+n*ws[i]*tp*sg)
    loc,nor=PROJ(np.array(P2));V=loc+nor*(lift+LIFT);Fc=[]
    for i in range(N-1):
        a,b=i*3,(i+1)*3
        for k in range(2):Fc+=[[a+k,b+k,b+k+1],[a+k,b+k+1,a+k+1]]
    M.add(V,col,Fc)
def band(M,lo,up,lift,col):
    """두 곡선(아래·위) 사이를 채운 납작한 면. 가운데 줄을 넣어 표면 곡률을 따른다"""
    lo=np.asarray(lo,float);up=np.asarray(up,float);N=len(lo);P2=[]
    for i in range(N):P2+=[lo[i],(lo[i]+up[i])/2,up[i]]
    loc,nor=PROJ(np.array(P2));V=loc+nor*(lift+LIFT);Fc=[]
    for i in range(N-1):
        a,b=i*3,(i+1)*3
        for k in range(2):Fc+=[[a+k,b+k,b+k+1],[a+k,b+k+1,a+k+1]]
    M.add(V,col,Fc)
def sup(cx,cy,rx,ry,n=2.0,m=20,rot=0.):
    o=[]
    for k in range(m):
        a=2*math.pi*k/m;c,s=math.cos(a),math.sin(a);x=rx*abs(c)**(2/n)*(1 if c>=0 else -1);y=ry*abs(s)**(2/n)*(1 if s>=0 else -1)
        o.append((cx+x*math.cos(rot)-y*math.sin(rot),cy+x*math.sin(rot)+y*math.cos(rot)))
    return o
def cr(pts,k=4,closed=False):
    """꼭짓점 사이를 매끄럽게 잇는다(Catmull-Rom)"""
    P=[np.array(p,float) for p in pts];n=len(P);o=[]
    g=lambda i:P[i%n] if closed else P[min(max(i,0),n-1)]
    for i in range(n if closed else n-1):
        p0,p1,p2,p3=g(i-1),g(i),g(i+1),g(i+2)
        for j in range(k):
            t=j/k;o.append(tuple(.5*((2*p1)+(-p0+p2)*t+(2*p0-5*p1+4*p2-p3)*t*t+(-p0+3*p1-3*p2+p3)*t**3)))
    if not closed:o.append(tuple(P[-1]))
    return o
def dk(h, k): return '#%02x%02x%02x' % tuple(int(round(v * k)) for v in hx(h))

FLAT = {
 'U': dict(name='여자 01', sex='f', eyl=222, lash='#543d35', rim='#6d5449', tone=('#785c4e', '#a58577', '#c4a292'), mouth='#a15148', brow='#966457',
           lash_up=[(476, 177), (492, 169), (520, 152), (550, 138), (580, 128), (615, 124), (650, 125), (690, 135), (725, 156), (751, 191)],
           lash_lo=[(503, 197), (512, 182), (528, 171), (550, 163), (580, 157), (620, 155), (660, 158), (700, 165), (727, 174), (750, 190)],
           wedge=[(476, 177), (503, 197), (499, 220), (500, 247), (490, 232), (481, 212), (478, 198)],
           flick=[(476, 177), (478, 198), (468, 197), (459, 193), (455, 187), (457, 181), (465, 178)],
           scl=[(503, 197), (512, 182), (528, 171), (580, 157), (660, 158), (725, 173), (745, 185), (751, 205), (753, 225), (751, 250), (742, 270), (725, 285), (700, 293), (605, 295), (575, 292), (545, 287), (520, 277), (506, 262), (500, 247), (499, 220)],
           shade=[[(725, 175), (737, 182), (748, 195), (752, 212), (745, 200), (735, 188), (724, 180)], [(505, 196), (513, 183), (528, 172), (552, 164), (570, 160), (566, 170), (540, 177), (522, 188), (512, 200)]],
           iris=(648, 222, 87, 75), ty=238, hill=(655, 300, 74, 50), hl=(48.5, 41, 198, 31.5, 11), rimline=[(603, 296), (640, 298), (670, 298), (702, 296)],
           brow_up=[(522, 84), (550, 74), (580, 65), (610, 59), (640, 57), (670, 59), (700, 65), (725, 74), (738, 83)],
           brow_lo=[(522, 86), (550, 80), (580, 74), (610, 70), (640, 69), (670, 71), (700, 75), (725, 80), (738, 84)],
           blush=(540, 362, 100, 48), hatch=[(508, 348, 526, 381), (545, 346, 565, 381), (583, 346, 601, 376)],
           mouthp=[(833, 386), (843, 390), (858, 393), (887, 394), (916, 393), (931, 390), (941, 386)]),
 'V': dict(name='남자 01', sex='m', eyl=672, lash='#5b453a', rim='#6d5448', tone=('#795b4e', '#aa897a', '#c9a797'), mouth='#a55249', brow='#614c42',
           lash_up=[(460, 629), (480, 618), (505, 606), (535, 596), (565, 590), (600, 587), (640, 589), (685, 598), (725, 614), (754, 647)],
           lash_lo=[(478, 640), (490, 641), (505, 634), (530, 626), (560, 620), (600, 616), (640, 615), (690, 619), (728, 630), (753, 646)],
           wedge=[(478, 640), (490, 641), (490, 657), (493, 676), (484, 660)],
           scl=[(490, 641), (505, 634), (560, 620), (640, 615), (728, 630), (748, 645), (753, 660), (754, 680), (751, 705), (742, 722), (725, 735), (700, 743), (605, 745), (580, 742), (555, 738), (530, 730), (512, 718), (500, 700), (494, 680), (490, 657)],
           shade=[[(725, 628), (738, 636), (749, 648), (753, 664), (746, 653), (736, 642), (724, 634)], [(492, 642), (505, 635), (530, 627), (565, 620), (570, 630), (535, 638), (508, 646), (494, 652)]],
           iris=(648, 672, 87, 75), ty=688, hill=(655, 750, 74, 50), hl=(48.5, 41, 649, 31.5, 12), rimline=[(603, 746), (640, 748), (670, 748), (702, 746)],
           brow_up=[(486, 540), (500, 527), (520, 518), (550, 511), (590, 508), (630, 509), (670, 513), (710, 519), (745, 527), (759, 544)],
           brow_lo=[(486, 542), (510, 540), (550, 538), (590, 538), (630, 541), (670, 545), (710, 550), (745, 554), (755, 554), (759, 547)],
           blush=(540, 812, 100, 48), hatch=[(508, 798, 526, 831), (545, 796, 565, 831), (583, 796, 601, 826)],
           mouthp=[(833, 834), (843, 838), (858, 841), (887, 842), (916, 841), (931, 838), (941, 834)]),
}
# 게임 화면 색 보정: 넣은 색(왼쪽)이 게임 화면에서 어떻게 보였는지(오른쪽)를 주민 클로즈업에서 잰 값. 조명·톤 보정 때문에 밝고 붉게 나온다.
# G(목표색)은 이 표를 채널별로 거꾸로 읽어 "그림의 색으로 보이려면 넣어야 할 색"을 돌려준다.
SEEN = [('#34261f', '#482e24'), ('#4d423e', '#5b463a'), ('#574942', '#7c5e4c'), ('#674c46', '#945e4d'), ('#705b53', '#c6a38f'), ('#82685e', '#d5b3a2'), ('#c4a292', '#f5e2d5'), ('#e4b491', '#feeeda')]
ADJ = dict(top=(.90, .91, .84), mid=(.85, .85, .86))      # 표로 맞춘 뒤에도 남은 차이(게임 화면 실측/그림)를 한 번 더 나눈 값
def adj(h, k): return '#%02x%02x%02x' % tuple(int(round(v * f)) for v, f in zip(hx(h), ADJ[k]))
MOUTH_IN = '#5a302e'            # 입은 붉은 색이라 위 표로는 안 맞아 따로 잰 값(그림 #a15148로 보이게)
def G(h):
    o = []
    for c in range(3):
        xs = [0] + [hx(a)[c] for a, _ in SEEN] + [255]; ys = [0] + [hx(b)[c] for _, b in SEEN] + [255]
        k = np.argsort(ys); o.append(int(round(float(np.interp(hx(h)[c], np.array(ys)[k], np.array(xs)[k])))))
    return '#%02x%02x%02x' % tuple(o)

def build_flat(q):
    global PROJ
    PROJ = proj_view; proj_view.miss = 0
    LAY = dict(globals()['LAY'], **q.get('lay', {}))
    M = Mesh(); EZ = C.EYE_Z + LAY['EY'] * C.EYE_X; u = C.EYE_X / LAY['PXF']; EYL = q['eyl']; ECX = q['iris'][0]
    XE = LAY['KE'] * C.EYE_X                                                         # 눈·눈썹: 홍채 가운데를 (±XE, EZ)에 두고 그림 모양 그대로
    def P(s, px, py): return (s * (XE + (ECX - px) * u), EZ + (EYL - py) * u)
    PB = P
    bx0, by0 = q['blush'][:2]; SB = LAY['SB']
    XB = LAY['KB'] * C.EYE_X if LAY['KB'] is not None else (CX - bx0) * u
    ZB = EZ - (LAY['DB'] * C.EYE_X if LAY['DB'] is not None else (by0 - EYL) * u)
    def PK(s, px, py): return (s * (XB + (bx0 - px) * u * SB), ZB + (by0 - py) * u * SB)   # 볼터치·빗금: 볼 가운데 기준
    my0 = sum(y for _, y in q['mouthp']) / len(q['mouthp'])
    ZM = LAY['MZ'] if LAY['MZ'] is not None else EZ - (my0 - EYL) * u
    def PM(px, py): return (px - CX) * u, ZM + (my0 - py) * u                          # 입: 가운데 기준
    LC, RC = G(q['lash']), G(q['rim'])
    for s in (1, -1):
        Q = lambda pts: [P(s, x, y) for x, y in pts]
        bx, by, brx, bry = q['blush']
        soft(M, [PK(s, bx + brx * math.cos(t), by + bry * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 17)[:-1]], .0012, radial=(mix(SKIN, G('#fbc7ba'), .9), SKIN, .0), rings=5)
        for (x0, y0, x1, y1) in q['hatch']:
            if s > 0: x0, x1 = 2 * bx - x1, 2 * bx - x0; y0, y1 = y1, y0                 # 빗금은 양쪽 모두 같은 방향
            ribbon(M, [PK(s, x, y) for x, y in [(x0, y0), ((x0 + x1) / 2, (y0 + y1) / 2), (x1, y1)]], 4 * u * SB, .0030, mix(SKIN, G('#f3b3aa'), .8), taper=(.9, .9))
        flat(M, Q(cr(q['scl'], 2, True)), .0012, '#fdf6ee', rings=3)
        for sh in q.get('shade', []): flat(M, Q(sh), .0020, G('#e3e0d9'))
        cx, cy, rx, ry = q['iris']; T0, T1, T2 = q['tone']
        el = lambda k=0, n=28: [(x, y) for x, y in sup(cx, cy, rx - k, ry - k, 2.5, n)]
        def inir(x, y, k=8):                                                          # 홍채(테두리 안쪽) 밖으로 나간 점은 안으로 당긴다
            f = abs((x - cx) / (rx - k)) ** 2.5 + abs((y - cy) / (ry - k)) ** 2.5
            if f > 1: g = f ** (-1 / 2.5); x, y = cx + (x - cx) * g, cy + (y - cy) * g
            return (x, y)
        flat(M, Q(el()), .0030, RC, rings=3)
        flat(M, Q(el(6)), .0042, adj(G(T1), 'mid'), rings=3)
        flat(M, Q([(x, min(y, q['ty'])) for x, y in el(6)]), .0054, adj(G(T0), 'top'), rings=3)
        hx_, hy_, hrx, hry = q['hill']                                                 # 아래쪽 밝은 언덕
        soft(M, Q([inir(x, y) for x, y in sup(hx_, hy_, hrx, hry, 2, 20)]), .0054, radial=(G(T2), adj(G(T1), 'mid'), .72), rings=4)
        band(M, Q(cr(q['lash_lo'], 3)), Q(cr(q['lash_up'], 3)), .0078, LC)
        flat(M, Q(q['wedge']), .0078, LC)
        if q.get('flick'): flat(M, Q(cr(q['flick'], 2, True)), .0078, LC)
        hl, hr, hy, hw, hh = q['hl']; hc = P(s, cx, hy)                               # 하이라이트: 가로로 긴 알약 모양, 양쪽 눈 모두 화면 왼쪽
        px0 = hc[0] - (hl if s < 0 else hr) * u; pill = []
        for sg, a0 in ((1, -90), (-1, 90)):
            for a in range(a0, a0 + 181, 30): pill.append((px0 + sg * (hw - hh) * u + math.cos(R(a)) * hh * u, hc[1] + math.sin(R(a)) * hh * u))
        flat(M, pill, .0090, '#ffffff')
        QB = lambda pts: [PB(s, x, y) for x, y in pts]
        band(M, QB(cr(q['brow_lo'], 2)), QB(cr(q['brow_up'], 2)), .0012, G(q['brow']))
    ribbon(M, [PM(x, y) for x, y in q['mouthp']], 5.5 * u, .0012, MOUTH_IN, taper=(1, 1))
    V = np.array(M.v); Fc = np.array(M.f); e = 2e-3; x, y, z = V[:, 0], V[:, 1], V[:, 2]
    nn = np.stack([C.sdf(x + e, y, z) - C.sdf(x - e, y, z), C.sdf(x, y + e, z) - C.sdf(x, y - e, z), C.sdf(x, y, z + e) - C.sdf(x, y, z - e)], 1)
    fn = np.cross(V[Fc[:, 1]] - V[Fc[:, 0]], V[Fc[:, 2]] - V[Fc[:, 0]]); fl = (fn * nn[Fc].mean(1)).sum(1) < 0; Fc[fl]=Fc[fl][:, ::-1]
    PROJ = proj
    if proj_view.miss: print('  표면을 못 만난 점:', proj_view.miss)
    return V, M.c, Fc

NEW = {k: (q['name'], q['sex'], (lambda q=q: build_flat(q))) for k, q in FLAT.items()}        # 글쇠: (이름, 성별, 만드는 함수)

def body_normals(T):
    """three 좌표의 점들 바로 밑 base15.glb 삼각형에서 꼭짓점 법선을 무게중심 보간한 값(게임이 피부를 칠할 때 쓰는 법선과 같다)"""
    import struct
    b = open(os.path.join(D0, 'base15.glb'), 'rb').read(); n = struct.unpack('<I', b[12:16])[0]; j = json.loads(b[20:20 + n]); bn = b[20 + n + 8:]
    def acc(i, dt=np.float32, k=3):
        a = j['accessors'][i]; v = j['bufferViews'][a['bufferView']]; dt = {5123: np.uint16, 5125: np.uint32, 5126: np.float32}[a['componentType']]
        return np.frombuffer(bn, dt, a['count'] * k, v.get('byteOffset', 0) + a.get('byteOffset', 0))
    pr = j['meshes'][0]['primitives'][0]; BP = acc(pr['attributes']['POSITION']).reshape(-1, 3).astype(float); BN = acc(pr['attributes']['NORMAL']).reshape(-1, 3).astype(float)
    I = acc(pr['indices'], k=1).reshape(-1, 3).astype(int); A, B, Cc = BP[I[:, 0]], BP[I[:, 1]], BP[I[:, 2]]; cen = (A + B + Cc) / 3
    out = np.zeros_like(T)
    for i, p in enumerate(T):
        near = np.argsort(((cen - p) ** 2).sum(1))[:16]; best = None
        for t in near:
            a, e1, e2 = A[t], B[t] - A[t], Cc[t] - A[t]; nrm = np.cross(e1, e2); nn = nrm / max(np.linalg.norm(nrm), 1e-12); q = p - nn * ((p - a) @ nn)
            d00, d01, d11, d20, d21 = e1 @ e1, e1 @ e2, e2 @ e2, (q - a) @ e1, (q - a) @ e2; den = d00 * d11 - d01 * d01
            v = (d11 * d20 - d01 * d21) / den; w = (d00 * d21 - d01 * d20) / den; bc = np.clip([1 - v - w, v, w], 0, 1); bc = bc / bc.sum()
            pc = bc[0] * A[t] + bc[1] * B[t] + bc[2] * Cc[t]; dist = np.linalg.norm(p - pc)
            if best is None or dist < best[0]: best = (dist, t, bc)
        _, t, bc = best; nv = bc[0] * BN[I[t, 0]] + bc[1] * BN[I[t, 1]] + bc[2] * BN[I[t, 2]]; out[i] = nv / max(np.linalg.norm(nv), 1e-9)
    return out

if __name__ == '__main__':
    pal = []; out = {'coord': 'three.js Y-up, +z front (base15.glb와 같은 공간)', 'names': NAMES, 'faces': {}}
    for st in NAMES:
        V, Cc, F = build(st); ci = []
        for h in Cc:
            if h not in pal: pal.append(h)
            ci.append(pal.index(h))
        T = np.stack([V[:, 0], V[:, 2], -V[:, 1]], 1)                # 모델(x, -y 정면, z 위) → three(x, y 위, z 정면)
        out['faces'][st] = {'v': [round(float(a), 4) for a in T.ravel()], 'c': ci, 'f': [int(a) for a in F.ravel()]}
        print(st, NAMES[st], 'verts', len(V), 'tris', len(F))
    out['sex'] = {}
    for st, (nm, sx, fn) in NEW.items():
        V, Cc, F = fn(); ci = []
        for h in Cc:
            if h not in pal: pal.append(h)
            ci.append(pal.index(h))
        T = np.stack([V[:, 0], V[:, 2], -V[:, 1]], 1)                # 모델(x, -y 정면, z 위) → three(x, y 위, z 정면)
        Nn = body_normals(T)                                         # 납작한 조각은 바로 밑 몸 메시의 음영 법선을 그대로 써야 피부와 이어진다
        out['faces'][st] = {'v': [round(float(a), 4) for a in T.ravel()], 'c': ci, 'f': [int(a) for a in F.ravel()], 'n': [round(float(a), 3) for a in Nn.ravel()]}
        out['names'][st] = nm; out['sex'][st] = sx
        print(st, nm, sx, 'verts', len(V), 'tris', len(F))
    out['pal'] = [lin(h) for h in pal]
    json.dump(out, open(os.path.join(D0, 'faces15.json'), 'w', encoding='utf-8'), ensure_ascii=False, separators=(',', ':'))
    print('faces15.json', os.path.getsize(os.path.join(D0, 'faces15.json')))
