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
    P2 = np.array(P2); loc, nor = proj(P2); V = loc + nor * (np.array(H)[:, None] + LIFT)
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
    loc, nor = proj(np.array(P2)); V = loc + nor * (np.array(H)[:, None] + LIFT); F = []
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
# 그림 시안(1024×665, 얼굴 중심 x=512, 눈 높이 y=318)에서 잰 좌표·색을 그대로 옮긴다. 눈 가운데까지 271px = base15의 EYE_X.
# 게임 카메라가 위에서 내려다봐 세로가 눌리므로 눈·속눈썹·눈썹 두께는 세로 1.25배, 입은 위로 올리고 너비 0.7배, 선 색은 그림보다 어둡게 넣는다.
# 볼터치 가장자리는 몸 피부색(HEX['skin'])과 같게 풀고, 눈썹은 헤어 색(hair01_lite.json)을 쓴다. 층마다 0.0012씩 띄워 겹친다.
SKIN = HEX['skin']
HAIR = json.load(open(os.path.join(D0, 'hair01_lite.json'), encoding='utf-8'))['color']['hair']
PX = 271.0
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
    loc,nor=proj(np.array(P2));V=loc+nor*(lift+LIFT);Fc=[]
    for i in range(N-1):
        a,b=i*3,(i+1)*3
        for k in range(2):Fc+=[[a+k,b+k,b+k+1],[a+k,b+k+1,a+k+1]]
    M.add(V,col,Fc)
def band(M,lo,up,lift,col):
    """두 곡선(아래·위) 사이를 채운 납작한 면. 가운데 줄을 넣어 표면 곡률을 따른다"""
    lo=np.asarray(lo,float);up=np.asarray(up,float);N=len(lo);P2=[]
    for i in range(N):P2+=[lo[i],(lo[i]+up[i])/2,up[i]]
    loc,nor=proj(np.array(P2));V=loc+nor*(lift+LIFT);Fc=[]
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
def build_u():
    M=Mesh();EZ=C.EYE_Z;EXB=C.EYE_X;u=EXB/PX
    VS=1.25
    def P(s,px,py):return (s*(512-px)*u,EZ+(318-py)*u*(VS if py<430 else 1))
    def PB(s,px,py):return (s*(512-px)*u,EZ+(318-198)*u*VS+(198-73)*u+(73-py)*u*VS)      # 눈썹(중심선 y=73)       # 그림의 왼쪽 눈(바깥쪽이 그림 왼쪽)을 양쪽에 대칭으로
    for s in(1,-1):
        Q=lambda pts:[P(s,x,y) for x,y in pts]
        soft(M,[P(s,190+112*math.cos(t),506+56*math.sin(t)) for t in np.linspace(0,2*math.pi,17)[:-1]],.0012,radial=(mix(SKIN,'#f07f8c',.42),SKIN,.0),rings=5)
        for (x0,y0,x1,y1) in((136,496,167,540),(175,489,206,536),(215,488,240,526)):
            if s>0:x0,x1=380-x1,380-x0;y0,y1=y1,y0          # 빗금은 양쪽 모두 같은 방향
            ribbon(M,Q([(x0,y0),((x0+x1)/2,(y0+y1)/2),(x1,y1)]),2.4*u,.0030,mix(SKIN,'#e06a7c',.55),taper=(.5,.5))
        # 흰자 + 위쪽 그늘
        scl=[(93,268),(150,243),(200,232),(250,231),(300,236),(340,247),(366,266),(385,297),(388,330),(378,368),(350,389),(300,398),(250,401),(200,398),(150,385),(120,364),(105,338),(97,300)]
        flat(M,Q(cr(scl,2,True)),.0012,'#fbf7eb',rings=3)
        band(M,Q(cr([(96,293),(160,277),(215,265),(280,262),(330,268),(368,286),(390,308)])),Q(cr([(93,268),(150,243),(205,232),(270,232),(325,241),(368,261),(391,293)])),.0020,'#ccccc2')
        # 홍채
        cx,cy,r=253,318,92;circ=lambda rr,n=22,ymax=398:[(cx+rr*math.cos(t),min(cy+rr*math.sin(t),ymax)) for t in np.linspace(0,2*math.pi,n+1)[:-1]]
        flat(M,Q(circ(r)),.0030,'#33221a',rings=3)
        flat(M,Q(circ(r-6,ymax=395)),.0042,'#93775f',rings=3)
        flat(M,Q([(x,min(y,306+6*abs((x-cx)/r))) for x,y in circ(r-6)]),.0054,'#553c2c',rings=3)
        flat(M,Q([(x,max(y,348)) for x,y in circ(r-6,ymax=395)]),.0054,'#d3c3a7',rings=3)
        flat(M,Q([(251+33*math.cos(t),311+48*math.sin(t)) for t in np.linspace(0,2*math.pi,15)[:-1]]),.0066,'#40291b')
        # 아래 속눈썹 선 + 바깥쪽 아래 잔가닥 2개
        ribbon(M,Q(cr([(142,385),(170,392),(200,398),(250,402),(290,401),(320,397)],2)),[1.2*u,2.2*u,2.8*u,2.8*u,2.2*u,1.2*u],.0078,'#1f120e',taper=(.4,.4))
        for b0,b1,tp in(((146,384),(156,389),(133,393)),((170,392),(182,396),(159,403))):band(M,Q([b0,((b0[0]+tp[0])/2,(b0[1]+tp[1])/2),tp]),Q([b1,((b1[0]+tp[0])/2+.3,(b1[1]+tp[1])/2+.3),(tp[0]+.2,tp[1]+.2)]),.0078,'#1f120e')
        # 윗속눈썹: 몸통 띠 + 바깥쪽 아래 쐐기 + 가닥 2개
        band(M,Q(cr([(22,253),(60,262),(92,268),(150,243),(200,232),(250,231),(300,236),(340,246),(372,264),(400,296)],3)),Q(cr([(22,253),(60,244),(100,226),(150,208),(200,198),(250,199),(300,211),(340,234),(375,268),(400,296)],3)),.0078,'#1f120e')
        flat(M,Q([(58,258),(92,266),(99,300),(106,340),(88,318),(70,288)]),.0078,'#1f120e')
        for b0,b1,tp in(((98,228),(124,214),(72,202)),((172,201),(196,198),(165,181))):band(M,Q([b0,((b0[0]*.45+tp[0]*.55),(b0[1]*.45+tp[1]*.55)),tp]),Q([b1,((b1[0]*.4+tp[0]*.6),(b1[1]*.4+tp[1]*.6)),(tp[0]+.2,tp[1]+.2)]),.0078,'#1f120e')
        # 하이라이트(양쪽 눈 모두 화면 왼쪽 위)
        hc=P(s,cx,cy);flat(M,sup(hc[0]-63*u,hc[1]+48*u*VS,25*u,25*u*VS,2,12),.0090,'#f9faf5')
        # 쌍꺼풀 선
        ribbon(M,Q(cr([(222,160),(260,162),(300,168),(350,187),(382,211)],2)),2.1*u,.0012,'#9a5a48',taper=(.4,.3))
        # 눈썹
        QB=lambda pts:[PB(s,x,y) for x,y in pts]
        band(M,QB(cr([(30,92.5),(60,86),(100,81),(150,78),(200,78),(250,81),(300,87),(350,95),(390,100),(403,101)],2)),QB(cr([(30,91.5),(60,78),(100,60),(150,49),(200,46),(250,48),(300,53),(350,61),(385,74),(402,97)],2)),.0012,HAIR)
    X=lambda px:(px-512)*u*.70;Z=lambda py:EZ+(318-(py-34))*u
    ribbon(M,[(X(407),Z(562)),(X(419),Z(571)),(X(465),Z(570.5)),(X(512),Z(570)),(X(559),Z(570.5)),(X(605),Z(571)),(X(617),Z(562))],3.4*u,.0012,'#7a2c24',taper=(.6,.6))
    V=np.array(M.v);Fc=np.array(M.f);e=2e-3;x,y,z=V[:,0],V[:,1],V[:,2]
    nn=np.stack([C.sdf(x+e,y,z)-C.sdf(x-e,y,z),C.sdf(x,y+e,z)-C.sdf(x,y-e,z),C.sdf(x,y,z+e)-C.sdf(x,y,z-e)],1)
    fn=np.cross(V[Fc[:,1]]-V[Fc[:,0]],V[Fc[:,2]]-V[Fc[:,0]]);fl=(fn*nn[Fc].mean(1)).sum(1)<0;Fc[fl]=Fc[fl][:,::-1]
    return V,M.c,Fc

NEW = {'U': ('여자 01', 'f', build_u)}        # 글쇠: (이름, 성별, 만드는 함수)

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
        e = 2e-3; x, y, z = V[:, 0], V[:, 1], V[:, 2]                # 납작한 조각은 머리 표면의 법선을 그대로 써야 피부와 음영이 이어진다
        n = np.stack([C.sdf(x + e, y, z) - C.sdf(x - e, y, z), C.sdf(x, y + e, z) - C.sdf(x, y - e, z), C.sdf(x, y, z + e) - C.sdf(x, y, z - e)], 1); n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
        T = np.stack([V[:, 0], V[:, 2], -V[:, 1]], 1); Nn = np.stack([n[:, 0], n[:, 2], -n[:, 1]], 1)
        out['faces'][st] = {'v': [round(float(a), 4) for a in T.ravel()], 'c': ci, 'f': [int(a) for a in F.ravel()], 'n': [round(float(a), 3) for a in Nn.ravel()]}
        out['names'][st] = nm; out['sex'][st] = sx
        print(st, nm, sx, 'verts', len(V), 'tris', len(F))
    out['pal'] = [lin(h) for h in pal]
    json.dump(out, open(os.path.join(D0, 'faces15.json'), 'w', encoding='utf-8'), ensure_ascii=False, separators=(',', ':'))
    print('faces15.json', os.path.getsize(os.path.join(D0, 'faces15.json')))
