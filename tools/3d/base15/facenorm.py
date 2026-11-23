# 얼굴 음영 줄이기: base15.glb의 모양은 그대로 두고 법선만 바꾼다.
# 얼굴 굴곡(눈자리·눈썹뼈·콧대·미간·볼·입 주변·코)의 세기를 FK배로 줄인 거리장의 기울기와 원래 기울기의 차이만큼 꼭짓점 법선을 돌린다.
# 얼굴 굴곡이 없는 곳(뒤통수·몸)은 두 기울기가 같아 법선이 그대로다. 실행 후 faces15.py를 다시 돌려 얼굴 조각 법선도 맞춘다.
#   python3 facenorm.py 0.55      (원본 법선은 base15_n0.npy에 한 번만 보관해 몇 번을 돌려도 원본 기준)
import os, sys, json, struct
import numpy as np
D0 = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('PARAMS', os.path.join(D0, 'params.json')); sys.path.insert(0, D0)
import chibi15 as C
k = float(sys.argv[1]) if len(sys.argv) > 1 else 0.55
fn = os.path.join(D0, 'base15.glb'); b = bytearray(open(fn, 'rb').read())
n = struct.unpack('<I', b[12:16])[0]; j = json.loads(b[20:20 + n]); B0 = 20 + n + 8
pr = j['meshes'][0]['primitives'][0]
def view(i):
    a = j['accessors'][i]; v = j['bufferViews'][a['bufferView']]; assert a['componentType'] == 5126 and 'byteStride' not in v
    return B0 + v.get('byteOffset', 0) + a.get('byteOffset', 0), a['count']
po, cnt = view(pr['attributes']['POSITION']); no, cn2 = view(pr['attributes']['NORMAL']); assert cnt == cn2
P = np.frombuffer(bytes(b[po:po + cnt * 12]), np.float32).reshape(-1, 3).astype(float)
f0 = os.path.join(D0, 'base15_n0.npy')
N0 = np.load(f0) if os.path.exists(f0) else np.frombuffer(bytes(b[no:no + cnt * 12]), np.float32).reshape(-1, 3).astype(float)
if not os.path.exists(f0): np.save(f0, N0)
x, y, z = P[:, 0], -P[:, 2], P[:, 1]                                      # three → 모델(x, -y 정면, z 위)
def grad(fk):
    C.FK = fk; e = 1e-3
    g = np.stack([C.sdf(x + e, y, z) - C.sdf(x - e, y, z), C.sdf(x, y + e, z) - C.sdf(x, y - e, z), C.sdf(x, y, z + e) - C.sdf(x, y, z - e)], 1)
    g /= np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9); return np.stack([g[:, 0], g[:, 2], -g[:, 1]], 1)   # → three
d = grad(k) - grad(1.0); C.FK = 1.0
N = N0 + d; N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
b[no:no + cnt * 12] = N.astype(np.float32).tobytes(); open(fn, 'wb').write(bytes(b))
ch = np.degrees(np.arccos(np.clip((N * N0).sum(1), -1, 1)))
print('FK', k, '바뀐 꼭짓점', int((ch > .05).sum()), '/', cnt, '최대 %.1f°' % ch.max())
