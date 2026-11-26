# 주민 베이스 (1.5등신 클레이)

머리가 키의 2/3인 1.5등신 캐릭터. 부위별 거리장(SDF)을 부드럽게 합쳐 한 덩어리 표면을 뽑으므로 관절 이음새가 형태로 드러나지 않는다.
치수는 `params.json`(조정 도구에서 확정한 값)에 있고, `chibi15.py`의 `Q` 기본값을 덮어쓴다.

| 파일 | 내용 |
|---|---|
| `chibi15.py` | 형태 생성(거리장 → 메시). `PARAMS=params.json MESH=mesh.npz python3 chibi15.py` |
| `render15.py` | 블렌더(bpy): 클레이 재질·뼈대(19개)·가중치·얼굴 파츠, 점검 렌더, GLB 내보내기 |
| `params.json` | 확정 치수 |
| `base15.glb` | 게임용 모델: 스킨 메시 1개(약 2,700 삼각형), 얼굴은 꼭짓점 색으로 구움 |
| `hair01_lite.json` | 헤어 01 데이터(프로젝트 문서 float-hair 참조) |
| `faces15.py`, `faces15.json` | 얼굴 생성기와 결과(머리 표면에 투영한 파츠, 꼭짓점 색): 점토 얼굴 20종(A~T, 지금은 안 씀) + 납작하게 찍은 새 얼굴(U = 여자 01, V = 남자 01. 그림 시안의 좌표·색을 옮겨 게임 카메라 방향으로 머리에 투영, 색은 게임 화면 실측으로 보정. 법선 `n`(몸 메시에서 보간)·성별 `sex` 포함) |
| `facenorm.py`, `base15_raw.glb` | 얼굴 피부 다듬기(`python3 facenorm.py 0.55` → `base15.glb`): render15가 뽑은 `base15_raw.glb`에서 뒤집혀 접힌 삼각형(관자놀이·턱 옆)을 펴고, 머리 법선을 얼굴 굴곡 0.55배 거리장 법선으로 바꾼다. 면 수 그대로(2000). 돌린 뒤 `faces15.py` 다시 실행 |
| `hairlod.mjs`, `hair0*_lo.json` | 헤어 저폴리(`node hairlod.mjs`, three 0.160 필요): 헤어·캡을 1/5로 줄인 모양. 게임은 줌 14보다 멀면 주민 머리를 이것으로 바꾼다(`play_game.js` HLK). 헤어를 고치면 다시 돌린다 |
| `laytool.py`, `laytool_src.html` | 얼굴 배치 도구(`python3 laytool.py` → `build/laytool.html`): 눈·입·볼터치의 자리·크기를 슬라이더로 옮겨 보고 값을 복사한다. 확정 값은 `faces15.py`의 `LAY`(또는 얼굴별 `FLAT[..]['lay']`)에 넣고 다시 굽는다 |
| `mktest.py` | 시험판 빌드 |

게임용 모델 다시 만들기(bpy, scikit-image 필요):

```
cd tools/3d/base15
PARAMS=params.json MESH=mesh.npz python3 chibi15.py
PARAMS=params.json MESH=mesh.npz JOBS=gglb GNAME=base15 GTRI=2000 GNOFACE=1 OUTD=. python3 render15.py
mv base15.glb base15_raw.glb && python3 facenorm.py 0.55
PARAMS=params.json python3 faces15.py
```

게임 쪽(`play_game.js`의 "주민" 구역): 모든 빌드에 이 모델을 싣는다(2026-11-22 결재로 기존 주민과 `ppl.json`은 삭제). 헤어는 `hair01_lite.json`(헤어 01: 머리카락 3000 + 두피 캡 1499 삼각형, 몸 모델과 같은 좌표)을 head 뼈 밑에 붙인다(`b15Hair`). 두상은 이 헤어에 맞춘 수정(목덜미 리세스·목 필렛·눈썹뼈/콧대 음영·코 +0.012)이 `chibi15.py`에 들어 있다. 얼굴은 `faces15.py` → `faces15.json`에서 주민마다 무작위로 붙인다. 주민은 처음 한 번 성별을 정하고(스님은 남자), `sex`가 `f`인 얼굴(U)은 여자에게, `m`인 얼굴(V)은 남자에게 간다. 표시 없는 옛 점토 얼굴(A~T)은 쓰지 않는다. 그래서 그래서 `base15.glb`는 얼굴 없이 몸만 뽑는다(`GNOFACE=1`). 옷은 아직 없다.
시험판: `python3 tools/3d/base15/mktest.py` → `build/float-resident-test.html`.

좌표: x 좌우, -y 정면, z 위, 바닥 z=0, 머리 높이 ≈ 1. 게임에서는 배율 .33(기존 주민과 같은 키).
