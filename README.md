# 📷 Vision AI 실습 강의노트 — YOLO26으로 웹캠 영상 분석하기

> **실습 날짜:** 2026-10-07
> **도구:** Python · [Ultralytics YOLO26](https://docs.ultralytics.com/) · OpenCV · GitHub · Claude Code

---

## 📌 오늘의 학습 목표

1. 비전 AI가 하는 대표적인 작업 4가지(분류, 탐지, 분할, 추적)의 차이를 이해한다.
2. 오픈소스 라이브러리 **Ultralytics**의 YOLO26 모델로 웹캠 실시간 분석 코드를 실행한다.
3. 작성한 코드를 **GitHub**에 올려 관리하는 방법을 익힌다.

---

## 1. 워밍업 — Teachable Machine

[Teachable Machine](https://teachablemachine.withgoogle.com/)은 Google이 만든 웹 서비스로, **코딩 없이** 브라우저에서 이미지·소리·포즈(모션)를 학습시켜 AI 모델을 만들 수 있다.

- 웹캠으로 사진을 몇십 장 찍어 클래스별로 나누고 → **학습** 버튼 → 바로 테스트
- 이때 하는 작업이 바로 아래 표의 **분류(Classification)** 이다.

> 💡 **포인트:** Teachable Machine은 "이 사진 전체가 무엇인가?"만 알려준다.
> "그 물체가 화면의 **어디에** 있는가?"를 알려면 다음 단계의 기술이 필요하다.

---

## 2. 비전 AI의 4가지 핵심 작업

오픈소스 비전 라이브러리 **Ultralytics**([공식 문서](https://docs.ultralytics.com/))는 아래 작업들을 모두 지원한다.

| 작업 (영문) | 한글 | 결과물 | 예시 도구 / 모델 |
|---|---|---|---|
| **Classification** | 분류 | 이미지 전체의 **라벨** 1개 (예: "고양이") | Teachable Machine, `yolo26n-cls.pt` |
| **Object Detection** | 객체 인식(탐지) | 객체마다 **사각형 박스 좌표** + 라벨 | `yolo26n.pt` |
| **Segmentation** | 분할 | 객체 모양 그대로의 **픽셀 마스크** (누끼, 박스보다 훨씬 정확) | `yolo26n-seg.pt` |
| **Object Tracking** | 객체 추적 | 영상에서 박스 좌표 + 프레임 간 유지되는 **ID** | `yolo26n.pt` + ByteTrack |

```
분류        →  "사람이 있다"
탐지        →  "사람이 (120, 80) ~ (340, 460) 박스 안에 있다"
분할        →  "사람의 정확한 윤곽은 이 픽셀들이다"
추적        →  "1번 사람은 계속 1번, 2번 사람은 계속 2번"
```

> 💡 **기억하기:** 아래로 갈수록 더 많은 정보를 준다. (라벨 → 박스 → 윤곽 → 시간에 따른 ID)

---

## 3. 실습 환경 준비

### 3-1. 라이브러리 설치

```bash
pip install -U ultralytics opencv-python
```

- `ultralytics` : YOLO 모델 (PyTorch도 함께 설치됨)
- `opencv-python` : 웹캠 영상 읽기 + 화면에 창 띄우기

### 3-2. Python 파일 실행 방법

```bash
python 파일이름.py
```

예) `python webcam_detect.py`

> 💡 모델 가중치 파일(`yolo26n.pt` 등)은 **처음 실행할 때 자동으로 다운로드**된다. 따로 받을 필요 없음!

---

## 4. 실습 코드

이 저장소에는 3개의 실습 파일이 있다. 모두 **웹캠(0번)** 을 입력으로 받는다.

| 파일 | 작업 | 기본 모델 |
|---|---|---|
| `webcam_detect.py` | 객체 탐지 | `yolo26n.pt` |
| `webcam_segment.py` | 분할 | `yolo26n-seg.pt` |
| `webcam_track.py` | 객체 추적 | `yolo26n.pt` + `bytetrack.yaml` |

### 4-1. 공통 뼈대 — 세 코드 모두 이 구조!

```python
from ultralytics import YOLO
import cv2

model = YOLO("yolo26n.pt")          # ① 모델 불러오기
cap = cv2.VideoCapture(0)           # ② 웹캠 열기 (0 = 기본 웹캠)

while True:
    ok, frame = cap.read()          # ③ 한 프레임 읽기
    results = model.predict(frame)  # ④ AI로 분석
    annotated = results[0].plot()   # ⑤ 결과를 그림으로 그리기
    cv2.imshow("YOLO26", annotated) # ⑥ 화면에 보여주기
    if cv2.waitKey(1) == ord("q"):  # ⑦ q 누르면 종료
        break

cap.release()
cv2.destroyAllWindows()
```

> 💡 **영상 = 사진의 연속.** 웹캠 프레임을 한 장씩 꺼내 AI에 넣고, 결과를 다시 화면에 그리는 것을 빠르게 반복하는 것이다.

### 4-2. 작업별로 바뀌는 부분

| 작업 | 바뀌는 코드 |
|---|---|
| 탐지 | `YOLO("yolo26n.pt")` + `model.predict(frame)` |
| 분할 | `YOLO("yolo26n-seg.pt")` ← 모델 이름만 `-seg`로 교체 |
| 추적 | `model.track(frame, persist=True)` ← `predict` 대신 `track` |

- `persist=True` : 이전 프레임의 추적 정보를 기억해서 **같은 물체에 같은 ID**를 계속 붙여준다.

### 4-3. 실행해보기

```bash
python webcam_detect.py
python webcam_segment.py
python webcam_track.py
```

**옵션 사용 예시**

```bash
python webcam_detect.py --model yolo26s.pt --conf 0.5   # 더 큰 모델, 신뢰도 50% 이상만
python webcam_track.py --tracker botsort.yaml           # 다른 추적 알고리즘
```

| 옵션 | 의미 |
|---|---|
| `--model` | 모델 크기: `n`(가장 빠름) < `s` < `m` < `l` < `x`(가장 정확) |
| `--source` | 웹캠 번호 (외장 웹캠이면 `1`) |
| `--conf` | 신뢰도 기준 (0~1, 높을수록 확실한 것만 표시) |
| `--imgsz` | 분석 이미지 크기 (작을수록 빠름) |
| `--device` | `cpu` 또는 `0` (GPU) |

**단축키**

| 키 | 동작 |
|---|---|
| `q` / `ESC` | 종료 |
| `m` | 마스크만 보기 (분할) |
| `c` | 이동 궤적 지우기 (추적) |

---

## 5. GitHub에 코드 올리기

### 5-1. GitHub 로그인 (최초 1회)

```bash
gh auth login
```

안내에 따라 브라우저에서 로그인하면 터미널에서 GitHub를 사용할 수 있다.

### 5-2. 저장소 만들고 올리기 (최초 1회)

```bash
git init -b main
git add .
git commit -m "첫 커밋"
gh repo create vision --public --source . --push
```

### 5-3. 코드 수정 후 다시 올리기

```bash
git add .
git commit -m "무엇을 바꿨는지 적기"
git push
```

> 💡 `.gitignore` 파일에 `*.pt`를 적어 두어 **용량 큰 모델 파일은 올리지 않는다.** (실행하면 자동으로 다시 받으니까!)

---

## 6. Claude Code 활용 팁

| 명령 | 기능 |
|---|---|
| `/usage` | 지금까지 사용한 토큰(사용량) 확인 |
| "실행해줘" | 패키지 설치부터 코드 실행까지 대신 해줌 |
| "github에 올려줘" | 저장소 생성 + 커밋 + 푸시 |

---

## ✅ 오늘의 정리

- **Teachable Machine** → 코딩 없이 이미지·모션 AI 학습 (분류)
- **Ultralytics YOLO26** → 분류 · 탐지 · 분할 · 추적을 몇 줄 코드로
- 탐지 = **박스**, 분할 = **누끼(윤곽)**, 추적 = **박스 + ID**
- 실행은 `python 파일.py`, 업로드는 `git add → commit → push`

## 📝 복습 퀴즈

1. 사진 속 물체의 **위치**까지 알고 싶다면 분류와 탐지 중 무엇을 써야 할까?
2. 배경을 지우고 사람만 오려내려면(누끼) 어떤 작업이 적합할까?
3. 탐지 코드를 추적 코드로 바꾸려면 어떤 함수로 바꿔야 할까?
4. 모델 파일(`.pt`)을 GitHub에 올리지 않는 이유는?

<details>
<summary>정답 보기</summary>

1. 탐지 (Object Detection)
2. 분할 (Segmentation)
3. `model.predict()` → `model.track(..., persist=True)`
4. 용량이 크고, 실행할 때 자동으로 다운로드되기 때문
</details>
