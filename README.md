# 모자이크 편집기 (FAST MOSAIC EDITOR)

영상 속 사람이나 물체에 모자이크를 넣는 프로그램입니다.
마우스로 영역(ROI)을 지정하면 사람을 자동으로 따라가며 모자이크하고, 결과를 새 영상으로 저장합니다.

## 실행

- **exe**: `dist\mosaic.exe` 더블클릭 (빌드 방법은 아래 "exe 빌드" 참고)
- **소스**: `py -3.11 mosaic.py` 또는 `py -3.11 mosaic.py 영상경로.mp4`

실행하면 영상 파일 선택 창이 뜹니다. 선택하지 않고 닫으면 같은 폴더의 `input.mp4`를 엽니다.
시작 화면에 사용 방법이 나오며, 아무 키나 누르면 편집을 시작합니다. (언제든 **H** 키로 다시 볼 수 있습니다.)

## 화면 표시

| 표시 | 의미 |
|---|---|
| 초록 테두리 | 추적 모드 ROI (사람을 따라감) |
| 빨간 테두리 | 고정 모드 ROI (제자리) |
| 노란 테두리 + 모서리 점 | 선택된 ROI |
| 분홍 테두리 + "대상 놓침" | 추적 실패 → 드래그해서 위치를 바로잡아 주세요 |
| 상단 정보 줄 | 현재 프레임 / 모드 / 재생 상태 / ROI 개수 |
| 상단 FRAME 바 | 원하는 위치로 이동 |

## 재생 / 이동

| 키 | 동작 |
|---|---|
| Space / A | 재생 · 일시정지 |
| ← / Z | 1프레임 뒤로 |
| → / C | 1프레임 앞으로 |
| ↓ | 10프레임 뒤로 |
| ↑ | 10프레임 앞으로 |
| Home / End | 처음 / 끝으로 |
| FRAME 바 드래그 | 원하는 위치로 이동 |

앞으로 이동하면(키, FRAME 바, 재생 모두) 그 사이 프레임도 자동으로 추적됩니다. 많이 건너뛰면 진행 막대가 표시됩니다.

## ROI 설정

| 조작 | 동작 |
|---|---|
| 빈 곳 드래그 | 새 ROI 만들기 |
| F | 추적 모드 ↔ 고정 모드 전환 (새로 그리는 ROI에 적용) |
| ROI 클릭 | 선택 |
| ROI 드래그 | 위치 이동 |
| 모서리 드래그 | 크기 조절 |
| B | 선택한 ROI를 과거 방향으로 역추적 |

- **추적 모드**: 그린 프레임부터 대상을 따라 움직입니다.
- **고정 모드**: 그린 위치에 계속 고정됩니다 (간판, 번호판 등).
- 추적 ROI를 이동하거나 크기를 바꾸면 **그 프레임부터 새 위치로 다시 추적**합니다. 추적이 틀어졌을 때 이렇게 바로잡으면 됩니다.
- 재생 중에 클릭하면 자동으로 일시정지됩니다.

## 삭제 / 되돌리기

| 조작 | 동작 |
|---|---|
| 우클릭 (ROI 위) | 이 프레임에서 ROI 끝내기 — 다음 프레임부터 모자이크 해제 |
| Delete / D | 선택한 ROI 전체 삭제 (모든 프레임) |
| X | 모든 ROI 삭제 |
| Ctrl+Z / U | 되돌리기 (최대 50단계) |

**ROI 하나 삭제**: ROI를 클릭해 선택(노란 테두리) → Delete 또는 D
**중간부터 모자이크 끄기**: 끝낼 프레임으로 이동 → 그 ROI 위에서 우클릭

## 테두리에서 들어오는 사람

1. 사람이 **완전히 화면에 들어온 프레임**으로 이동합니다.
2. 추적 모드로 그 사람에게 ROI를 그립니다.
3. **B** 키를 누릅니다.

과거 방향으로 추적하면서 사람이 테두리 밖으로 나가는 시점까지 따라갑니다.
결과적으로 재생하면 **테두리에서 조금씩 커지는 모자이크**가 되고, 사람이 다 들어온 뒤로는 **함께 이동**합니다.
화면 밖으로 나가는 경우는 별도 조작 없이 자동으로 따라 나가며 줄어듭니다. (좌/우 테두리 기준)

- 역추적은 최대 10초 분량까지 진행하며, Esc로 중단할 수 있습니다.
- 다른 사람에게 가려지면 추적이 빗나갈 수 있습니다. 해당 프레임에서 ROI를 드래그해 바로잡으세요.

## 저장 / 종료

| 키 | 동작 |
|---|---|
| S | 저장 (Esc로 취소) |
| H | 도움말 |
| Q / Esc | 종료 (저장하지 않은 변경이 있으면 한 번 더 눌러야 종료) |

- 결과 파일: 원본과 같은 폴더에 `원본이름_mosaic.mp4`
- 저장하면서 아직 추적하지 않은 구간도 이어서 추적합니다.
- **소리는 저장되지 않습니다.**

## 참고

- 한글 입력 상태에서도 단축키가 동작합니다 (예: D = ㅇ, X = ㅌ, U = ㅕ).
- 모자이크 칸 크기, 역추적 최대 길이 등은 `mosaic.py` 상단의 설정값으로 바꿀 수 있습니다.

## exe 빌드

```
py -3.11 -m PyInstaller --onefile --noconsole --collect-all cv2 mosaic.py
```

필요 패키지: `opencv-contrib-python`, `numpy`, `pillow`, `pyinstaller`

---

# 개발 기록
# mosaic.py_9
python 3.10.11/vscode
C:\Users\owner\Desktop\v_mosaic\input.mp4
 
# mosaic.py_10
python.3.11.
C:\Users\owner\Desktop\v_mosaic>pip list
Package                   Version
 
albucore                  0.0.24
albumentations            2.0.8
altair                    5.5.0
altgraph                  0.17.5
annotated-types           0.7.0
attrs                     25.4.0
blinker                   1.9.0
cachetools                6.2.1
certifi                   2025.10.5
cffi                      2.0.0
charset-normalizer        3.4.4
click                     8.3.0
colorama                  0.4.6
contourpy                 1.3.3
cycler                    0.12.1
Cython                    3.2.4
easydict                  1.13
filelock                  3.20.0
Flask                     3.1.2
flatbuffers               25.12.19
fonttools                 4.61.1
fsspec                    2025.10.0
gitdb                     4.0.12
GitPython                 3.1.45
idna                      3.11
ImageIO                   2.37.2
insightface               0.7.3
itsdangerous              2.2.0
Jinja2                    3.1.6
joblib                    1.5.3
jsonschema                4.25.1
jsonschema-specifications 2025.9.1
kiwisolver                1.4.9
lazy_loader               0.4
llvmlite                  0.45.1
MarkupSafe                3.0.3
matplotlib                3.10.8
ml_dtypes                 0.5.4
more-itertools            10.8.0
mpmath                    1.3.0
narwhals                  2.10.2
networkx                  3.5
numba                     0.62.1
numpy                     1.26.4
onnx                      1.20.1
onnxruntime               1.24.1
openai-whisper            20250625
opencv-contrib-python     4.8.1.78
packaging                 25.0
pandas                    2.3.3
pefile                    2024.8.26
pillow                    12.0.0
pip                       25.1.1
prettytable               3.17.0
protobuf                  6.33.0
pyarrow                   21.0.0
pycparser                 2.23
pydantic                  2.12.5
pydantic_core             2.41.5
pydeck                    0.9.1
pyinstaller               6.18.0
pyinstaller-hooks-contrib 2026.0
pyparsing                 3.3.2
python-dateutil           2.9.0.post0
pytz                      2025.2
pywin32-ctypes            0.2.3
PyYAML                    6.0.3
referencing               0.37.0
regex                     2025.11.3
requests                  2.32.5
rpds-py                   0.28.0
scikit-image              0.26.0
scikit-learn              1.8.0
scipy                     1.17.0
setuptools                80.9.0
simsimd                   6.5.12
six                       1.17.0
smmap                     5.0.2
sounddevice               0.5.3
streamlit                 1.51.0
stringzilla               4.6.0
sympy                     1.14.0
tenacity                  9.1.2
threadpoolctl             3.6.0
tifffile                  2026.1.28
tiktoken                  0.12.0
toml                      0.10.2
torch                     2.9.0
tornado                   6.5.2
tqdm                      4.67.1
typing_extensions         4.15.0
typing-inspection         0.4.2
tzdata                    2025.2
urllib3                   2.5.0
watchdog                  6.0.0
wcwidth                   0.6.0
Werkzeug                  3.1.5

# mosaic.py 13 pyinstaller
pyinstaller --onefile --noconsole --collect-all cv2 --hidden-import=cv2.legacy main.py
dist 폴더 안에 input.mp4 파일을 넣어야 작동된다


#TRACK MODE 와 FIXED MODE 정상작동하고 있다/ 즉 무 MODE 모두다 ROI 설정후 20프레임에서 50프레임까지 A를 눌러 모자이크 처리한 후 50프레임에서 ROI 를 제거해도 기존에 모자이크 처리된 20프레임에서 50프레임까지 모자이크 처리는 그대로 남아 있다/그러나 Pyinstaller 로 생성된 exe 를 작동시켰을때 fixed mode는 roi 를 지우면 그때까지 모자이크 처리된 프레임의 모자이크도 함께 사라진다/이런 현상을 방지하기 위하여 어떻게 해야하는지 알려주세요
-문제의 핵심 원인을 먼저 정확히 설명드리겠습니다.
현재 코드 구조를 보면

화면 표시(redraw_current) → trackers 기반
영상 저장(save_video) → trackers 기반
과거 기록 → mosaic_history 일부만 사용

즉 실제 모자이크는 trackers에 의존합니다.

그래서 Python 실행에서는 우연히 정상처럼 보이지만
PyInstaller exe에서는 ROI 삭제 시 trackers가 비활성화(end 설정) 되면서

20 ~ 50 프레임 모자이크
→ trackers가 사라짐
→ redraw_current가 더 이상 모자이크 안함
→ 과거 모자이크도 사라짐

이 문제가 발생합니다.
-해결 원리 (중요)

모자이크는 반드시 mosaic_history만 사용해서 그려야 합니다.

즉 구조를 이렇게 바꿔야 합니다.

trackers → ROI 추적용
mosaic_history → 실제 모자이크 기록
redraw_current → mosaic_history만 사용
save_video → mosaic_history만 사용

그러면

ROI 삭제
→ trackers만 종료
→ mosaic_history는 그대로
→ 과거 모자이크 절대 안사라짐

Python / EXE 동일 동작합니다./ 아래 수정한 부분

-def redraw_current():

    frame = get_frame(current_idx)

    if frame is None:
        return None

    if current_idx in mosaic_history:

        for roi in mosaic_history[current_idx]:
            frame = apply_mosaic(frame, roi)

    return frame

    -def save_video():

    print("💾 저장 시작")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    cap2 = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)

    for f in range(total_frames):

        ret, frame = cap2.read()

        if not ret:
            break

        if f in mosaic_history:

            for roi in mosaic_history[f]:
                frame = apply_mosaic(frame, roi)

        out.write(frame)

        if f % 200 == 0:
            print("저장 진행:", f)

    out.release()
    cap2.release()

    print("저장 완료")
    - 결과 정상 작동

 
