import cv2
import sys
import os


# 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

video_path = os.path.join(BASE_DIR, "input.mp4")
output_path = os.path.join(BASE_DIR, "output_mosaic.mp4")

def load_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    if not cap.isOpened(): return None, 0
    while True:
        ret, f = cap.read()
        if not ret: break
        frames.append(f)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return frames, fps

# 1. 영상 로딩 및 리로딩 (기능 100% 유지)
print("영상 데이터를 로딩 중입니다...")
origin_frames, fps = load_frames(video_path)
if fps <= 0: fps = 30

# 기존 작업물 확인 및 로드
if os.path.exists(output_path):
    print(f"기존 작업물({output_path})을 로드하여 이어 편집합니다.")
    processed_frames, _ = load_frames(output_path)
    if not processed_frames or len(processed_frames) != len(origin_frames):
        processed_frames = [f.copy() for f in origin_frames]
else:
    processed_frames = [f.copy() for f in origin_frames]

total_frames = len(origin_frames)
h, w, _ = origin_frames[0].shape

# 상태 변수
current_idx = 0
paused = True
trackers = []
window_name = "100% Sticky-Edge Editor"

def on_trackbar(val):
    global current_idx
    current_idx = val

# [핵심] 100% 모서리 고정(Sticky Edge) 로직
def apply_sticky_mosaic(img, roi_rect):
    x, y, rw, rh = map(int, roi_rect)
    
    # 자석 감지 범위 (15픽셀 이내면 벽에 붙은 것으로 간주)
    margin = 15
    
    # 새로운 좌표 계산용 변수
    nx, ny, nw, nh = x, y, rw, rh
    
    # 좌측 및 상단 벽 고정
    if x < margin:
        nx = 0
        nw = rw + x  # x가 줄어든 만큼 폭을 늘려 오른쪽 끝점 유지
    
    if y < margin:
        ny = 0
        nh = rh + y  # y가 줄어든 만큼 높이를 늘려 아래쪽 끝점 유지
        
    # 우측 및 하단 벽 고정
    if (nx + nw) > (w - margin):
        nw = w - nx
        
    if (ny + nh) > (h - margin):
        nh = h - ny

    # 최종 안전 클리핑 (이미지 범위를 절대 벗어나지 않음)
    nx, ny = max(0, nx), max(0, ny)
    nw, nh = min(nw, w - nx), min(nh, h - ny)
    
    if nw <= 2 or nh <= 2: return img
    
    # 모자이크 처리
    roi_zone = img[ny:ny+nh, nx:nx+nw]
    level = 25
    small = cv2.resize(roi_zone, (max(1, nw//level), max(1, nh//level)))
    img[ny:ny+nh, nx:nx+nw] = cv2.resize(small, (nw, nh), interpolation=cv2.INTER_NEAREST)
    return img

# GUI 및 타임라인 설정 (100% 유지)
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.createTrackbar("Timeline", window_name, 0, total_frames - 1, on_trackbar)

while True:
    # 현재 타임라인의 작업된 프레임 표시
    display_frame = processed_frames[current_idx].copy()
    
    # 재생 및 실시간 누적 편집 (100% 유지)
    if not paused and trackers:
        for t in trackers:
            ok, roi = t.update(origin_frames[current_idx])
            if ok:
                # 고정 로직이 적용된 모자이크를 작업본에 누적 기록
                processed_frames[current_idx] = apply_sticky_mosaic(processed_frames[current_idx], roi)
        
        display_frame = processed_frames[current_idx]
        if current_idx < total_frames - 1:
            current_idx += 1
        else:
            paused = True

    # 정보 표시
    color = (0, 0, 255) if not paused else (0, 255, 0)
    status = "● RECORDING" if not paused else "■ PAUSED"
    cv2.putText(display_frame, f"{status} | Frame: {current_idx}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow(window_name, display_frame)
    cv2.setTrackbarPos("Timeline", window_name, current_idx)

    # 키 조작 인터페이스 (100% 유지)
    key = cv2.waitKey(20) & 0xFF
    if key == 27: # ESC: 저장 및 종료
        break
    elif key == ord(' '): # Space: 재생/일시정지
        paused = not paused
    elif key == ord('s'): # S: ROI 지정 (모서리 고정 시작)
        new_rois = cv2.selectROIs("Select Target", processed_frames[current_idx], False, False)
        if len(new_rois) > 0:
            trackers = []
            for r in new_rois:
                tr = cv2.TrackerCSRT_create()
                tr.init(origin_frames[current_idx], tuple(r))
                trackers.append(tr)
            paused = False   # 
        cv2.destroyWindow("Select Target")
    elif key == ord('a'): # A: 뒤로 1프레임
        current_idx = max(0, current_idx - 1)
        paused = True
    elif key == ord('d'): # D: 앞으로 1프레임
        current_idx = min(total_frames - 1, current_idx + 1)
        paused = True

# 최종 파일 업데이트 (100% 유지)
print("\n[저장] 작업 내용을 파일에 기록 중...")
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
for f in processed_frames:
    out.write(f)
out.release()

cv2.destroyAllWindows()
print(f"저장이 완료되었습니다: {output_path}")
