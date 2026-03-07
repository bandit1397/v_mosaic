import cv2
import sys
import collections

INPUT_VIDEO = "input.mp4"
OUTPUT_VIDEO = "output_mosaic.mp4"

video_path = INPUT_VIDEO
print("작업 기준 영상:", video_path)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("영상 파일을 열 수 없습니다.")
    sys.exit()

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("전체 프레임:", total_frames)

CACHE_SIZE = 400
frame_cache = collections.OrderedDict()
last_read_index = -1


def get_frame(idx):

    global last_read_index

    if idx in frame_cache:
        frame_cache.move_to_end(idx)
        return frame_cache[idx].copy()

    if idx == last_read_index + 1:

        ret, frame = cap.read()

        if not ret:
            return None

    else:

        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if not ret:
            return None

    last_read_index = idx

    frame_cache[idx] = frame

    if len(frame_cache) > CACHE_SIZE:
        frame_cache.popitem(last=False)

    return frame.copy()


current_idx = 0
paused = True

drawing = False
moving_roi_index = None
selected_roi = None

move_offset_x = 0
move_offset_y = 0

start_x = 0
start_y = 0

fixed_mode = False

trackers = []

mosaic_history = {}

WINDOW = "FAST MOSAIC EDITOR"
TRACKBAR = "FRAME"
TRACKBAR2 = "JUMP"

cv2.namedWindow(WINDOW)


def apply_mosaic(frame, roi):

    x, y, w, h = map(int, roi)

    x = max(0, x)
    y = max(0, y)

    if w <= 0 or h <= 0:
        return frame

    if x + w > frame.shape[1]:
        w = frame.shape[1] - x

    if y + h > frame.shape[0]:
        h = frame.shape[0] - y

    if w <= 0 or h <= 0:
        return frame

    sub = frame[y:y+h, x:x+w]

    if sub.size == 0:
        return frame

    scale = 20

    small = cv2.resize(sub, (max(1, w//scale), max(1, h//scale)))
    mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    frame[y:y+h, x:x+w] = mosaic

    return frame


def redraw_current():

    frame = get_frame(current_idx)

    if frame is None:
        return None

    if current_idx in mosaic_history:

        for roi in mosaic_history[current_idx]:
            frame = apply_mosaic(frame, roi)

    for t in trackers:

        if t["start"] <= current_idx:

            if t["mode"] == "track":

                idx = current_idx - t["start"]

                if idx < len(t["history"]):
                    roi = t["history"][idx]
                else:
                    roi = t["roi"]

            else:
                roi = t["roi"]

            frame = apply_mosaic(frame, roi)

    return frame


def record_history(frame_idx, roi):

    if frame_idx not in mosaic_history:
        mosaic_history[frame_idx] = []

    mosaic_history[frame_idx].append(roi)


def process_until(target_frame):

    for t in trackers:

        start = t["start"]

        if target_frame <= start:
            continue

        for f in range(start, target_frame + 1):

            if t["mode"] == "fixed":

                record_history(f, t["roi"])
                continue

            idx = f - start

            if idx < len(t["history"]):

                roi = t["history"][idx]

            else:

                frame = get_frame(f)

                if frame is None:
                    break

                success, box = t["tracker"].update(frame)

                if success:
                    t["roi"] = box

                t["history"].append(t["roi"])

                roi = t["roi"]

            record_history(f, roi)


def update_trackers():

    frame = get_frame(current_idx)

    if frame is None:
        return

    for t in trackers:

        if t["start"] <= current_idx:

            if t["mode"] == "track":

                idx = current_idx - t["start"]

                if idx >= len(t["history"]):

                    success, box = t["tracker"].update(frame)

                    if success:
                        t["roi"] = box

                    t["history"].append(t["roi"])

                record_history(current_idx, t["roi"])

            elif t["mode"] == "fixed":

                record_history(current_idx, t["roi"])


def save_video():

    print("💾 저장 시작")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    cap2 = cv2.VideoCapture(video_path)

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

    print("✅ 저장 완료")


def on_trackbar(pos):

    global current_idx
    global paused

    paused = True

    target = min(pos, total_frames - 1)

    if target > current_idx:
        process_until(target)

    current_idx = target


def on_trackbar2(pos):

    global current_idx
    global paused

    paused = True
    current_idx = min(pos, total_frames - 1)


cv2.createTrackbar(TRACKBAR, WINDOW, 0, total_frames - 1, on_trackbar)
cv2.createTrackbar(TRACKBAR2, WINDOW, 0, total_frames - 1, on_trackbar2)


def find_roi(x, y):

    for i in reversed(range(len(trackers))):

        rx, ry, rw, rh = trackers[i]["roi"]

        if rx <= x <= rx+rw and ry <= y <= ry+rh:
            return i

    return None


def mouse_callback(event, x, y, flags, param):

    global drawing
    global start_x
    global start_y
    global moving_roi_index
    global move_offset_x
    global move_offset_y
    global selected_roi

    if not paused:
        return

    if event == cv2.EVENT_LBUTTONDOWN:

        idx = find_roi(x, y)

        if idx is not None:

            selected_roi = idx
            moving_roi_index = idx

            rx, ry, rw, rh = trackers[idx]["roi"]

            move_offset_x = x - rx
            move_offset_y = y - ry

            return

        drawing = True
        start_x = x
        start_y = y

    elif event == cv2.EVENT_MOUSEMOVE:

        if moving_roi_index is not None:

            rx, ry, rw, rh = trackers[moving_roi_index]["roi"]

            trackers[moving_roi_index]["roi"] = (
                x - move_offset_x,
                y - move_offset_y,
                rw,
                rh
            )

    elif event == cv2.EVENT_LBUTTONUP:

        if drawing:

            drawing = False

            rx = min(start_x, x)
            ry = min(start_y, y)
            rw = abs(x - start_x)
            rh = abs(y - start_y)

            if rw > 10 and rh > 10:

                if fixed_mode:

                    trackers.append({
                        "roi": (rx, ry, rw, rh),
                        "start": current_idx,
                        "mode": "fixed"
                    })

                else:

                    tracker = cv2.legacy.TrackerKCF_create()

                    frame = get_frame(current_idx)

                    tracker.init(frame, (rx, ry, rw, rh))

                    trackers.append({
                        "roi": (rx, ry, rw, rh),
                        "start": current_idx,
                        "tracker": tracker,
                        "history": [(rx, ry, rw, rh)],
                        "mode": "track"
                    })

        moving_roi_index = None

    elif event == cv2.EVENT_RBUTTONDOWN:

        idx = find_roi(x, y)

        if idx is not None:
            trackers.pop(idx)


cv2.setMouseCallback(WINDOW, mouse_callback)


while True:

    if not paused:

        current_idx += 1

        if current_idx >= total_frames:
            current_idx = total_frames - 1
            paused = True

        update_trackers()

    frame = redraw_current()

    if frame is None:
        break

    mode_text = "FIXED MODE" if fixed_mode else "TRACK MODE"

    cv2.putText(
        frame,
        mode_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,0,255) if fixed_mode else (0,255,0),
        2
    )

    if paused:

        for i, t in enumerate(trackers):

            if t["start"] <= current_idx:

                x, y, w, h = map(int, t["roi"])

                color = (0,255,0) if t["mode"]=="track" else (0,0,255)

                if i == selected_roi:
                    color = (255,255,0)

                cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)

    cv2.imshow(WINDOW, frame)

    try:
        cv2.setTrackbarPos(TRACKBAR, WINDOW, int(current_idx))
        cv2.setTrackbarPos(TRACKBAR2, WINDOW, int(current_idx))
    except:
        pass

    key = cv2.waitKeyEx(1)

    if key == ord('q') or key == ord('Q'):
        break

    elif key == ord('a') or key == ord('A'):
        paused = not paused

    elif key == ord('f') or key == ord('F'):
        fixed_mode = not fixed_mode

    elif key == ord('s') or key == ord('S'):
        save_video()

    elif key == ord('z') or key == ord('Z'):
        current_idx = max(0, current_idx - 1)

    elif key == 2555904:
        current_idx = min(total_frames-1, current_idx+1)

    elif key == 2424832:
        current_idx = max(0, current_idx-1)

    elif key == 3014656:
        current_idx = min(total_frames-1, current_idx+10)

    elif key == 2162688:
        current_idx = max(0, current_idx-10)

    elif key == ord('x') or key == ord('X'):
        trackers.clear()

cv2.destroyAllWindows()
cap.release()

print("프로그램 종료")
