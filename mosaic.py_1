import cv2
import sys
import collections
import os
import multiprocessing

multiprocessing.freeze_support()

os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

INPUT_VIDEO = "input.mp4"
OUTPUT_VIDEO = "output_mosaic.mp4"

video_path = INPUT_VIDEO
print("작업 기준 영상:", video_path)

cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)

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

    sub = frame[y:y + h, x:x + w]

    scale = 20

    small = cv2.resize(sub, (max(1, w // scale), max(1, h // scale)))
    mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    frame[y:y + h, x:x + w] = mosaic

    return frame


def record_history(frame_idx, roi):
    rois = mosaic_history.setdefault(frame_idx, [])

    r = tuple(map(int, roi))

    if r not in rois:
        rois.append(r)


def is_tracker_active_at(t, frame_idx):
    if frame_idx < t["start"]:
        return False

    end = t.get("end", None)
    if end is not None and frame_idx > end:
        return False

    return True


def update_trackers():

    if paused and current_idx == 0:
        return

    frame = get_frame(current_idx)

    if frame is None:
        return

    for t in trackers:

        if not is_tracker_active_at(t, current_idx):
            continue

        if t["mode"] == "track":

            idx = current_idx - t["start"]

            if idx >= len(t["history"]):

                success, box = t["tracker"].update(frame)

                if success:
                    t["roi"] = tuple(map(int, box))

                t["history"].append(tuple(map(int, t["roi"])))

            else:
                t["roi"] = tuple(map(int, t["history"][idx]))

            record_history(current_idx, t["roi"])

        else:
            record_history(current_idx, t["roi"])


def redraw_current():

    frame = get_frame(current_idx)

    if frame is None:
        return None

    if current_idx in mosaic_history:

        for roi in mosaic_history[current_idx]:
            frame = apply_mosaic(frame, roi)

    return frame


def save_video():

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

def on_trackbar(pos):

    global current_idx, paused

    paused = True

    target = min(pos, total_frames - 1)

    if target > current_idx:

        original_idx = current_idx

        for f in range(original_idx + 1, target + 1):
            current_idx = f
            update_trackers()

    current_idx = target


def on_trackbar2(pos):

    global current_idx, paused

    paused = True

    current_idx = min(pos, total_frames - 1)


cv2.createTrackbar(TRACKBAR, WINDOW, 0, total_frames - 1, on_trackbar)
cv2.createTrackbar(TRACKBAR2, WINDOW, 0, total_frames - 1, on_trackbar2)


def find_roi(x, y):

    for i in reversed(range(len(trackers))):

        t = trackers[i]

        if not is_tracker_active_at(t, current_idx):
            continue

        rx, ry, rw, rh = map(int, t["roi"])

        if rx <= x <= rx + rw and ry <= y <= ry + rh:
            return i

    return None


def mouse_callback(event, x, y, flags, param):

    global drawing, start_x, start_y
    global moving_roi_index
    global move_offset_x, move_offset_y
    global selected_roi
    global fixed_mode

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
        selected_roi = None

    elif event == cv2.EVENT_MOUSEMOVE and moving_roi_index is not None:

        t = trackers[moving_roi_index]

        if not is_tracker_active_at(t, current_idx):
            return

        rx, ry, rw, rh = t["roi"]

        new_roi = (
            x - move_offset_x,
            y - move_offset_y,
            rw,
            rh,
        )

        t["roi"] = new_roi

        if t["mode"] == "track":
            idx = current_idx - t["start"]
            if idx >= 0:
                if idx < len(t["history"]):
                    t["history"][idx] = tuple(map(int, new_roi))
                elif idx == len(t["history"]):
                    t["history"].append(tuple(map(int, new_roi)))

    elif event == cv2.EVENT_LBUTTONUP:

        if drawing:

            drawing = False

            rx = min(start_x, x)
            ry = min(start_y, y)
            rw = abs(x - start_x)
            rh = abs(y - start_y)

            if rw > 10 and rh > 10:

                if fixed_mode:

                    trackers.append(
                        {
                            "roi": (rx, ry, rw, rh),
                            "start": current_idx,
                            "end": None,
                            "mode": "fixed",
                        }
                    )

                    record_history(current_idx, (rx, ry, rw, rh))

                else:

                    tracker = cv2.legacy.TrackerKCF_create()

                    frame = get_frame(current_idx)

                    if frame is not None:
                        tracker.init(frame, (rx, ry, rw, rh))

                        trackers.append(
                            {
                                "roi": (rx, ry, rw, rh),
                                "start": current_idx,
                                "end": None,
                                "tracker": tracker,
                                "history": [(rx, ry, rw, rh)],
                                "mode": "track",
                            }
                        )

        moving_roi_index = None

    elif event == cv2.EVENT_RBUTTONDOWN:

        idx = find_roi(x, y)

        if idx is not None:

            t = trackers[idx]

            if t["start"] <= current_idx:

                if t["mode"] == "track":

                    end_frame = current_idx

                    if len(t["history"]) > 0:
                        for f in range(t["start"], end_frame + 1):
                            hist_idx = min(f - t["start"], len(t["history"]) - 1)
                            roi = t["history"][hist_idx]
                            record_history(f, roi)

                else:

                    roi = tuple(map(int, t["roi"]))

                    end_frame = max(current_idx, t["start"])

                    for f in range(t["start"], end_frame + 1):
                        record_history(f, roi)

            t["end"] = current_idx

            if selected_roi == idx:
                selected_roi = None

            if moving_roi_index == idx:
                moving_roi_index = None


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
        (0, 0, 255) if fixed_mode else (0, 255, 0),
        2,
    )

    if paused:

        for i, t in enumerate(trackers):

            if not is_tracker_active_at(t, current_idx):
                continue

            x, y, w, h = map(int, t["roi"])

            color = (0, 255, 0) if t["mode"] == "track" else (0, 0, 255)

            if i == selected_roi:
                color = (255, 255, 0)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    cv2.imshow(WINDOW, frame)

    try:
        cv2.setTrackbarPos(TRACKBAR, WINDOW, int(current_idx))
        cv2.setTrackbarPos(TRACKBAR2, WINDOW, int(current_idx))
    except:
        pass

    key = cv2.waitKeyEx(1)

    if key in [ord("q"), ord("Q")]:
        break

    elif key in [ord("a"), ord("A")]:
        paused = not paused

    elif key in [ord("f"), ord("F")]:
        fixed_mode = not fixed_mode

    elif key in [ord("s"), ord("S")]:
        save_video()

    elif key in [ord("x"), ord("X")]:
        trackers.clear()
        selected_roi = None
        moving_roi_index = None

    elif key == ord("z") or key == 2424832:
        current_idx = max(0, current_idx - 1)

    elif key == 2555904:
        current_idx = min(total_frames - 1, current_idx + 1)

    elif key == 3014656:
        current_idx = min(total_frames - 1, current_idx + 10)

    elif key == 2162688:
        current_idx = max(0, current_idx - 10)

cv2.destroyAllWindows()

cap.release()

print("프로그램 종료")
