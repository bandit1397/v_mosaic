import cv2
import sys
import os
import time
import collections
import multiprocessing

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None

os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

DEFAULT_INPUT = "input.mp4"
OUTPUT_SUFFIX = "_mosaic.mp4"

WINDOW = "FAST MOSAIC EDITOR"
TRACKBAR = "FRAME"

MOSAIC_BLOCK = 20          # 모자이크 한 칸 크기(px)
CACHE_SIZE = 120           # 프레임 캐시 (720p 기준 약 330MB)
MIN_ROI = 10               # 이보다 작은 드래그는 무시
HANDLE = 10                # 모서리 크기조절 판정 거리(px)

EDGE_MARGIN = 3            # 박스가 테두리에서 이 거리 이내면 "닿았다"고 판단
MIN_EXIT_SPEED = 1.0       # 바깥쪽으로 이 속도(px/frame) 이상이면 화면 밖으로 나가는 중
VEL_WINDOW = 6             # 속도 추정에 쓰는 최근 박스 수
MAX_COAST_SEC = 3          # 화면 밖으로 나가는 중 추적 없이 이동시키는 최대 시간
BACKTRACK_MAX_SEC = 10     # 역추적 최대 길이
BACK_CHUNK = 60            # 역추적 시 한 번에 읽는 프레임 수
UNDO_LIMIT = 50
TRACK_MAX_SIDE = 640       # 추적은 이 크기로 줄인 영상에서 (속도)

FONT_PATHS = [
    "C:/Windows/Fonts/malgun.ttf",
    "C:/Windows/Fonts/gulim.ttc",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
]

# waitKeyEx 특수키 코드 (Windows)
KEY_LEFT, KEY_UP, KEY_RIGHT, KEY_DOWN = 2424832, 2490368, 2555904, 2621440
KEY_HOME, KEY_END, KEY_DELETE = 2359296, 2293760, 3014656
KEY_ESC, KEY_SPACE, KEY_CTRL_Z = 27, 32, 26

# 한글 입력 상태에서 눌러도 동작하도록
HANGUL_KEYS = {
    "ㅁ": "a", "ㄹ": "f", "ㄴ": "s", "ㅂ": "q", "ㅌ": "x", "ㅋ": "z",
    "ㅊ": "c", "ㅇ": "d", "ㅠ": "b", "ㅕ": "u", "ㅗ": "h",
}

HELP_LINES = [
    ("", "모자이크 편집기 - 사용 방법"),
    None,
    ("", "[재생 / 이동]"),
    ("Space / A", "재생 · 일시정지"),
    ("← → (Z / C)", "1프레임 뒤로 / 앞으로"),
    ("↓ ↑", "10프레임 뒤로 / 앞으로"),
    ("Home / End", "처음 / 끝으로"),
    ("FRAME 바", "원하는 위치로 이동 (앞으로 이동하면 그 사이도 자동 추적)"),
    None,
    ("", "[ROI 설정]"),
    ("빈 곳 드래그", "새 ROI 만들기"),
    ("F", "추적 모드(사람을 따라감) ↔ 고정 모드(제자리) 전환"),
    ("ROI 드래그", "위치 이동 - 이 프레임부터 다시 추적"),
    ("모서리 드래그", "크기 조절 - 이 프레임부터 다시 추적"),
    ("B", "선택한 ROI를 과거 방향으로 역추적"),
    None,
    ("", "[삭제 / 되돌리기]"),
    ("우클릭", "이 프레임에서 ROI 끝내기 (이후 모자이크 해제)"),
    ("Delete / D", "선택한 ROI 전체 삭제"),
    ("Ctrl+Z / U", "되돌리기"),
    ("X", "모든 ROI 삭제"),
    None,
    ("", "[테두리에서 들어오는 사람]"),
    ("", "사람이 완전히 보이는 프레임에서 ROI를 그린 뒤 B를 누르세요."),
    ("", "등장 구간은 화면 밖 부분이 잘린 채 점점 커지고, 이후로는 사람을 따라갑니다."),
    ("", "화면 밖으로 나갈 때는 자동으로 따라 나가며 줄어듭니다."),
    None,
    ("S", "저장 (원본이름_mosaic.mp4, 소리 제외)"),
    ("H", "이 도움말     Q / Esc  종료"),
    None,
    ("", "아무 키나 누르면 시작합니다"),
]


# ---------------------------------------------------------------- 글자 출력

_fonts = {}


def get_font(size):
    if Image is None:
        return None
    if size not in _fonts:
        font = None
        for p in FONT_PATHS:
            try:
                font = ImageFont.truetype(p, size)
                break
            except OSError:
                pass
        _fonts[size] = font
    return _fonts[size]


def draw_text(img, text, org, size=20, color=(255, 255, 255), bg=False):
    """한글 출력 (PIL). 글자가 들어가는 영역만 변환해서 빠르게 처리."""
    x, y = org
    font = get_font(size)

    if font is None:
        cv2.putText(img, text, (x, y + size), cv2.FONT_HERSHEY_SIMPLEX,
                    size / 32, color, 1, cv2.LINE_AA)
        return

    _, _, r, b = font.getbbox(text)
    pad = 4
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1 = min(img.shape[1], x + r + pad)
    y1 = min(img.shape[0], y + b + pad)

    if x1 <= x0 or y1 <= y0:
        return

    if bg:
        img[y0:y1, x0:x1] = (img[y0:y1, x0:x1] * 0.35).astype(np.uint8)

    sub = Image.fromarray(cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(sub).text((x - x0, y - y0), text, font=font, fill=color[::-1])
    img[y0:y1, x0:x1] = cv2.cvtColor(np.asarray(sub), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------- 박스 계산

def clip_box(box, W, H):
    """화면 안에 보이는 부분만 정수 박스로. 완전히 밖이면 None."""
    x, y, w, h = box
    x0, y0 = max(0, int(round(x))), max(0, int(round(y)))
    x1, y1 = min(W, int(round(x + w))), min(H, int(round(y + h)))

    if x1 - x0 <= 0 or y1 - y0 <= 0:
        return None

    return (x0, y0, x1 - x0, y1 - y0)


def shift_box(box, vel):
    return (box[0] + vel[0], box[1] + vel[1], box[2], box[3])


def outward_speed(box, vel, W, H):
    """좌/우 테두리에 닿은 채 바깥쪽으로 움직이는 속도. 0이면 나가는 중이 아님.
    위/아래는 제외: 카메라 쪽으로 걸어오는 사람은 아래 테두리에 닿은 채 커지기만 함."""
    x, y, w, h = box
    vx = vel[0]

    if x <= EDGE_MARGIN:
        return max(0.0, -vx)
    if x + w >= W - EDGE_MARGIN:
        return max(0.0, vx)

    return 0.0


def apply_mosaic(frame, box):
    x, y, w, h = box
    sub = frame[y:y + h, x:x + w]
    small = cv2.resize(sub, (max(1, w // MOSAIC_BLOCK), max(1, h // MOSAIC_BLOCK)))
    frame[y:y + h, x:x + w] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def create_tracker():
    for name in ("TrackerCSRT_create", "TrackerKCF_create"):
        if hasattr(cv2, name):
            return getattr(cv2, name)()
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, name):
            return getattr(cv2.legacy, name)()
    raise RuntimeError("OpenCV 추적기를 찾을 수 없습니다 (opencv-contrib-python 필요)")


# ---------------------------------------------------------------- 추적

class TrackState:
    """한 방향(정방향 또는 역방향)으로 진행 중인 추적 상태."""

    def __init__(self, recent_boxes, W, H, fps):
        self.tracker = None
        self.coast = None           # 화면 밖으로 나가는 중일 때 고정 속도
        self.coast_frames = 0
        self.lost = False
        self.recent = collections.deque(recent_boxes, maxlen=VEL_WINDOW)
        self.W, self.H = W, H
        self.max_coast = int(fps * MAX_COAST_SEC)
        self.scale = min(1.0, TRACK_MAX_SIDE / max(W, H))

    def small(self, frame):
        if self.scale == 1.0:
            return frame
        return cv2.resize(frame, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)

    def velocity(self):
        if len(self.recent) < 2:
            return (0.0, 0.0)

        a, b = self.recent[0], self.recent[-1]
        n = len(self.recent) - 1

        return (
            ((b[0] + b[2] / 2) - (a[0] + a[2] / 2)) / n,
            ((b[1] + b[3] / 2) - (a[1] + a[3] / 2)) / n,
        )

    def step(self, prev_frame, frame, prev_box):
        """다음 프레임 박스를 반환. 화면 밖으로 완전히 나가면 None."""
        W, H = self.W, self.H

        if self.coast is None and self.tracker is None:
            vis = clip_box(prev_box, W, H)

            if vis is None or vis[2] < 8 or vis[3] < 8:
                self.coast = self.velocity()
            else:
                k = self.scale
                self.tracker = create_tracker()
                self.tracker.init(self.small(prev_frame), tuple(int(round(v * k)) for v in vis))

        if self.coast is not None:
            # 화면 밖으로 나가는 중: 보이는 부분이 적어 추적이 불가능하므로
            # 직전 속도로 계속 이동시키고, 모자이크는 화면 안쪽만 잘려서 적용됨
            self.coast_frames += 1
            if self.coast_frames > self.max_coast:
                return None
            box = shift_box(prev_box, self.coast)

        else:
            ok, b = self.tracker.update(self.small(frame))
            vel = self.velocity()

            if ok:
                box = tuple(float(v) / self.scale for v in b)
                self.lost = False

                if outward_speed(box, vel, W, H) > MIN_EXIT_SPEED:
                    self.coast = vel
                    box = shift_box(prev_box, vel)

            elif outward_speed(prev_box, vel, W, H) > MIN_EXIT_SPEED:
                self.coast = vel
                box = shift_box(prev_box, vel)

            else:
                self.lost = True
                box = prev_box

        if clip_box(box, W, H) is None:
            return None

        self.recent.append(box)
        return box


class Roi:

    def __init__(self, mode, start, box):
        self.mode = mode            # "track" | "fixed"
        self.start = start
        self.end = None             # 사용자가 끝낸 프레임(포함)
        self.exited_at = None       # 추적 중 화면 밖으로 나간 프레임(포함)
        self.boxes = {start: box}   # fixed 는 start 하나만 사용
        self.last = start           # 추적된 마지막 프레임
        self.state = None           # 정방향 TrackState (되돌리기 후엔 다시 생성)

    def stop_frame(self):
        stops = [s for s in (self.end, self.exited_at) if s is not None]
        return min(stops) if stops else None

    def box_at(self, f):
        if f < self.start:
            return None

        stop = self.stop_frame()
        if stop is not None and f > stop:
            return None

        if self.mode == "fixed":
            return self.boxes[self.start]

        return self.boxes.get(f)

    def needs_tracking(self, f):
        return (
            self.mode == "track"
            and self.exited_at is None
            and self.last < f
            and (self.end is None or self.last < self.end)
        )

    def set_box(self, f, box):
        """사용자가 f 프레임에서 박스를 고침 → 이후는 새 위치로 다시 추적."""
        if self.mode == "fixed":
            self.boxes[self.start] = box
            return

        for k in [k for k in self.boxes if k > f]:
            del self.boxes[k]

        self.boxes[f] = box
        self.last = f
        self.exited_at = None
        self.state = None

    def cut_after(self, f):
        self.end = f

        if self.mode == "track":
            for k in [k for k in self.boxes if k > f]:
                del self.boxes[k]
            self.last = min(self.last, f)

    def snapshot(self):
        return (self.mode, self.start, self.end, self.exited_at, dict(self.boxes), self.last)

    @classmethod
    def from_snapshot(cls, snap):
        mode, start, end, exited_at, boxes, last = snap
        r = cls(mode, start, boxes[start])
        r.end, r.exited_at, r.boxes, r.last = end, exited_at, dict(boxes), last
        return r


# ---------------------------------------------------------------- 영상

class Video:

    def __init__(self, path):
        self.path = path
        self.cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)

        if not self.cap.isOpened():
            raise RuntimeError("영상 파일을 열 수 없습니다:\n" + path)

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            self.fps = 30

        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.cache = collections.OrderedDict()
        self.last_read = -1

    def get(self, idx):
        """프레임 반환 (캐시 원본이므로 그리기 전에 copy 할 것)."""
        if idx < 0 or idx >= self.total:
            return None

        if idx in self.cache:
            self.cache.move_to_end(idx)
            return self.cache[idx]

        if idx != self.last_read + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

        ret, frame = self.cap.read()

        if not ret:
            self.last_read = -1
            return None

        self.last_read = idx
        self.cache[idx] = frame

        if len(self.cache) > CACHE_SIZE:
            self.cache.popitem(last=False)

        return frame

    def release(self):
        self.cap.release()


# ---------------------------------------------------------------- 편집기

class Editor:

    def __init__(self, path):
        self.video = Video(path)
        self.W, self.H = self.video.width, self.video.height
        self.fps = self.video.fps

        base, _ = os.path.splitext(path)
        self.output_path = base + OUTPUT_SUFFIX

        self.cur = 0
        self.playing = False
        self.fixed_mode = False

        self.rois = []
        self.selected = None
        self.undo_stack = []
        self.dirty = False
        self.quit_armed = False

        # 마우스 상태
        self.drag = None            # "draw" | "move" | "resize"
        self.drag_start = (0, 0)
        self.drag_rect = None
        self.drag_offset = (0, 0)
        self.drag_anchor = (0, 0)
        self.drag_snapshot = None
        self.modal = False

        self.seek_request = None
        self.trackbar_pos = 0

        self.message = ""
        self.message_until = 0

    # ---------------------------------------------- 상태 / 되돌리기

    def notify(self, text, sec=2.5):
        self.message = text
        self.message_until = time.time() + sec

    def snapshot(self):
        sel = self.rois.index(self.selected) if self.selected in self.rois else None
        return ([r.snapshot() for r in self.rois], sel)

    def push_undo(self, snap=None):
        self.undo_stack.append(snap if snap is not None else self.snapshot())
        if len(self.undo_stack) > UNDO_LIMIT:
            self.undo_stack.pop(0)
        self.dirty = True

    def undo(self):
        if not self.undo_stack:
            self.notify("되돌릴 작업이 없습니다")
            return

        snaps, sel = self.undo_stack.pop()
        self.rois = [Roi.from_snapshot(s) for s in snaps]
        self.selected = self.rois[sel] if sel is not None else None
        self.notify("되돌렸습니다")

    # ---------------------------------------------- 추적

    def advance(self, r, f, prev_frame, frame):
        if r.state is None:
            recent = [r.boxes[k] for k in range(max(r.start, r.last - VEL_WINDOW + 1), r.last + 1)]
            r.state = TrackState(recent, self.W, self.H, self.fps)

        box = r.state.step(prev_frame, frame, r.boxes[r.last])

        if box is None:
            r.exited_at = r.last
        else:
            r.boxes[f] = box
            r.last = f

    def ensure_tracked(self, target):
        pending = [r for r in self.rois if r.start <= target and r.needs_tracking(target)]
        if not pending:
            return

        f0 = min(r.last for r in pending) + 1
        show = target - f0 > 15

        for f in range(f0, target + 1):
            prev_frame = self.video.get(f - 1)
            frame = self.video.get(f)

            if frame is None or prev_frame is None:
                break

            for r in pending:
                if r.last == f - 1 and r.needs_tracking(f):
                    self.advance(r, f, prev_frame, frame)

            if show and (f - f0) % 5 == 0:
                self.show_busy(frame, f"추적 중... {f} / {target}", (f - f0) / (target - f0 + 1))

    def track_backward(self, r):
        if r.mode != "track":
            self.notify("고정 모드 ROI는 역추적할 수 없습니다")
            return

        s = r.start
        if s == 0:
            self.notify("이미 첫 프레임입니다")
            return

        self.push_undo()

        lo = max(0, s - int(self.fps * BACKTRACK_MAX_SEC))
        recent = [r.boxes[k] for k in range(min(r.last, s + VEL_WINDOW - 1), s - 1, -1)]
        st = TrackState(recent, self.W, self.H, self.fps)

        prev_frame = self.video.get(s)
        prev_box = r.boxes[s]
        f = s - 1
        reason = "최대 길이"

        while f >= lo:
            a = max(lo, f - BACK_CHUNK + 1)
            frames = [self.video.get(g) for g in range(a, f + 1)]

            for g in range(f, a - 1, -1):
                frame = frames[g - a]

                if frame is None:
                    reason = "읽기 실패"
                    f = -1
                    break

                box = st.step(prev_frame, frame, prev_box)

                if box is None:
                    reason = "화면 밖으로 나감"
                    f = -1
                    break

                if st.lost:
                    reason = "대상을 놓침"
                    f = -1
                    break

                r.boxes[g] = box
                r.start = g
                prev_frame, prev_box = frame, box

                if (s - g) % 5 == 0:
                    key = self.show_busy(frame, f"역추적 중... {g} (Esc 중단)", (s - g) / (s - lo))
                    if key == KEY_ESC:
                        reason = "중단"
                        f = -1
                        break
            else:
                f = a - 1
                continue

            break

        if r.start == 0 and reason == "최대 길이":
            reason = "첫 프레임 도달"

        self.notify(f"역추적 완료: {s - r.start}프레임 ({reason})", 4)

    # ---------------------------------------------- 이동

    def goto(self, f):
        f = max(0, min(self.video.total - 1, f))

        if f > self.cur:
            self.ensure_tracked(f)

        self.cur = f

    def on_trackbar(self, pos):
        if pos != self.cur:
            self.seek_request = pos

    # ---------------------------------------------- 마우스

    def rois_at_cur(self):
        return [(r, r.box_at(self.cur)) for r in self.rois if r.box_at(self.cur) is not None]

    def hit_test(self, x, y):
        items = self.rois_at_cur()

        # 선택된 ROI 우선
        items.sort(key=lambda it: it[0] is self.selected)

        for r, box in reversed(items):
            bx, by, bw, bh = box
            corners = [(bx, by), (bx + bw, by), (bx, by + bh), (bx + bw, by + bh)]

            for i, (cx, cy) in enumerate(corners):
                if abs(x - cx) <= HANDLE and abs(y - cy) <= HANDLE:
                    return r, "resize", corners[3 - i]

        for r, box in reversed(items):
            bx, by, bw, bh = box
            if bx <= x <= bx + bw and by <= y <= by + bh:
                return r, "move", None

        return None, None, None

    def on_mouse(self, event, x, y, flags, param):
        if self.modal:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.playing = False
            r, kind, anchor = self.hit_test(x, y)

            if r is not None:
                self.selected = r
                self.drag = kind
                self.drag_snapshot = self.snapshot()
                self.drag_rect = None

                bx, by, _, _ = r.box_at(self.cur)
                self.drag_offset = (x - bx, y - by)
                self.drag_anchor = anchor
                return

            self.selected = None
            self.drag = "draw"
            self.drag_start = (x, y)
            self.drag_rect = None

        elif event == cv2.EVENT_MOUSEMOVE and self.drag:
            if self.drag == "draw":
                sx, sy = self.drag_start
                self.drag_rect = (min(sx, x), min(sy, y), abs(x - sx), abs(y - sy))
                return

            r = self.selected
            old = r.box_at(self.cur)
            if old is None:
                return

            if self.drag == "move":
                new = (x - self.drag_offset[0], y - self.drag_offset[1], old[2], old[3])
            else:
                ax, ay = self.drag_anchor
                x = max(0, min(self.W, x))
                y = max(0, min(self.H, y))
                new = (min(ax, x), min(ay, y), max(MIN_ROI, abs(x - ax)), max(MIN_ROI, abs(y - ay)))

            if self.drag_snapshot is not None:
                self.push_undo(self.drag_snapshot)
                self.drag_snapshot = None

            r.set_box(self.cur, new)

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drag == "draw" and self.drag_rect is not None:
                box = clip_box(self.drag_rect, self.W, self.H)

                if box is not None and box[2] > MIN_ROI and box[3] > MIN_ROI:
                    self.push_undo()
                    r = Roi("fixed" if self.fixed_mode else "track", self.cur, tuple(map(float, box)))
                    self.rois.append(r)
                    self.selected = r

            self.drag = None
            self.drag_rect = None
            self.drag_snapshot = None

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.playing = False
            r, _, _ = self.hit_test(x, y)

            if r is not None:
                self.push_undo()
                r.cut_after(self.cur)
                self.selected = r
                self.notify(f"ROI를 {self.cur} 프레임에서 끝냈습니다 (되돌리기: Ctrl+Z)")

    # ---------------------------------------------- 화면

    def render(self):
        src = self.video.get(self.cur)
        if src is None:
            return None

        frame = src.copy()

        for r, box in self.rois_at_cur():
            vis = clip_box(box, self.W, self.H)
            if vis is not None:
                apply_mosaic(frame, vis)

        for r, box in self.rois_at_cur():
            vis = clip_box(box, self.W, self.H)
            if vis is None:
                continue

            x, y, w, h = vis
            lost = r.mode == "track" and r.state is not None and r.state.lost and r.last == self.cur

            if r is self.selected:
                color = (0, 255, 255)
            elif lost:
                color = (255, 0, 255)
            elif r.mode == "track":
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            if r is self.selected:
                for cx, cy in [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]:
                    cv2.rectangle(frame, (cx - 4, cy - 4), (cx + 4, cy + 4), color, -1)

            if lost:
                draw_text(frame, "대상 놓침 - 드래그로 위치 수정", (x, max(0, y - 24)), 16, color, bg=True)

        if self.drag_rect is not None:
            x, y, w, h = map(int, self.drag_rect)
            color = (0, 0, 255) if self.fixed_mode else (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)

        self.draw_hud(frame)
        return frame

    def draw_hud(self, frame):
        size = max(14, self.H // 40)
        mode = "고정 모드" if self.fixed_mode else "추적 모드"
        state = "재생" if self.playing else "일시정지"
        text = f"{self.cur} / {self.video.total - 1}   |   {mode}   |   {state}   |   ROI {len(self.rois)}개   |   H 도움말"
        color = (80, 80, 255) if self.fixed_mode else (80, 255, 80)

        draw_text(frame, text, (10, 8), size, color, bg=True)

        if self.message and time.time() < self.message_until:
            draw_text(frame, self.message, (10, self.H - size - 14), size, (255, 255, 255), bg=True)

    def show_busy(self, frame, text, ratio):
        img = frame.copy()
        size = max(14, self.H // 36)
        draw_text(img, text, (10, self.H // 2 - size), size, (255, 255, 255), bg=True)

        bar_w = self.W - 20
        y = self.H // 2 + 10
        cv2.rectangle(img, (10, y), (10 + bar_w, y + 10), (80, 80, 80), -1)
        cv2.rectangle(img, (10, y), (10 + int(bar_w * min(1.0, ratio)), y + 10), (0, 200, 255), -1)

        cv2.imshow(WINDOW, img)
        return cv2.waitKeyEx(1)

    def show_help(self):
        src = self.video.get(self.cur)
        img = (src * 0.25).astype(np.uint8) if src is not None else np.zeros((self.H, self.W, 3), np.uint8)

        lines = len(HELP_LINES)
        size = max(12, min(24, int(self.H / (lines * 1.45))))
        step = int(size * 1.4)
        y = max(10, (self.H - step * lines) // 2)
        x_key = max(20, self.W // 10)
        x_desc = x_key + size * 8

        for item in HELP_LINES:
            if item is not None:
                key, desc = item

                if key:
                    draw_text(img, key, (x_key, y), size, (0, 255, 255))
                    draw_text(img, desc, (x_desc, y), size, (255, 255, 255))
                else:
                    color = (120, 255, 120) if desc.startswith("[") or y < step * 3 else (220, 220, 220)
                    draw_text(img, desc, (x_key, y), size, color)

            y += step

        self.modal = True
        cv2.imshow(WINDOW, img)

        while True:
            key = cv2.waitKeyEx(50)
            if key != -1:
                break
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break

        self.modal = False

    # ---------------------------------------------- 저장

    def save(self):
        self.playing = False

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.W, self.H))

        if not out.isOpened():
            self.notify("저장 파일을 만들 수 없습니다: " + self.output_path, 5)
            return

        cap = cv2.VideoCapture(self.video.path, cv2.CAP_FFMPEG)
        prev = None
        cancelled = False
        total = self.video.total

        for f in range(total):
            ret, frame = cap.read()
            if not ret:
                break

            # 아직 추적하지 않은 구간은 저장하면서 이어서 추적
            for r in self.rois:
                if prev is not None and r.last == f - 1 and r.needs_tracking(f):
                    self.advance(r, f, prev, frame)

            prev = frame
            result = frame.copy()

            for r in self.rois:
                box = r.box_at(f)
                if box is not None:
                    vis = clip_box(box, self.W, self.H)
                    if vis is not None:
                        apply_mosaic(result, vis)

            out.write(result)

            if f % 10 == 0:
                key = self.show_busy(result, f"저장 중... {f} / {total} (Esc 취소)", f / total)
                if key == KEY_ESC:
                    cancelled = True
                    break

        out.release()
        cap.release()

        if cancelled:
            try:
                os.remove(self.output_path)
            except OSError:
                pass
            self.notify("저장을 취소했습니다")
        else:
            self.dirty = False
            self.notify("저장 완료: " + os.path.basename(self.output_path), 5)

    # ---------------------------------------------- 키

    def handle_key(self, key):
        """False 를 반환하면 종료."""
        if key == -1:
            return True

        if 0 <= key < 0x110000:
            ch = chr(key)
            ch = HANGUL_KEYS.get(ch, ch).lower()
        else:
            ch = ""

        if ch == "q" or key == KEY_ESC:
            if self.dirty and not self.quit_armed:
                self.quit_armed = True
                self.notify("저장하지 않은 변경이 있습니다. 한 번 더 누르면 종료합니다 (S: 저장)", 4)
                return True
            return False

        self.quit_armed = False

        if key == KEY_SPACE or ch == "a":
            if self.cur >= self.video.total - 1:
                self.cur = 0
            self.playing = not self.playing

        elif ch == "f":
            self.fixed_mode = not self.fixed_mode

        elif ch == "s":
            self.save()

        elif ch == "h":
            self.playing = False
            self.show_help()

        elif ch == "b":
            if self.selected is None:
                self.notify("먼저 ROI를 클릭해서 선택하세요")
            else:
                self.playing = False
                self.track_backward(self.selected)

        elif ch == "u" or key == KEY_CTRL_Z:
            self.undo()

        elif ch == "d" or key == KEY_DELETE:
            if self.selected in self.rois:
                self.push_undo()
                self.rois.remove(self.selected)
                self.selected = None
                self.notify("ROI를 삭제했습니다 (되돌리기: Ctrl+Z)")

        elif ch == "x":
            if self.rois:
                self.push_undo()
                self.rois.clear()
                self.selected = None
                self.notify("모든 ROI를 삭제했습니다 (되돌리기: Ctrl+Z)")

        elif ch == "z" or key == KEY_LEFT:
            self.playing = False
            self.goto(self.cur - 1)

        elif ch == "c" or key == KEY_RIGHT:
            self.playing = False
            self.goto(self.cur + 1)

        elif key == KEY_DOWN:
            self.playing = False
            self.goto(self.cur - 10)

        elif key == KEY_UP:
            self.playing = False
            self.goto(self.cur + 10)

        elif key == KEY_HOME:
            self.playing = False
            self.goto(0)

        elif key == KEY_END:
            self.playing = False
            self.goto(self.video.total - 1)

        return True

    # ---------------------------------------------- 메인 루프

    def run(self):
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

        scale = min(1.0, 1280 / self.W, 760 / self.H)
        cv2.resizeWindow(WINDOW, int(self.W * scale), int(self.H * scale))

        cv2.createTrackbar(TRACKBAR, WINDOW, 0, max(1, self.video.total - 1), self.on_trackbar)
        cv2.setMouseCallback(WINDOW, self.on_mouse)

        self.show_help()

        while True:
            t0 = time.time()

            if self.seek_request is not None:
                target, self.seek_request = self.seek_request, None
                self.playing = False
                self.goto(target)

            if self.playing:
                if self.cur >= self.video.total - 1:
                    self.playing = False
                else:
                    self.goto(self.cur + 1)

            frame = self.render()

            if frame is None:
                # 프레임 수 정보가 실제보다 큰 영상
                self.video.total = max(1, self.cur)
                self.cur = self.video.total - 1
                self.playing = False
                continue

            cv2.imshow(WINDOW, frame)

            if self.trackbar_pos != self.cur:
                self.trackbar_pos = self.cur
                cv2.setTrackbarPos(TRACKBAR, WINDOW, self.cur)

            elapsed = int((time.time() - t0) * 1000)
            wait = max(1, int(1000 / self.fps) - elapsed) if self.playing else 15
            key = cv2.waitKeyEx(wait)

            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break

            if not self.handle_key(key):
                break

        cv2.destroyAllWindows()
        self.video.release()


# ---------------------------------------------------------------- 시작

def show_error(text):
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("모자이크 편집기", text)
        root.destroy()
    except Exception:
        print(text)


def choose_input():
    if len(sys.argv) > 1:
        return sys.argv[1]

    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="모자이크할 영상 선택",
            initialdir=os.getcwd(),
            filetypes=[("영상", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("모든 파일", "*.*")],
        )
        root.destroy()

        if path:
            return path
    except Exception:
        pass

    return DEFAULT_INPUT if os.path.exists(DEFAULT_INPUT) else None


def main():
    path = choose_input()

    if not path:
        return

    try:
        Editor(path).run()
    except Exception as e:
        show_error(str(e))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
