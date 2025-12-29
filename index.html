import cv2

video_path = "input.mp4"
output_path = "output_mosaic.mp4"

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
if not ret:
    print("영상 로드 실패")
    exit()

# ROI 선택
roi = cv2.selectROI("Select Face (ENTER)", frame, False)
cv2.destroyWindow("Select Face (ENTER)")

# 추적기 생성
tracker = cv2.TrackerCSRT_create()
tracker.init(frame, roi)

# 비디오 저장 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(
    output_path,
    fourcc,
    cap.get(cv2.CAP_PROP_FPS),
    (frame.shape[1], frame.shape[0])
)

def mosaic(img, r, size=15):
    x,y,w,h = map(int, r)
    roi = img[y:y+h, x:x+w]
    if roi.size == 0:
        return img
    roi = cv2.resize(roi, (w//size, h//size))
    roi = cv2.resize(roi, (w, h), interpolation=cv2.INTER_NEAREST)
    img[y:y+h, x:x+w] = roi
    return img

while True:
    ret, frame = cap.read()
    if not ret:
        break

    ok, roi = tracker.update(frame)
    if ok:
        frame = mosaic(frame, roi)

    out.write(frame)
    cv2.imshow("Mosaic Tracking", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
