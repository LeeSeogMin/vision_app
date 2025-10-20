"""Lab 03: ROI 침입 감지 실습"""
import cv2
import numpy as np
from ultralytics import YOLO

roi_points = []

def mouse_callback(event, x, y, flags, param):
    global roi_points
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_points.append((x, y))
        print(f"Point {len(roi_points)}: ({x}, {y})")

def lab03_roi_intrusion():
    global roi_points
    print("=== Lab 03: ROI 침입 감지 ===\n")
    print("마우스로 ROI 영역 클릭 (4개 점, ESC: 완료)\n")

    model = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture(0)

    # ROI 설정
    ret, frame = cap.read()
    cv2.namedWindow('ROI Setup')
    cv2.setMouseCallback('ROI Setup', mouse_callback)

    while len(roi_points) < 4:
        display = frame.copy()
        for i, pt in enumerate(roi_points):
            cv2.circle(display, pt, 5, (0, 255, 0), -1)
            cv2.putText(display, str(i+1), (pt[0]+10, pt[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if len(roi_points) > 1:
            cv2.polylines(display, [np.array(roi_points)], False, (0, 255, 0), 2)

        cv2.imshow('ROI Setup', display)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    if len(roi_points) < 4:
        print("❌ ROI 설정 실패")
        return

    roi_polygon = np.array(roi_points, dtype=np.int32)
    print(f"✅ ROI 설정 완료: {roi_points}\n")

    # 침입 감지
    intrusion_tracks = {}
    threshold_seconds = 3

    frame_idx = 0
    fps = 30

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLOv8 탐지
        results = model(frame, conf=0.5, classes=[0], verbose=False)

        # ROI 그리기
        overlay = frame.copy()
        cv2.fillPoly(overlay, [roi_polygon], (0, 0, 255))
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        cv2.polylines(frame, [roi_polygon], True, (0, 0, 255), 3)

        # 침입 검사
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                center = (int((x1+x2)/2), int((y1+y2)/2))

                # ROI 내부 검사
                is_inside = cv2.pointPolygonTest(roi_polygon, center, False) >= 0

                track_id = int(box.conf[0] * 1000)  # 임시 ID

                if is_inside:
                    if track_id not in intrusion_tracks:
                        intrusion_tracks[track_id] = frame_idx / fps

                    duration = (frame_idx / fps) - intrusion_tracks[track_id]

                    # 바운딩 박스
                    color = (0, 0, 255) if duration >= threshold_seconds else (0, 165, 255)
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

                    # 경고
                    if duration >= threshold_seconds:
                        cv2.putText(frame, "INTRUSION!", (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('Lab 03: Intrusion Detection', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    lab03_roi_intrusion()
