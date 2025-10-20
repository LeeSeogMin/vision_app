"""Lab 04: 배회 감지 실습 - 간소화 버전"""
import cv2, numpy as np
from ultralytics import YOLO
from collections import deque

tracks_history = {}

def calculate_movement(history):
    if len(history) < 2:
        return 0
    total = sum(np.sqrt((history[i][0]-history[i-1][0])**2 + (history[i][1]-history[i-1][1])**2)
                for i in range(1, len(history)))
    return total

def lab04_loitering():
    print("=== Lab 04: 배회 감지 ===\n")
    model, tracker = YOLO('yolov8n.pt'), {}
    cap, fps, frame_idx = cv2.VideoCapture(0), 30, 0
    loitering_threshold_sec, movement_threshold_px = 10, 100

    while True:
        ret, frame = cap.read()
        if not ret: break

        results = model(frame, conf=0.5, classes=[0], verbose=False)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                tid = int(box.conf[0]*1000)
                center = (int((x1+x2)/2), int((y1+y2)/2))

                if tid not in tracks_history:
                    tracks_history[tid] = deque(maxlen=int(loitering_threshold_sec * fps))
                tracks_history[tid].append(center)

                # 배회 판단
                if len(tracks_history[tid]) >= int(loitering_threshold_sec * fps):
                    movement = calculate_movement(list(tracks_history[tid]))
                    is_loitering = movement < movement_threshold_px

                    color = (0, 165, 255) if is_loitering else (0, 255, 0)
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

                    if is_loitering:
                        cv2.putText(frame, f"LOITERING! ({movement:.0f}px)", (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    # 궤적
                    pts = np.array(list(tracks_history[tid]), dtype=np.int32)
                    cv2.polylines(frame, [pts], False, color, 2)

        cv2.imshow('Lab 04: Loitering Detection', frame)
        if cv2.waitKey(1) & 0xFF == 27: break
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    lab04_loitering()
