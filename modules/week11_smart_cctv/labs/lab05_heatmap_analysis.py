"""Lab 05: 히트맵 분석 실습 - 간소화 버전"""
import cv2, numpy as np
from ultralytics import YOLO

def lab05_heatmap():
    print("=== Lab 05: 히트맵 분석 ===\n")
    model = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    h, w = frame.shape[:2]

    heatmap = np.zeros((h, w), dtype=np.float32)
    decay_factor = 0.995

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 시간 감쇠
        heatmap *= decay_factor

        # YOLOv8 탐지
        results = model(frame, conf=0.5, classes=[0], verbose=False)

        # 히트맵 누적
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                center_x, center_y = int((x1+x2)/2), int((y1+y2)/2)

                # Gaussian 블러
                for i in range(max(0, center_y-20), min(h, center_y+20)):
                    for j in range(max(0, center_x-20), min(w, center_x+20)):
                        dist = np.sqrt((j-center_x)**2 + (i-center_y)**2)
                        if dist <= 20:
                            value = np.exp(-(dist**2) / (2 * (20/3)**2))
                            heatmap[i, j] += value

        # 시각화
        heatmap_normalized = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(frame, 0.6, heatmap_colored, 0.4, 0)

        # 핫스팟 추출
        threshold = np.percentile(heatmap, 95)
        hotspot_mask = (heatmap > threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(hotspot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for idx, cnt in enumerate(contours[:3]):
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
                cv2.circle(overlay, (cx, cy), 30, (255, 255, 255), 2)
                cv2.putText(overlay, f"#{idx+1}", (cx-10, cy+10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Lab 05: Heatmap Analysis', overlay)
        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    lab05_heatmap()
