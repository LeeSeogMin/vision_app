"""
Lab 02: ByteTrack 객체 추적 실습
=================================

학습 목표:
1. 간단한 ByteTrack 구현
2. IoU 기반 매칭
3. Track ID 부여 및 궤적 시각화
4. 가려짐(occlusion) 처리

Author: Smart Vision Team
Date: 2025-01-20
"""

import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque, defaultdict
import time


class SimpleTrack:
    """단순화된 Track 클래스"""

    def __init__(self, track_id, bbox, class_id, confidence):
        self.id = track_id
        self.bbox = bbox
        self.class_id = class_id
        self.confidence = confidence
        self.age = 1
        self.time_since_update = 0
        self.history = deque(maxlen=30)
        self.history.append(self.get_center())

    def get_center(self):
        """바운딩 박스 중심점"""
        x1, y1, x2, y2 = self.bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    def update(self, bbox, confidence):
        """탐지 결과로 업데이트"""
        self.bbox = bbox
        self.confidence = confidence
        self.age += 1
        self.time_since_update = 0
        self.history.append(self.get_center())

    def predict(self):
        """등속 모델 예측"""
        if len(self.history) >= 2:
            velocity_x = self.history[-1][0] - self.history[-2][0]
            velocity_y = self.history[-1][1] - self.history[-2][1]

            w = self.bbox[2] - self.bbox[0]
            h = self.bbox[3] - self.bbox[1]

            center_x = self.history[-1][0] + velocity_x
            center_y = self.history[-1][1] + velocity_y

            self.bbox = [
                center_x - w/2,
                center_y - h/2,
                center_x + w/2,
                center_y + h/2
            ]

        self.time_since_update += 1


class SimpleByteTracker:
    """간소화된 ByteTrack 구현"""

    def __init__(self, high_thresh=0.5, low_thresh=0.1, max_age=30):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.max_age = max_age
        self.tracks = {}
        self.next_id = 1

    def update(self, detections):
        """
        detections: List[dict] - [{'bbox': [x1,y1,x2,y2], 'conf': 0.9, 'class': 0}, ...]
        """
        # 고/저 신뢰도 분리
        high_dets = [d for d in detections if d['conf'] >= self.high_thresh]
        low_dets = [d for d in detections if self.low_thresh <= d['conf'] < self.high_thresh]

        # 1단계: 고신뢰도 탐지와 Track 매칭
        matches, unmatched_dets, unmatched_tracks = self._match(high_dets, list(self.tracks.keys()))

        # 매칭된 Track 업데이트
        for det_idx, track_id in matches:
            self.tracks[track_id].update(high_dets[det_idx]['bbox'], high_dets[det_idx]['conf'])

        # 2단계: 저신뢰도 탐지와 미매칭 Track 재매칭
        matches2, _, _ = self._match(low_dets, unmatched_tracks)

        for det_idx, track_id in matches2:
            self.tracks[track_id].update(low_dets[det_idx]['bbox'], low_dets[det_idx]['conf'])
            unmatched_tracks.remove(track_id)

        # 새로운 Track 생성
        for det_idx in unmatched_dets:
            det = high_dets[det_idx]
            self.tracks[self.next_id] = SimpleTrack(
                self.next_id, det['bbox'], det['class'], det['conf']
            )
            self.next_id += 1

        # 미매칭 Track 예측
        for track_id in unmatched_tracks:
            self.tracks[track_id].predict()

        # 오래된 Track 제거
        to_remove = []
        for track_id, track in self.tracks.items():
            if track.time_since_update > self.max_age:
                to_remove.append(track_id)

        for track_id in to_remove:
            del self.tracks[track_id]

        return self.tracks

    def _match(self, detections, track_ids):
        """IoU 기반 greedy 매칭"""
        if len(detections) == 0 or len(track_ids) == 0:
            return [], list(range(len(detections))), track_ids

        # IoU 행렬
        iou_matrix = np.zeros((len(detections), len(track_ids)))
        for i, det in enumerate(detections):
            for j, track_id in enumerate(track_ids):
                iou = self._calculate_iou(det['bbox'], self.tracks[track_id].bbox)
                iou_matrix[i, j] = iou

        # Greedy 매칭
        matches = []
        matched_dets = set()
        matched_tracks = set()
        threshold = 0.3

        while True:
            max_iou = np.max(iou_matrix)
            if max_iou < threshold:
                break

            i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            matches.append((i, track_ids[j]))
            matched_dets.add(i)
            matched_tracks.add(track_ids[j])

            iou_matrix[i, :] = 0
            iou_matrix[:, j] = 0

        unmatched_dets = [i for i in range(len(detections)) if i not in matched_dets]
        unmatched_tracks = [tid for tid in track_ids if tid not in matched_tracks]

        return matches, unmatched_dets, unmatched_tracks

    def _calculate_iou(self, box1, box2):
        """IoU 계산"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0


def lab02_basic_tracking():
    """기본 추적"""
    print("=== Lab 02-1: 기본 ByteTrack 추적 ===\n")

    model = YOLO('yolov8n.pt')
    tracker = SimpleByteTracker()

    print("웹캠 또는 샘플 영상 선택 (w/Enter): ", end="")
    choice = input().strip().lower()

    cap = cv2.VideoCapture(0 if choice == 'w' else 'sample.mp4')
    if not cap.isOpened():
        print("❌ 비디오 열기 실패")
        return

    print("\n✅ 추적 시작 (ESC 종료)\n")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLOv8 탐지
        results = model(frame, conf=0.3, classes=[0, 2], verbose=False)

        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'conf': float(box.conf[0]),
                    'class': int(box.cls[0])
                })

        # ByteTrack 추적
        tracks = tracker.update(detections)

        # 시각화
        for track in tracks.values():
            x1, y1, x2, y2 = map(int, track.bbox)

            # 바운딩 박스
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Track ID
            label = f"ID:{track.id}"
            cv2.putText(frame, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 궤적
            if len(track.history) > 1:
                points = np.array(list(track.history), dtype=np.int32)
                cv2.polylines(frame, [points], False, (0, 0, 255), 2)

        # 정보 표시
        cv2.putText(frame, f"Tracks: {len(tracks)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Lab 02-1: Tracking', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n✅ 추적 완료: {frame_count} 프레임 처리")


def main():
    """메인 함수"""
    print("=" * 60)
    print("Lab 02: ByteTrack 추적 실습")
    print("=" * 60)

    lab02_basic_tracking()


if __name__ == "__main__":
    main()
