"""
Lab 01: YOLOv8 객체 탐지 실습
================================

학습 목표:
1. YOLOv8 모델 로드 및 추론
2. 바운딩 박스 시각화
3. 클래스 필터링
4. 성능 측정

Author: Smart Vision Team
Date: 2025-01-20
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time

# COCO 클래스 이름
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog'
]

def lab01_basic_detection():
    """기본 탐지"""
    print("=== Lab 01-1: 기본 YOLOv8 탐지 ===\n")

    # 1. 모델 로드
    print("1. YOLOv8 nano 모델 로드...")
    model = YOLO('yolov8n.pt')  # 자동 다운로드
    print("✅ 모델 로드 완료\n")

    # 2. 테스트 이미지 생성 (또는 실제 이미지 사용)
    print("2. 테스트 이미지 준비...")
    # 실제 사용 시: frame = cv2.imread('test.jpg')
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 200  # 회색 이미지
    print("✅ 이미지 준비 완료\n")

    # 3. 추론
    print("3. YOLOv8 추론 실행...")
    start_time = time.time()
    results = model(frame, conf=0.5, verbose=False)
    inference_time = (time.time() - start_time) * 1000
    print(f"✅ 추론 완료 (소요 시간: {inference_time:.1f}ms)\n")

    # 4. 결과 파싱
    print("4. 탐지 결과:")
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f'Class {class_id}'

            detections.append({
                'bbox': [x1, y1, x2, y2],
                'conf': confidence,
                'class_id': class_id,
                'class_name': class_name
            })

            print(f"   - {class_name}: {confidence:.2%} at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")

    if len(detections) == 0:
        print("   (탐지된 객체 없음)")

    print(f"\n총 {len(detections)}개 객체 탐지")
    return detections


def lab01_class_filtering():
    """클래스 필터링"""
    print("\n\n=== Lab 01-2: 클래스 필터링 (사람, 차량만) ===\n")

    model = YOLO('yolov8n.pt')
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 200

    # 사람(0), 차량(2), 버스(5), 트럭(7)만 탐지
    target_classes = [0, 2, 5, 7]
    print(f"1. 필터 클래스: {[COCO_CLASSES[c] for c in target_classes]}\n")

    print("2. 필터링된 추론 실행...")
    results = model(frame, conf=0.5, classes=target_classes, verbose=False)

    print("3. 탐지 결과:")
    count = 0
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = COCO_CLASSES[class_id]
            confidence = float(box.conf[0])
            print(f"   - {class_name}: {confidence:.2%}")
            count += 1

    if count == 0:
        print("   (탐지된 객체 없음)")

    print(f"\n✅ 필터링으로 {count}개 객체만 탐지")


def lab01_visualization():
    """시각화"""
    print("\n\n=== Lab 01-3: 탐지 결과 시각화 ===\n")

    model = YOLO('yolov8n.pt')

    # 샘플 이미지 또는 웹캠
    print("1. 비디오 소스 선택:")
    print("   - 웹캠을 사용하려면 'w' 입력")
    print("   - 샘플 영상을 사용하려면 Enter")

    choice = input("선택: ").strip().lower()

    if choice == 'w':
        cap = cv2.VideoCapture(0)
        print("✅ 웹캠 연결\n")
    else:
        # 샘플 영상이 있다면 사용, 없으면 생성한 이미지
        cap = None
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 200
        print("✅ 샘플 이미지 사용\n")

    print("2. YOLOv8 탐지 시작 (ESC 종료)...")
    print("   - 초록색 박스: 사람")
    print("   - 파란색 박스: 차량\n")

    frame_count = 0
    fps_list = []

    while True:
        if cap is not None:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            # 정적 이미지는 한 번만
            if frame_count > 0:
                break

        start_time = time.time()

        # 추론
        results = model(frame, conf=0.5, classes=[0, 2], verbose=False)

        # 시각화
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = COCO_CLASSES[class_id]

                # 색상 설정
                if class_id == 0:  # 사람
                    color = (0, 255, 0)  # 초록
                else:  # 차량
                    color = (255, 0, 0)  # 파란

                # 바운딩 박스
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # 레이블
                label = f"{class_name} {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # FPS 계산
        fps = 1.0 / (time.time() - start_time)
        fps_list.append(fps)

        # FPS 표시
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 결과 표시
        cv2.imshow('Lab 01-3: YOLOv8 Detection', frame)

        # 종료 (ESC)
        if cv2.waitKey(1) & 0xFF == 27:
            break

        frame_count += 1

        # 정적 이미지는 잠시 대기
        if cap is None:
            cv2.waitKey(3000)
            break

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()

    # 통계
    if fps_list:
        print(f"\n✅ 처리 완료:")
        print(f"   - 총 프레임: {frame_count}")
        print(f"   - 평균 FPS: {np.mean(fps_list):.1f}")
        print(f"   - 최소/최대 FPS: {np.min(fps_list):.1f} / {np.max(fps_list):.1f}")


def lab01_performance_comparison():
    """성능 비교"""
    print("\n\n=== Lab 01-4: 모델 크기별 성능 비교 ===\n")

    models = {
        'YOLOv8n': 'yolov8n.pt',  # nano
        'YOLOv8s': 'yolov8s.pt',  # small
        'YOLOv8m': 'yolov8m.pt',  # medium
    }

    frame = np.ones((480, 640, 3), dtype=np.uint8) * 200

    print("모델별 추론 속도 측정 (10회 평균):\n")
    print(f"{'Model':<10} {'Avg Time (ms)':<15} {'FPS':<10} {'파라미터':<15}")
    print("-" * 60)

    for model_name, model_path in models.items():
        try:
            model = YOLO(model_path)

            # Warm-up
            model(frame, verbose=False)

            # 측정
            times = []
            for _ in range(10):
                start = time.time()
                model(frame, verbose=False)
                times.append((time.time() - start) * 1000)

            avg_time = np.mean(times)
            fps = 1000 / avg_time

            # 파라미터 수
            params = {
                'YOLOv8n': '3.2M',
                'YOLOv8s': '11.2M',
                'YOLOv8m': '25.9M'
            }

            print(f"{model_name:<10} {avg_time:<15.1f} {fps:<10.1f} {params[model_name]:<15}")

        except Exception as e:
            print(f"{model_name:<10} Error: {e}")

    print("\n✅ 성능 비교 완료")
    print("\n💡 교육 포인트:")
    print("   - nano (n): 가장 빠름, 실시간 처리 적합")
    print("   - small (s): 균형잡힌 속도와 정확도")
    print("   - medium (m): 높은 정확도, GPU 권장")


def main():
    """메인 함수"""
    print("=" * 60)
    print("Lab 01: YOLOv8 객체 탐지 실습")
    print("=" * 60)

    while True:
        print("\n\n=== 실습 메뉴 ===")
        print("1. Lab 01-1: 기본 탐지")
        print("2. Lab 01-2: 클래스 필터링")
        print("3. Lab 01-3: 시각화 (웹캠/이미지)")
        print("4. Lab 01-4: 성능 비교")
        print("5. 전체 실행")
        print("0. 종료")

        choice = input("\n선택 (0-5): ").strip()

        if choice == '1':
            lab01_basic_detection()
        elif choice == '2':
            lab01_class_filtering()
        elif choice == '3':
            lab01_visualization()
        elif choice == '4':
            lab01_performance_comparison()
        elif choice == '5':
            lab01_basic_detection()
            lab01_class_filtering()
            lab01_visualization()
            lab01_performance_comparison()
        elif choice == '0':
            print("\n프로그램을 종료합니다.")
            break
        else:
            print("잘못된 선택입니다.")


if __name__ == "__main__":
    main()
