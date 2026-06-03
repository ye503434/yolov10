import cv2
import torch
from ultralytics import YOLOv10


video_path = "車流.mp4"
print("正在載入 YOLOv10m 模型...")
model = YOLOv10('yolov10m.pt')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

cap = cv2.VideoCapture(video_path)
# 順利開啟後，讀取影片資訊
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"影片開啟成功！解析度: {width}x{height}，總影格數: {total_frames}")

line_y = int(height * 0.5)
track_history = {}
counted_ids = set()

vehicle_counts = {
    "car": 0,
    "truck": 0,
    "bus": 0,
    "motorcycle": 0
}

class_map = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

frame_count = 0
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1

    # 降低 conf 門檻至 0.15 捕捉小目標
    results = model.track(frame, persist=True, classes=[2, 3, 5, 7], conf=0.15, verbose=False)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        clss = results[0].boxes.cls.cpu().numpy().astype(int)

        for box, track_id, cls in zip(boxes, track_ids, clss):
            x1, y1, x2, y2 = box
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            label_name = class_map.get(cls, "unknown")

            prev_cy = track_history.get(track_id, cy)
            track_history[track_id] = cy

            # 寬鬆判定：只要跨過中央橫線就計數
            if cy > line_y and track_id not in counted_ids:
                counted_ids.add(track_id)
                if label_name in vehicle_counts:
                    vehicle_counts[label_name] += 1

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, f"{label_name} ID:{track_id}", (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # 繪製 UI
    cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 3)
    cv2.rectangle(frame, (10, 10), (280, 160), (0, 0, 0), -1)
    cv2.putText(frame, f"Traffic Summary (Frame {frame_count}):", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 2)

    y_offset = 65
    for name, count in vehicle_counts.items():
        cv2.putText(frame, f"- {name.upper()}: {count}", (30, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y_offset += 22

    # 顯示畫面
    cv2.imshow("YOLOv10 - Intelligent Traffic Vehicle Counter", frame)

    # Windows 系統下必須確保有 waitKey，視窗才會刷新渲染
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

for name, count in vehicle_counts.items():
    print(f"{name.upper()}: {count} 輛")