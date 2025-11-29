import cv2
import time
from ultralytics import YOLO
import linebot

# 設定警報冷卻時間 (秒)，例如 10 秒內不重複發送
ALERT_COOLDOWN = 120 


CLASS_NAMES = ["helmet", "person", "head"]

# 載入模型
model = YOLO("best.pt")

# 開啟攝影機
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# 記錄上一次發送警報的時間
last_alert_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # 進行偵測
    results = model(frame, imgsz=640, conf=0.5)
    result = results[0]

    detected_violation = False 

    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        class_name = CLASS_NAMES[cls_id]
        label = f"{class_name} {conf:.2f}"

        # 繪製方框顏色：head(沒戴安全帽)用紅色，其他用綠色
        color = (0, 0, 255) if class_name == "head" else (0, 255, 0)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
        )

        # 判斷邏輯：如果偵測到 'head'，視為未戴安全帽
        if class_name == "head":
            detected_violation = True

    # === 警報觸發邏輯 ===
    # 如果 1. 偵測到違規 且 2. 距離上次發送超過冷卻時間
    current_time = time.time()
    if detected_violation and (current_time - last_alert_time > ALERT_COOLDOWN):
        print("偵測到違規！正在發送 LINE 通知...")
        
        # 呼叫 linebot.py 裡面的功能
        linebot.send_message("警告：偵測到人員未戴安全帽！")
        
        last_alert_time = current_time

    cv2.imshow("Hard Hat Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
