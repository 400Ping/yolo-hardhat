import cv2
import time
import requests  # 新增 requests 套件
from ultralytics import YOLO

# ================= 設定區 =================
# 請填入你的 LINE Messaging API 資訊
LINE_ACCESS_TOKEN = ""
LINE_USER_ID = ""

# 設定警報冷卻時間 (秒)，避免訊息轟炸
ALERT_COOLDOWN = 300 
# =========================================

CLASS_NAMES = ["helmet", "person", "head"]

model = YOLO("best.pt")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

# 記錄上一次發送警報的時間
last_alert_time = 0

def send_line_warning(message):
    """發送 LINE 訊息 (使用 Messaging API Push Message)"""
    url = "https://api.line.me/v2/bot/message/push"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LINE_ACCESS_TOKEN}"
    }
    payload = {
        "to": LINE_USER_ID,
        "messages": [
            {
                "type": "text",
                "text": message
            }
        ]
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            print("LINE 警報發送成功")
        else:
            print(f"LINE 警報發送失敗: {response.status_code}, {response.text}")
    except Exception as e:
        print(f"發送錯誤: {e}")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    results = model(frame, imgsz=640, conf=0.5)
    result = results[0]

    detected_violation = False # 標記本幀是否偵測到違規

    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        class_name = CLASS_NAMES[cls_id]
        label = f"{class_name} {conf:.2f}"

        # 繪製方框
        # 如果是 head (未戴安全帽) 用紅色框，其他用綠色
        color = (0, 0, 255) if class_name == "head" else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
        )

        # 判斷是否違規：假設 'head' 代表未戴安全帽
        if class_name == "head":
            detected_violation = True

    # 檢查是否需要發送警報 (檢測到違規 且 超過冷卻時間)
    current_time = time.time()
    if detected_violation and (current_time - last_alert_time > ALERT_COOLDOWN):
        send_line_warning("警告：偵測到有人員未戴安全帽！")
        last_alert_time = current_time

    cv2.imshow("Hard Hat Detector (Mac)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
