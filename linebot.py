import requests

# 將LINE資訊填入這裡
LINE_ACCESS_TOKEN = ""
LINE_USER_ID = ""

def send_message(message_text):
    """
    發送 LINE 訊息給指定使用者
    """
    url = "https://api.line.me/v2/bot/message/push"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LINE_ACCESS_TOKEN}"
    }
    payload = {
        "to": LINE_USER_ID,
        "messages": 
        [
            {
                "type": "text",
                "text": message_text
            }
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            print(">> [LINE 通知] 發送成功")
        else:
            print(f">> [LINE 通知] 發送失敗: {response.status_code}, {response.text}")
    except Exception as e:
        print(f">> [LINE 通知] 錯誤: {e}")
