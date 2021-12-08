import requests
import json
call_ids = ["04fad854-f00b-44f1-8186-2f7c343ab71a.wav_knowlarity_stt_alphac_9178479786287149939"]
for call_id in call_ids:
    response = requests.get(f"http://127.0.0.1:8000/sentiment/{call_id}",  headers = {'Content-Type': 'application/json' })
    status_code = response.status_code
    if status_code >= 200 and status_code <= 299:
        data = response.text.encode('utf8')
        json_data = json.loads(data)
        if json_data.get("status") == "success":
            print(f"Successfully Done for {call_id}")
        else:
            print(f"Failed for {call_id}")
