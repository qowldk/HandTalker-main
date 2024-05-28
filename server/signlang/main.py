import json
import cv2
import mediapipe as mp
import numpy as np
import os
import base64
import asyncio
import websockets
import queue
import threading
import time
from tensorflow.keras.models import load_model


import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from fingerspell.fs_8081 import finger_spell
from LLM.LLM_8082 import generate_sentence


mp_holistic = mp.solutions.holistic  # holistic: 얼굴, 손 등 감지

def process_frame(image_data):
    # base64 형식의 이미지 데이터를 bytes로 디코딩
    image_bytes = base64.b64decode(image_data)
    # bytes를 NumPy 배열로 변환
    np_arr = np.frombuffer(image_bytes, np.uint8)
    # NumPy 배열을 이미지로 변환
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False  # 이미지 수정 불가 (결과 왜곡 방지?)
    results = model.process(image)  # 모델을 사용해 입력 이미지에 대한 예측 수행
    image.flags.writeable = True  # 이미지 다시 수정가능
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
    return image, results


script_directory = os.path.dirname(os.path.abspath(__file__))
print("현재 작업 디렉토리:", script_directory)

MODEL_PATH = os.path.join(script_directory, 'model_ko.h5')
model = load_model(MODEL_PATH, compile=False)  # 코랩 사용시 compile=False 필수


# 수어 단어와 라벨 목록 (actions)
word_list = 'db.txt'
actions = {}
word_list_dir = os.path.join(script_directory, "db.txt")
with open(word_list_dir, 'r', encoding='utf-8') as file:
    for line in file:
        parts = line.split()
        if len(parts) >= 2:
            a = parts[0]
            b = parts[1]
            actions[int(a)] = b
print(actions) # debug

sentence_length = 10
seq_length = 30

frame_queue = queue.Queue()
result_queue = queue.Queue() # 멀티스레딩 - 처리할 작업 목록

# 데이터 전처리, lstm predict 스레드
def frame_processor():
    global frame_queue
    seq = []
    previous = ''
    detected_word = ['','','']
    detected_word_len = 3
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4)
    
    while True:
        message = frame_queue.get()
        if message==None:
            print("None Message")
            continue
        frame = process_frame(message)
        if frame is None:
            break
        image, result = mediapipe_detection(frame, hands)
        
        if result.multi_hand_landmarks is not None:
            h = 0  
            d1 = np.empty(0)
            d2 = np.empty(0)
            for res in result.multi_hand_landmarks:  # 감지된 손의 수만큼 반복
                h += 1
                joint = np.zeros((21, 2))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y] 

                # 각 손가락 마디 벡터 계산
                v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]  # Parent joint
                v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]  # Child joint
                v = v2 - v1  # [20, 3]
                
                # 정규화 (크기 1의 단위벡터로)
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                # 내적의 arcos으로 손가락 각 마디의 사이각 계산
                angle = np.arccos(np.einsum('nt,nt->n',
                                            v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                            v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))

                angle = np.degrees(angle)  # 라디안 -> 도
                angle_label = np.array([angle], dtype=np.float32)
                if h == 1:
                    d1 = np.concatenate([joint.flatten(), angle_label[0]])
                else:
                    d2 = np.concatenate([joint.flatten(), angle_label[0]])

            d = np.concatenate([d1, d2])

            if len(d) <= 57: # 한손만 감지될 경우 나머지 손 제로패딩
                d = np.concatenate([d, np.zeros(len(d))])

            seq.append(d)

            if len(seq) < seq_length:  # 시퀀스 최소치가 쌓인 이후부터 판별
                continue

            if len(seq) > seq_length * 50:  # 메모리 정리
                seq = seq[-seq_length:]

            # 시퀀스 데이터를 신경망 모델에 입력으로 사용할 수 있는 형태로 변환
            input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)

            y_pred = model.predict(input_data).squeeze()  # 각 동작에 대한 예측 결과 (각각의 확률)

            i_pred = int(np.argmax(y_pred))  # 최댓값 인덱스: 예측값이 가장 높은 값(동작)의 인덱스
            conf = y_pred[i_pred]  # 가장 확률 높은 동작의 확률
            print("conf? ", conf)
            if conf < 0.9:  # 90% 이상일 때만 수행
                continue

            action = actions[i_pred]
            # print("-", action)

            detected_word.append(action)
            if len(detected_word)>50:
                detected_word = detected_word[-detected_word_len:]
            
            detected = True

            for a in detected_word[-detected_word_len:-1]:
                if a!=detected_word[-1]:
                    detected = False
                    break
            
            if not detected: continue

            if previous == action: 
                print("인식됨(중복)", action)
                seq = []
                frame_queue = queue.Queue() # 작업 리스트 초기화
                continue  # 중복 전달 회피
            previous = action
            print("인식됨", action)
            time.sleep(0.5)
            seq = []
            frame_queue = queue.Queue() # 작업 리스트 초기화
            print("큐 초기화")
            
            result_dict = {'result': action}
            result_json = json.dumps(result_dict)

            result_queue.put(result_json)


# 송수신 스레드
async def handle_client(websocket, path):
    try:
        while True:
            # print(frame_queue.qsize())
            message = await websocket.recv()
            if message:
                frame_queue.put(message)
            if not result_queue.empty():
                result_json = result_queue.get()
                try:
                    if websocket.open:
                        await websocket.send(result_json)
                except Exception as e:
                    print(f"send error: {str(e)}")
    except websockets.exceptions.ConnectionClosedOK:
        pass
    finally:
        frame_queue.put(None)
        result_queue.put(None)


start_server_1 = websockets.serve(handle_client, "localhost", 8080)
start_server_2 = websockets.serve(finger_spell, "localhost", 8081)  
start_server_3 = websockets.serve(generate_sentence, "localhost", 8082)

async def main():
    await asyncio.gather(start_server_1, start_server_2, start_server_3)

if __name__ == "__main__":
    processor_thread = threading.Thread(target=frame_processor)
    processor_thread.start()
    asyncio.get_event_loop().run_until_complete(main())
    asyncio.get_event_loop().run_forever()
    processor_thread.join()



