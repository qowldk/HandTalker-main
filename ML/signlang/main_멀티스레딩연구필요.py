import json
import cv2
import mediapipe as mp
import numpy as np
import os
import base64
import asyncio
import websockets
import time
import queue
import threading

from tensorflow.keras.models import load_model

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
# dataset_directory = os.path.join(script_directory, "dataset")
print("현재 작업 디렉토리:", script_directory)

MODEL_PATH = os.path.join(script_directory, 'model_ko.h5')
model = load_model(MODEL_PATH, compile=False)  # 코랩 사용시 compile=False 필수


exit_flag = threading.Event()


word_list = 'db.txt'

# 빈 딕셔너리를 초기화합니다.
actions = {}

# 텍스트 파일을 읽기 모드로 엽니다.
word_list_dir = os.path.join(script_directory, "db.txt")

with open(word_list_dir, 'r', encoding='utf-8') as file:
    # 파일의 각 라인을 순회합니다.
    for line in file:
        # 라인을 공백 문자를 기준으로 나눕니다.
        parts = line.split()
        # 첫 번째 요소가 a, 두 번째 요소가 b일 때 딕셔너리에 추가합니다.
        if len(parts) >= 2:
            a = parts[0]
            b = parts[1]
            actions[int(a)] = b

# 파일 이름으로 리스트 읽기
# data_file_list = os.listdir(dataset_directory)
# data_file_list = sorted(data_file_list, reverse=True)

# actions = {}
# for file_name in data_file_list:
#     parts = file_name.split("_")
#     if parts[1] not in actions:
#         actions[int(parts[0])] = parts[1]

print(actions)

sentence_length = 10
seq_length = 30
extra_time = 2.5  # 다음 동작 전달까지 텀

frame_queue = queue.Queue()
result_queue = queue.Queue()

exit_count = 0

def frame_processor():
    seq = []
    previous = ''
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4)
    
    # count=0
    
    global exit_count
    print("스레드 start!!")
    while not exit_flag.is_set():

        message = frame_queue.get()
        if message == None:
            print("None Processing Error!")
            continue
        # print(len(frame_queue))
        frame = process_frame(message)
        if frame is None:
            break
        image, result = mediapipe_detection(frame, hands)
        
        if result.multi_hand_landmarks is None:
            exit_count+=1
            continue
        if result.multi_hand_landmarks is not None:
            h = 0  # 손 두개 감지 로직을 위한 임시 값
            d1 = np.empty(0)
            d2 = np.empty(0)
            for res in result.multi_hand_landmarks:  # 감지된 손의 수만큼 반복
                h += 1
                joint = np.zeros((23, 3))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z]  # visibility: 신뢰도 (0~1)
                joint[21] = [0, 0, 0]
                joint[22] = [1, 1, 1]
                # 각 관절의 벡터 계산
                v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19, 21], :3]  # Parent joint
                v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22], :3]  # Child joint
                v = v2 - v1  # [20, 3]
                # 정규화 (크기 1의 단위벡터로)
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                # 내적의 arcos으로 각도 계산
                angle = np.arccos(np.einsum('nt,nt->n',
                                            v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 0, 16], :],
                                            v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 20, 20], :]))

                angle = np.degrees(angle)  # 라디안 -> 도
                angle_label = np.array([angle], dtype=np.float32)
                if h == 1:
                    d1 = np.concatenate([angle_label[0]])
                else:
                    d2 = np.concatenate([angle_label[0]])

            d = np.concatenate([d1, d2])
            print("DEBUG1", len(d))

            if len(d) <= 17:
                d = np.concatenate([d, np.zeros(len(d))])

            print("DEBUG1.1", len(seq))
            seq.append(d)
            # print("debug ",count)
            # count+=1
            if len(seq) < seq_length:  # 시퀀스 최소치가 쌓인 이후부터 판별
                continue

            print("DEBUG2")
            if len(seq) > seq_length * 100:  # 과도하게 쌓임 방지
                seq = seq[-seq_length:]

            # 시퀀스 데이터를 신경망 모델에 입력으로 사용할 수 있는 형태로 변환
            input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)

            print("DEBUG3")
            y_pred = model.predict(input_data).squeeze()  # 각 동작에 대한 예측 결과 (각각의 확률)

            i_pred = int(np.argmax(y_pred))  # 최댓값 인덱스: 예측값이 가장 높은 값(동작)의 인덱스
            conf = y_pred[i_pred]  # 가장 확률 높은 동작의 확률
            print("conf? ", conf)
            if conf < 0.8:  # 90% 이상일 때만 수행
                continue

            action = actions[i_pred]

            if previous == action: 
                print("인식됨(중복)", action)
                continue  # 중복 전달 회피
            previous = action
            print("인식됨", action)
            seq = []

            result_dict = {'result': action}
            result_json = json.dumps(result_dict)

            result_queue.put(result_json)
    print("off")

# 안되던 코드 -> 수신, 송신이 자주 겹쳐서 확률적으로 송신 되던 것으로 추정

# async def result_sender(websocket):
#     while True:
#         result_json = await asyncio.get_event_loop().run_in_executor(None, result_queue.get)
#         if result_json is None:
#             print("got None")
#             break
#         try:
#             if websocket.open:
#                 print("가주세요..")
#                 await websocket.send(result_json)
#         except Exception as e:
#             print(f"send error: {str(e)}")

# async def handle_client(websocket, path):
#     asyncio.create_task(result_sender(websocket))  # 비동기 작업으로 result_sender 시작
#     try:
#         while True:
#             message = await websocket.recv()
#             if message:
#                 frame_queue.put(message)
#     except websockets.exceptions.ConnectionClosedOK:
#         pass
#     finally:
#         frame_queue.put(None)
#         result_queue.put(None)

socket_closed = True

async def handle_client(websocket, path):
    try:
        global exit_count
        while True:
            if exit_count>20: # 손이 없는 프레임 5개 이상 수신 시 스레드 재시작?
                exit_count = 0
                exit_flag.set() # 종료 요청
                # print("종료 요청")
                # processor_thread.join() # 대기 
                # print("종료 성공")
                exit_flag.clear() # 종료 요청 플래그 초기화
                socket_closed = True
                # print("플래그 초기화")
                print("스레드 종료")
                while not frame_queue.empty():
                    frame_queue.get()
                
                
                processor_thread = threading.Thread(target=frame_processor)
                processor_thread.start()

            message = await websocket.recv()
            if message:
                frame_queue.put(message)
                # print("PUT")
            if not result_queue.empty():
                result_json = result_queue.get()
                print("send?")
                try:
                    if websocket.open:
                        # print("send..")
                        await websocket.send(result_json)
                except Exception as e:
                    print(f"send error: {str(e)}")
    except websockets.exceptions.ConnectionClosedOK:
        pass
    finally:
        frame_queue.put(None)
        result_queue.put(None)


start_server = websockets.serve(handle_client, "localhost", 8080)

async def main():
    await start_server

if __name__ == "__main__":
    processor_thread = threading.Thread(target=frame_processor)
    processor_thread.start()
    asyncio.get_event_loop().run_until_complete(main())
    asyncio.get_event_loop().run_forever()
    processor_thread.join()
