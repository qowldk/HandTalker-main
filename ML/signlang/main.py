import json

import cv2
import mediapipe as mp
import numpy as np
import os
import base64
import asyncio
import websockets
from matplotlib import pyplot as plt
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical  # 원핫인코딩으로 변경
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
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
    image.flags.writeable = False  # 이미지 수정 불가
    results = model.process(image)  # 모델을 사용해 입력 이미지에 대한 예측 수행
    image.flags.writeable = True  # 이미지 다시 수정가능
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
    return image, results




script_directory = os.path.dirname(os.path.abspath(__file__))
dataset_directory = os.path.join(script_directory, "dataset")
print("현재 작업 디렉토리:", script_directory)

MODEL_PATH = os.path.join(script_directory,'model_ko.h5')
model = load_model(MODEL_PATH)

data_file_list = os.listdir(dataset_directory)
data_file_list = sorted(data_file_list, reverse=True)

actions = {}
for file_name in data_file_list:
    parts = file_name.split("_")
    if parts[1] not in actions:
        actions[int(parts[0])] = parts[1]
        # actions.append((parts[1], parts[0])) # 파일 목록(npy)에서 단어 추출

# actions 콘솔 출력
# for action, label in actions:
#     print(action, ':', label)
print(actions)


sentence_length = 10
seq_length = 30

extra_time = 2.5 # 다음 동작 전달까지 텀


async def handle_client(websocket, path):
    try:
        # sentence = []  # 현재 디버깅 용, 감지 단어 리스트 콘솔 출력
        seq = []
        action_seq = [] 
        previous = '' # 이전 단어

        dc=0 # debug_count

        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands( 
        max_num_hands=2,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4)

        extra_time_start = time.time() # for extra_time

        while True:
            message = await websocket.recv()
            # print("debug: message")
            if message != "":

                # 프레임(이미지) 전처리
                frame = process_frame(message)

                # mediapipe 감지
                image, result = mediapipe_detection(frame, hands)
                # print(f"result: {results}")

            if result.multi_hand_landmarks is not None:
                for res in result.multi_hand_landmarks: # 감지된 손의 수만큼 반복
                    joint = np.zeros((21, 4))
                    for j, lm in enumerate(res.landmark):
                        joint[j] = [lm.x, lm.y, lm.z, lm.visibility] # visibility: 신뢰도 (0~1)

                    # 각 관절의 벡터 계산
                    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
                    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
                    v = v2 - v1 # [20, 3]
                    # 정규화
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                    # 내적 arcos으로 각도 계산
                    angle = np.arccos(np.einsum('nt,nt->n',
                        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                    angle = np.degrees(angle) # Convert radian to degree

                    d = np.concatenate([joint.flatten(), angle])

                    seq.append(d)
                    # print("debug2", len(seq))


                    dc+=1 
                    print(dc, "debug1.seq크기:", len(seq), end = "__")

                    if len(seq) < seq_length: # 시퀀스 최소치가 쌓인 이후부터 판별
                        continue


                    if len(seq)>seq_length*100:  # 과도하게 쌓임 방지
                        seq=seq[-seq_length:]
                    
                    # 시퀀스 데이터를 신경망 모델에 입력으로 사용할 수 있는 형태로 변환
                    input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)

                    y_pred = model.predict(input_data).squeeze() # 각 동작에 대한 예측 결과 (각각의 확률)

                    i_pred = int(np.argmax(y_pred)) # 최댓값 인덱스: 예측값이 가장 높은 값(동작)의 인덱스
                    conf = y_pred[i_pred] # 가장 확률 높은 동작의 확률이

                    if conf < 0.9:   # 90% 이상일 때만 수행
                        continue

                    # debug
                    print("debug2.예측동작인덱스:", i_pred)


                    action = actions[i_pred]
                    ####
                    action_seq.append(action)

                    if len(action_seq) < 3:
                        continue

                    this_action = ''
                    if action_seq[-1] == action_seq[-2] == action_seq[-3]:
                        this_action = action
                    ####
                    # debug
                    print(dc, "debug3.예측동작(출력동작):", this_action)

                    seq=[]
                    
                    
                    # print("DEBUG", this_action)
                    # if this_action !='':      
                    #     sentence.append(this_action)
                    # if len(sentence) > sentence_length:
                    #     sentence = sentence[-sentence_length:]
                        # print(' '.join(sentence))

                    print("send?1")
                    # if time.time() - extra_time_start < extra_time or action=='':  # 데이터 전달 최소 텀
                    if time.time() - extra_time_start < extra_time or this_action=='':  # 데이터 전달 최소 텀
                        continue
                    extra_time_start = time.time()
                    print("send?2")


                    # print("왜멈춰?1 ", action)
                    # print("왜멈춰?2 ", previous, action, previous==action)

                    if previous == this_action: continue  # 중복 전달 회피  ???
                    previous = this_action
                    # if previous == action: continue  # 중복 전달 회피  ???
                    # previous = action
                    print("보냄?3")


                    result_dict = {'result': this_action}
                    # result_dict = {'result': action}
                    result_json = json.dumps(result_dict)

                    try:
                        if websocket.open:
                            await websocket.send(result_json)
                    except Exception as e:
                        print(f"send error: {str(e)}")
                    
    except websockets.exceptions.ConnectionClosedOK:
        pass


start_server = websockets.serve(handle_client, "localhost", 8080)


async def main():
    await start_server


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
    asyncio.get_event_loop().run_forever()