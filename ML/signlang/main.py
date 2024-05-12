import json

import cv2
import mediapipe as mp
import numpy as np
import os
import base64
import asyncio
import websockets
import time
from tensorflow.keras.models import load_model

import hand

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
dataset_directory = os.path.join(script_directory, "dataset")
print("현재 작업 디렉토리:", script_directory)

MODEL_PATH = os.path.join(script_directory,'model_ko.h5')
model = load_model(MODEL_PATH, compile=False) # 코랩 사용시 compile=False 필수

data_file_list = os.listdir(dataset_directory)
data_file_list = sorted(data_file_list, reverse=True)

actions = {}
for file_name in data_file_list:
    parts = file_name.split("_")
    if parts[1] not in actions:
        actions[int(parts[0])] = parts[1]

print(actions)


sentence_length = 10
seq_length = 30



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


        tmp = []

        while True:
            message = await websocket.recv()
            # print('오냐?')
            # print("debug: message")
            if message != "":
                # 프레임(이미지) 전처리
                frame = process_frame(message)
                # mediapipe 감지
                image, result = mediapipe_detection(frame, hands)
                # print(f"result: {results}")
                
            if result.multi_hand_landmarks is not None:
                h = 0 # 손 두개 감지 로직을 위한 임시 값
                resource1 = np.zeros((21, 3))
                resource2 = np.zeros((21, 3))

                for res in result.multi_hand_landmarks: # 감지된 손의 수만큼 반복
                    h+=1
                    joint = np.zeros((21, 3))
                    for j, lm in enumerate(res.landmark):
                        joint[j] = [lm.x, lm.y, lm.z]

                    # # 각 관절의 벡터 계산
                    # v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
                    # v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
                    # v = v2 - v1 # [20, 3]
                    # # 정규화 (크기 1의 단위벡터로)
                    # v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                    # # 내적의 arcos으로 각도 계산
                    # ### 벡터 a 와 벡터 b 를 내적(dot product)하면 [벡터 a의 크기] × [벡터 b의 크기] × [두 벡터가 이루는 각의 cos값] 이 된다.
                    # ### 그런데 바로 위에서 벡터들의 크기를 모두 1로 표준화시켰으므로 두 벡터의 내적값은 곧 [두 벡터가 이루는 각의 cos값]이 된다.
                    # ### 따라서 이것을 코사인 역함수인 arccos에 대입하면 두 벡터가 이루는 각이 나오게 된다.
                    # angle = np.arccos(np.einsum('nt,nt->n',
                    #     v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                    #     v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                    # angle = np.degrees(angle) # 라디안 -> 도
                    # angle_label = np.array([angle], dtype=np.float32)

                    if h==1:
                        resource1 = joint
                    else:
                        resource2 = joint
                
                rate_resource = np.concatenate((resource1, resource2))
                # print(rate_resource.shape)
                rate_resource = rate_resource.tolist()
                # if len(rate_resource) != 42:
                #     print("what?")
                #     continue
                seq.append(rate_resource)

                dc+=1 
                print(dc, "debug1:", len(seq), np.array(seq).shape)

                if len(seq) < seq_length: # 시퀀스 최소치가 쌓인 이후부터 판별
                    continue
                seq=seq[-seq_length:]

                # print("normalization legnth", len(seq), np.array(seq).shape)
                normalized_seq = hand.normalization(seq)

                # print("DEBUG1", np.array(normalized_seq).shape)
                normalized_seq = np.array(normalized_seq)
                seq_=[]
                for i in range(30):
                    seq_.append(normalized_seq[i].flatten())
                normalized_seq = seq_
                # print("DEBUG1.1", np.array(seq_).shape)
                # 시퀀스 데이터를 신경망 모델에 입력으로 사용할 수 있는 형태로 변환
                input_data = np.expand_dims(np.array(normalized_seq[-seq_length:], dtype=np.float32), axis=0)
                # print("DEBUG1.2", input_data.shape)

                # 에러?
                y_pred = model.predict(input_data).squeeze() # 각 동작에 대한 예측 결과 (각각의 확률)
                # print("DEBUG2")

                i_pred = int(np.argmax(y_pred)) # 최댓값 인덱스: 예측값이 가장 높은 값(동작)의 인덱스
                conf = y_pred[i_pred] # 가장 확률 높은 동작의 확률이

                print("conf debug", conf)
                if conf < 0.8:   # 90% 이상일 때만 수행
                    continue

                # debug
                print("debug2 :", i_pred)


                action = actions[i_pred]
                
                if previous == action: continue  # 중복 전달 회피
                previous = action

                seq=[]


                result_dict = {'result': action}
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