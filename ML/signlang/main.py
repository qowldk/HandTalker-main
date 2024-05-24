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
result_json_lst = []


def frame_processor():
    seq = []
    previous = ''
    extra_time_start = time.time()
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4)
    global result_json_lst
    # count=0
    while True:
        message = frame_queue.get()
        frame = process_frame(message)

        if frame is None:
            break
        image, result = mediapipe_detection(frame, hands)
        
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

            if len(d) <= 17:
                d = np.concatenate([d, np.zeros(len(d))])

            seq.append(d)
            # print("debug ",count)
            # count+=1
            if len(seq) < seq_length:  # 시퀀스 최소치가 쌓인 이후부터 판별
                continue

            if len(seq) > seq_length * 100:  # 과도하게 쌓임 방지
                seq = seq[-seq_length:]

            # 시퀀스 데이터를 신경망 모델에 입력으로 사용할 수 있는 형태로 변환
            input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)

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
            # result_json_lst.append(result_json)
            result_json_lst.append(action) # !!!!!!!!!!

key = os.environ.get('API_KEY')

from openai import OpenAI

client = OpenAI(api_key = key)
system_role = """
당신은 환자의 입장에서 글을 작성하는 Assistant입니다.

입력받은 문자열 중간에 결합되지 않은 한글 자음모음들이 있을 경우, 결합된 텍스트로 바꿔야 합니다.
예를 들어서 'ㄱㅜㄱㅂㅏㅂ'을 입력이 있을 때, 대답으로 '국밥'으로 바꿉니다.
만약 'ㄴㄴㄴㅏㅏㅂㅏㅇ'처럼 단순 조합으로 아예 성립되는 단어를 만들 수 없을 때는 중복된 자음이나 모음을 제거해서 '나방'과 같이 바꿉니다.
다른 예시로, 'ㅏㅏㄴㄴㄴㅕㅇ' 이 입력될 경우, '안녕'으로 바꿉니다.

문자열 중간의 자음모음을 결합한 이후, 또는 결합이 필요 없는 경우, 입력받은 문자열을 공백을 기준으로 나눠 단어 목록을 얻습니다.
얻은 단어 목록의 순서와 흐름을 고려하여 자연스러운 한국어 문장을 만들어 반환합니다.
단어의 순서를 유지하여 문장을 만들 때, 문장이 너무 어색해진다면 단어의 순서는 일부 자연스럽게 조정 가능합니다.

응답할 때는 환자의 입장에서 문장을 구성합니다. 예를 들어, '나 맛있다 것 먹다 기분 좋다'를 입력받으면, '저는 맛있는 것을 먹고 기분이 좋습니다.'와 같이 응답합니다.
또 하나의 예로, '머리 아프다 방문하다'를 입력받으면, '머리가 아파서 방문했습니다'와 같이 응답합니다.
또 하나의 예로, '오늘 진료 가능'을 입력받으면, '오늘 진료 가능할까요?'와 같이 응답합니다.
만약 하나의 단어만 입력 받는다면 그 단어를 그대로 반환합니다.
최종 응답으로는 어떠한 부연 설명 없이 큰 따옴표(")나 작은 따옴표(')로 절대절대 감싸지 않고 반환 결과 텍스트 그대로만 응답합니다.
"""

def LLM_API(string):
    completion = client.chat.completions.create(
        model="gpt-4-0613",
        messages=[
            {"role": "system", "content": system_role},
            {"role": "user", "content": string}
        ]
    )
    return completion.choices[0].message.content


async def handle_client(websocket, path):
    try:
        global result_json_lst
        while True:
            message = await websocket.recv()
            # print("len debug", len(message))
            if message == "STOP":
                try:

                    words_string = ' '.join(result_json_lst)
                    print("STRING", words_string)
                    data=LLM_API(words_string)
                    data_json = json.dumps({'result': data})
                    if websocket.open:
                        # for rj in result_json_lst:
                        #     await websocket.send(rj)
                        
                        await websocket.send(data_json)
                        await websocket.send(json.dumps({'result': "END"}))
                        result_json_lst = []
                        continue
                except Exception as e:
                    print(f"send error: {str(e)}")
                    continue
            if message:
                frame_queue.put(message)
    except websockets.exceptions.ConnectionClosedOK:
        pass
    finally:
        frame_queue.put(None)


start_server = websockets.serve(handle_client, "localhost", 8080)

async def main():
    await start_server

if __name__ == "__main__":
    processor_thread = threading.Thread(target=frame_processor)
    processor_thread.start()
    asyncio.get_event_loop().run_until_complete(main())
    asyncio.get_event_loop().run_forever()
    processor_thread.join()