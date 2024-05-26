import cv2
import numpy as np
import json
import asyncio
import websockets
import os
import base64
import mediapipe as mp



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
DATA_PATH = os.path.join(script_directory,'dataset_ko.txt')

file = np.genfromtxt(DATA_PATH, delimiter=',')
angleFile = file[:, :-1]
labelFile = file[:, -1]
angle = angleFile.astype(np.float32)
label = labelFile.astype(np.float32)

knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)


actions = {
    # 0: 'ㄱ', 1: 'ㄱ', 2: 'ㄴ', 3: 'ㄴ', 4: 'ㄷ', 5: 'ㄷ', 6: 'ㄹ', 7: 'ㄹ', 8: 'ㅁ', 9: 'ㅁ',
    # 10: 'ㅂ', 11: 'ㅂ', 12: 'ㅅ', 13: 'ㅅ', 14: 'ㅇ', 15: 'ㅇ', 16: 'ㅈ', 17: 'ㅈ', 18: 'ㅊ',
    # 19: 'ㅊ', 20: 'ㅋ', 21: 'ㅋ', 22: 'ㅌ', 23: 'ㅌ', 24: 'ㅍ', 25: 'ㅍ', 26: 'ㅎ', 27: 'ㅎ',
    # 28: 'ㅏ', 29: 'ㅑ', 30: 'ㅓ', 31: 'ㅕ', 32: 'ㅗ', 33: 'ㅛ', 34: 'ㅜ', 35: 'ㅠ', 36: 'ㅡ',
    # 37: 'ㅣ', 38: 'ㅐ', 39: 'ㅒ', 40: 'ㅔ', 41: 'ㅖ', 42: 'ㅚ', 43: 'ㅟ', 44: 'ㅢ'

    0: 'ㄱ', 1: 'ㄴ', 2: 'ㄷ', 3: 'ㄹ', 4: 'ㅁ', 5: 'ㅂ', 6: 'ㅅ', 7: 'ㅇ', 8: 'ㅈ', 9: 'ㅊ',
    10: 'ㅋ', 11: 'ㅌ', 12: 'ㅍ', 13: 'ㅎ',
    14: 'ㅏ', 15: 'ㅑ', 16: 'ㅓ', 17: 'ㅕ', 18: 'ㅗ', 19: 'ㅛ', 20: 'ㅜ', 21: 'ㅠ', 22: 'ㅡ',
    23: 'ㅣ', 24: 'ㅐ', 25: 'ㅒ', 26: 'ㅔ', 27: 'ㅖ', 28: 'ㅚ', 29: 'ㅟ', 30: 'ㅢ'
}


async def finger_spell(websocket, path):
    try:
        previous = '' # 이전 문자

        dc=0 # debug_count

        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands( 
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

        while True:
            message = await websocket.recv()
            #####
            if message != "":
                # 프레임(이미지) 전처리
                frame = process_frame(message)
                # mediapipe 감지
                image, result = mediapipe_detection(frame, hands)
                # print(f"result: {results}")
                
            if result.multi_hand_landmarks is not None:
                #h = 0 # 손 두개 감지 로직을 위한 임시 값

                for res in result.multi_hand_landmarks: # 감지된 손의 수만큼 반복
                    #if h==1: break
                    #h+=1
                    joint = np.zeros((23, 3))
                    for j, lm in enumerate(res.landmark):
                        joint[j] = [lm.x, lm.y, lm.z] # visibility: 신뢰도 (0~1)
                    joint[21] = [0,0,0]
                    joint[22] = [1,1,1]
                    # 각 관절의 벡터 계산
                    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19,21], :3] # Parent joint
                    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22], :3] # Child joint
                    v = v2 - v1 # [20, 3]
                    # 정규화 (크기 1의 단위벡터로)
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                    # 내적의 arcos으로 각도 계산
                    ### 벡터 a 와 벡터 b 를 내적(dot product)하면 [벡터 a의 크기] × [벡터 b의 크기] × [두 벡터가 이루는 각의 cos값] 이 된다.
                    ### 그런데 바로 위에서 벡터들의 크기를 모두 1로 표준화시켰으므로 두 벡터의 내적값은 곧 [두 벡터가 이루는 각의 cos값]이 된다.
                    ### 따라서 이것을 코사인 역함수인 arccos에 대입하면 두 벡터가 이루는 각이 나오게 된다.
                    angle = np.arccos(np.einsum('nt,nt->n',
                        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,0,16],:], 
                        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19,20,20],:]))

                    angle = np.degrees(angle) # 라디안 -> 도
                    angle_label = np.array([angle], dtype=np.float32)
                    d = np.concatenate([joint.flatten(), angle_label[0]]).tolist()
                

                data = np.array([d], dtype=np.float32)
                ret, results, neighbours, dist = knn.findNearest(data, 3)
                predicted_idx = int(results[0][0])

                predicted_gesture = actions.get(predicted_idx, "Unknown Gesture")
                if previous == predicted_gesture: continue  # 중복 전달 회피  ???
                previous = predicted_gesture
                result_dict = {'result': predicted_gesture}

                result_json = json.dumps(result_dict)
                print(result_json)
                try:
                    if websocket.open:
                        print("sended!")
                        await websocket.send(result_json)
                except Exception as e:
                    print(f"send error: {str(e)}")

    except websockets.exceptions.ConnectionClosedOK:
        pass


start_server = websockets.serve(finger_spell, "localhost", 8081)  


async def main():
    await start_server


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
    asyncio.get_event_loop().run_forever()