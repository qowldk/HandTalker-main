import cv2
import numpy as np
import json
import asyncio
import websockets
import os

script_directory = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(script_directory,'dataSet_ko.txt')

file = np.genfromtxt(DATA_PATH, delimiter=',')
angleFile = file[:, :-1]
labelFile = file[:, -1]
angle = angleFile.astype(np.float32)
label = labelFile.astype(np.float32)

knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)


async def handle_client(websocket, path):
    try:
        while True:
            message = await websocket.recv()
            print(message)
            if len(message) == 0 or message == "undefined":
                continue
            data = [json.loads(message)]
            numpy_array = np.array([[item["x"], item["y"], item["z"]] for item in data[0]], dtype=np.float32)

            num_joints = 21
            sample_input = numpy_array

            v1 = sample_input[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]
            v2 = sample_input[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]
            v = v2 - v1
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            compareV1 = v[[0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 16, 17], :]
            compareV2 = v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]
            angles = np.arccos(np.einsum('nt,nt->n', compareV1, compareV2))
            angles = np.degrees(angles)

            data = np.array([angles], dtype=np.float32)
            ret, results, neighbours, dist = knn.findNearest(data, 3)
            predicted_idx = int(results[0][0])

            gesture_mapping = { # 쌍자음은 2회씩
                0: 'ㄱ', 1: 'ㄴ', 2: 'ㄷ', 3: 'ㄹ', 4: 'ㅁ', 5: 'ㅂ', 6: 'ㅅ', 7: 'ㅇ',
                8: 'ㅈ', 9: 'ㅊ', 10: 'ㅋ', 11: 'ㅌ', 12: 'ㅍ', 13: 'ㅎ', 
                14: 'ㅏ', 15: 'ㅑ', 16: 'ㅓ', 17: 'ㅕ', 18: 'ㅗ', 19: 'ㅛ', 20: 'ㅜ', 21: 'ㅠ',
                22: 'ㅡ', 23: 'ㅣ', 24: 'ㅐ', 25: 'ㅒ', 26: 'ㅔ', 27: 'ㅖ', 
                28: 'ㅢ', 29: 'ㅚ', 30: 'ㅟ', 
            } # 숫자는 보류
            # 데이터가 많지 않고, 팔의 각도까지 계산하지 않기에 비슷한 문자 구별에는 어려움이 있음
            # https://www.youtube.com/watch?v=0eTc8GPMv74
            # https://www.youtube.com/watch?v=8DUbB62294E

            # gesture_mapping = { # en
            #     0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h',
            #     8: 'i', 9: 'j', 10: 'k', 11: 'l', 12: 'm', 13: 'n', 14: 'o',
            #     15: 'p', 16: 'q', 17: 'r', 18: 's', 19: 't', 20: 'u', 21: 'v',
            #     22: 'w', 23: 'x', 24: 'y', 25: 'z', 26: 'spacing', 27: 'backspace', 28: '1',
            #     29: '2', 30: '3', 31: '4', 32: '5', 33: '6', 34: '7', 35: '8', 36: '9', 37: 'b'
            # }

            predicted_gesture = gesture_mapping.get(predicted_idx, "Unknown Gesture")
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


start_server = websockets.serve(handle_client, "localhost", 8081)  


async def main():
    await start_server


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
    asyncio.get_event_loop().run_forever()
