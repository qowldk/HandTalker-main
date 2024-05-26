import cv2
import mediapipe as mp
import numpy as np
import os
"""
웹캠에서 손을 감지하고, 손의 각도를 계산하여 그 값을 파일에 저장
"""
input_announce="""
l 라벨 변경
. 데이터 저장 

(된소리 = 예사소리 * 2)
0, 1: 'ㄱ', 2, 3: 'ㄴ', 4, 5: 'ㄷ', 6, 7: 'ㄹ', 8, 9: 'ㅁ', 10, 11: 'ㅂ', 12, 13: 'ㅅ', 14, 15: 'ㅇ',
16, 17: 'ㅈ', 18, 19: 'ㅊ', 20, 21: 'ㅋ', 22, 23: 'ㅌ', 24, 25: 'ㅍ', 26, 27: 'ㅎ', 
28: 'ㅏ', 29: 'ㅑ', 30: 'ㅓ', 31: 'ㅕ', 32: 'ㅗ', 33: 'ㅛ', 34: 'ㅜ', 35: 'ㅠ',
36: 'ㅡ', 37: 'ㅣ', 38: 'ㅐ', 39: 'ㅒ', 40: 'ㅔ', 41: 'ㅖ', 
42: 'ㅚ', 43: 'ㅟ', 44: 'ㅢ'

라벨(숫자)을 입력하세요: """
def calculate_angle(joint):
    v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19, 21], :]
    v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22], :]
    v = v2 - v1
    # Nomalize v
    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

    # Dot product의 아크코사인으로 각도를 구한다.
    compareV1 = v[[0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 16, 17, 0, 16], :]
    compareV2 = v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 20, 20], :]

    angle = np.arccos(np.einsum('nt,nt->n', compareV1, compareV2))
    angle = np.degrees(angle)  # radian값을 degree로 변환

    return angle

cap = cv2.VideoCapture(0)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

label=None
# 사진 띄우기?
with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.\n웹캠을 사용중인 프로세스를 중지해주세요.")
            continue

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for res in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, res, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('MediaPipe Hands', image)


        key = cv2.waitKey(1)
        if key == ord('l'):
            # 라벨 입력 받기
            label = input(input_announce)
        if key == ord('.'):
            if label is None:
                print("라벨을 입력해주세요.(l)")
                continue
            if results.multi_hand_landmarks:
                res=None
                for res_ in results.multi_hand_landmarks:
                    res=res_
                
                joint = np.zeros((23, 3))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z]
                joint[21] = [0,0,0]
                joint[22] = [1,1,1]
                
                angle = calculate_angle(joint)
                d = np.concatenate([joint.flatten(), angle])
                print(d.shape)
                d = d.tolist()
                # 라벨과 각도 값을 텍스트 파일에 저장
                script_directory = os.path.dirname(os.path.abspath(__file__))
                PATH = os.path.join(script_directory, 'dataset_ko.txt')
                with open(PATH, 'a') as file:
                    file.write(f"{','.join(str(a) for a in d)},{label}\n")
                print("saved:", label)
            else:
                print("감지된 손이 없습니다.")

        elif key == 27 or key=='q':  # ESC 키를 누르면 루프 종료
            break

cap.release()
cv2.destroyAllWindows()