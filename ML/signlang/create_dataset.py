import cv2
import mediapipe as mp
import numpy as np
import time, os

actions = [
    ('안녕하세요', 0),
    ('만나다', 1),
    ('반갑습니다', 2)
]
# labels = [
#     0,
#     1,
#     2
# ]

seq_length = 30
secs_for_action = 2 # 1.7
repeat = 2 # 같은 동작 몇 번 학습 횟수

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4)

cap = cv2.VideoCapture(0)

created_time = int(time.time())
os.makedirs('dataset', exist_ok=True)
a=0 # frame debug
while cap.isOpened():
    for action, idx  in actions:
        data = []

        for r in range(repeat):  # 같은 동작 몇번 학습할 것인지
            ret, img = cap.read()
            img = cv2.flip(img, 1)
            # cv2.putText(img, f'Waiting for collecting {action.upper()} action...', org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
            print('debug', idx, action)
            cv2.imshow('img', img)
            cv2.waitKey(3500 if r==0 else 800) # 밀리초 단위

            start_time = time.time()

            while time.time() - start_time < secs_for_action:
                a+=1
                ret, img = cap.read()

                img = cv2.flip(img, 1)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result = hands.process(img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                if result.multi_hand_landmarks is not None:
                    for res in result.multi_hand_landmarks:
                        joint = np.zeros((21, 4))
                        for j, lm in enumerate(res.landmark):
                            joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                        # Compute angles between joints
                        v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
                        v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
                        v = v2 - v1 # [20, 3]
                        # Normalize v
                        v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                        # Get angle using arcos of dot product
                        angle = np.arccos(np.einsum('nt,nt->n',
                            v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                            v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                        angle = np.degrees(angle) # Convert radian to degree

                        angle_label = np.array([angle], dtype=np.float32)
                        angle_label = np.append(angle_label, idx)

                        d = np.concatenate([joint.flatten(), angle_label])

                        data.append(d)

                        mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

                cv2.imshow('img', img)
                if cv2.waitKey(1) == ord('q'):
                    break
            print("frame: ", a)
        data = np.array(data)
        print(action, data.shape)
        # file_name = 'C:\\Users\\user\\Downloads\\HandTalker-main_\\ML\\signlang\\dataset\\raw_' + str(action) + '_' + str(created_time)
        # np.save(file_name, data)
        
        # Create sequence data
        full_seq_data = []
        for seq in range(len(data) - seq_length):
            full_seq_data.append(data[seq:seq + seq_length])

        full_seq_data = np.array(full_seq_data)
        print(action, full_seq_data.shape)


        same_file_num = ''
        file_name = 'C:\\Users\\user\\Downloads\\HandTalker-main_\\ML\\signlang\\dataset\\' + str(idx) + '_' + str(action) + '_' + same_file_num
        # while os.path.exists(file_name):
        #     file_name = file_name[:-1]
        #     same_file_num += 1
        #     file_name += str(same_file_num)
        np.save(file_name, full_seq_data)        

    break
