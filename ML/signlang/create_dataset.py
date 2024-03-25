import cv2
import mediapipe as mp
import numpy as np
import time, os

actions = [
    ('안녕하세요', 0),
    ('만나다', 1),
    ('반갑다', 2)
]

seq_length = 30
secs_for_action = 20 # 1.7
repeat = 1 # 같은 동작 몇 번 학습 횟수

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
            cv2.waitKey(2000 if r==0 else 800) # 밀리초 단위

            start_time = time.time()

            while time.time() - start_time < secs_for_action:
                a+=1
                ret, img = cap.read()

                img = cv2.flip(img, 1)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result = hands.process(img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                if result.multi_hand_landmarks is None: continue
                if len(result.multi_hand_landmarks) == 2:                    
                    h = 0 # 손 두개 감지 로직을 위한 임시 값
                    d1 = np.empty(0)
                    d2 = np.empty(0)
                    for res in result.multi_hand_landmarks:
                        h+=1
                        joint = np.zeros((21, 4))
                        for j, lm in enumerate(res.landmark):
                            joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                        # 각 관절의 벡터 계산
                        v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
                        v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
                        v = v2 - v1 # [20, 3]
                        # 정규화 (크기 1의 단위벡터로)
                        v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                        # 내적의 arcos으로 각도 계산
                        ### 벡터 a 와 벡터 b 를 내적(dot product)하면 [벡터 a의 크기] × [벡터 b의 크기] × [두 벡터가 이루는 각의 cos값] 이 된다.
                        ### 그런데 바로 위에서 벡터들의 크기를 모두 1로 표준화시켰으므로 두 벡터의 내적값은 곧 [두 벡터가 이루는 각의 cos값]이 된다.
                        ### 따라서 이것을 코사인 역함수인 arccos에 대입하면 두 벡터가 이루는 각이 나오게 된다.                        
                        angle = np.arccos(np.einsum('nt,nt->n',
                            v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                            v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                        angle = np.degrees(angle) # 라디안 -> 도
                        angle_label = np.array([angle], dtype=np.float32)
                        if h==1:
                            d1 = np.concatenate([joint.flatten(), angle_label[0]])
                        else:
                            d2 = np.concatenate([joint.flatten(), angle_label[0], [idx]])

                        # 파이썬 실행 화면(웹캠)에 랜드마크 그림
                        mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)
                    
                    d=np.concatenate([d1, d2])
                    print(d[-1], end=' ')
                    data.append(d)

                cv2.imshow('img', img)
                if cv2.waitKey(1) == ord('q'):
                    break
            print("frame: ", a)
        data = np.array(data)
        print(action, data.shape)
        # file_name = 'C:\\Users\\user\\Downloads\\HandTalker-main_\\ML\\signlang\\dataset\\raw_' + str(action) + '_' + str(created_time)
        # np.save(file_name, data)
        
        if len(data) - seq_length < 5:
            print("2개의 손이 감지된 프레임이 너무 적어 데이터 생성에 실패했습니다.")
            continue
        # 시퀀스 데이터 생성
        full_seq_data = []
        for seq in range(len(data) - seq_length):
            full_seq_data.append(data[seq:seq + seq_length])

        full_seq_data = np.array(full_seq_data)
        print(action, full_seq_data.shape)


        script_directory = os.path.dirname(os.path.abspath(__file__))
        dataset_directory = os.path.join(script_directory, "dataset")

        same_file_num = '1'
        file_name = str(idx) + '_' + str(action) + '_' + same_file_num

        save_data = os.path.join(dataset_directory, file_name)

        # while os.path.exists(file_name):
        #     file_name = file_name[:-1]
        #     same_file_num += 1
        #     file_name += str(same_file_num)
        np.save(save_data, full_seq_data)        

    break
