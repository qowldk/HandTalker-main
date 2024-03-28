import cv2
import mediapipe as mp
import numpy as np
import time, os

seq_length = 30


secs_for_action = 30 # 초
time_to_start = 3 # 초

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

action, idx = None, None
stop_=False

a=0 # frame debug

print(f'1회 데이터 입력 시간: {secs_for_action}초')
print(f'데이터 입력 시작 시 딜레이: {time_to_start}초')

anounce_for_user = '''
웹캠 화면에서 메뉴 선택
l 데이터를 쌓을 수어의 한글 단어 입력
. 수어 데이터 입력 시작
t 데이터 입력 시간 변경(default: 30s)
y 데이터 입력 준비 시간 변경(defalut: 3s)
ESC 종료
'''
print(anounce_for_user)

while cap.isOpened():
    ###
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
    cv2.imshow('img', image)


    key = cv2.waitKey(1)
    if key == ord('l'):
        # 단어, 라벨 입력
        action = input('단어 입력: ')
        while True:
            idx = input('라벨 입력: ')
            try:
                int_value = int(idx)
                break
            except ValueError:
                print("정수 값을 입력해주세요.")
        print(f'({action}, {idx}) 입력 완료')
        # print(anounce_for_user)
    
    if key == ord('t'):
        while True:
            secs_for_action = input(f'{secs_for_action} -> ')
            try:
                float_value = float(secs_for_action)
                break
            except ValueError:
                print("실수 값을 입력해주세요.")            
        print('데이터 입력 시간 변경 완료')

    if key == ord('y'):
        while True:
            time_to_start = input(f'{time_to_start} -> ')
            try:
                float_value = float(time_to_start)
                break
            except ValueError:
                print("실수 값을 입력해주세요.")
        print('데이터 입력 준비 시간 변경 완료')
        
    if key == ord('.'):
        if action is None or idx is None:
            print("l을 눌러 입력할 단어를 설정해주세요.")
            continue
        print(f'({action}, {idx}): {secs_for_action}초간 데이터 생성을 {time_to_start}초 뒤 시작합니다.')
        print('q: 중단(준비 시간 이후 중단 가능)')
        data = []
        ###
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        cv2.putText(img, f'Ready...', org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,0), thickness=2)
        cv2.imshow('img', img)
        cv2.waitKey(int(time_to_start*1000)) 

        start_time = time.time()

        while time.time() - start_time < secs_for_action:
            a+=1
            ret, img = cap.read()

            img = cv2.flip(img, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if result.multi_hand_landmarks is not None:
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
                # print(d[-1], end=' ')
                if len(d)==99:
                    d=np.concatenate([d, np.zeros(99), [idx]])
                    # print(d)
                data.append(d)
                
            cv2.imshow('img', img)
            if cv2.waitKey(1) == ord('q'):
                stop_=True
                break
        if stop_:
            stop_=False
            print("데이터 생성 중단")
            continue
        print("frame: ", a)
        ###
        data = np.array(data)
        print(action, data.shape) #debug
        

        if len(data) - seq_length < 1:
            print("프레임이 너무 적어 데이터 생성에 실패했습니다.")
            continue

        # 시퀀스 데이터 생성
        full_seq_data = []
        for seq in range(len(data) - seq_length):
            full_seq_data.append(data[seq:seq + seq_length])

        # list -> numpy
        full_seq_data = np.array(full_seq_data)
        print(action, full_seq_data.shape) # debug

        # 저장할 npy 파일 이름
        # save_file_num = input('파일 번호 입력')
        file_name = str(idx) + '_' + str(action) + '_0.npy'
        
        # 저장
        script_directory = os.path.dirname(os.path.abspath(__file__))
        save_data = os.path.join(script_directory, "dataset", file_name)
        ##
        save_file_num = 0
        while os.path.exists(save_data):
            save_file_num += 1
            length_except_num = len(save_data.split('_')[-1])
            save_data = save_data[:-length_except_num] # 'label_action_'
            save_data += str(save_file_num) + '.npy'
        ##
        np.save(save_data, full_seq_data)        

        # 프레임 단위 데이터 저장 디렉토리
        file_name = str(idx) + '_' + str(action) + '_'+str(save_file_num)+'.npy'
        frame_data = os.path.join(script_directory, "dataset_frame", file_name) # 프레임 데이터
        np.save(frame_data, data)   # 저장

        print(f'({action}, {idx}):', data.shape, full_seq_data.shape, '데이터 생성 완료')

        
        

    if key == 27:  # ESC 키를 누르면 루프 종료
        break
    ###







