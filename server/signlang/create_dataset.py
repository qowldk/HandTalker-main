import cv2
import mediapipe as mp
import numpy as np
import time, os

actions = '약'
seq_length = 30
secs_for_action = 30

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

created_time = int(time.time())
os.makedirs('dataset', exist_ok=True)

while cap.isOpened():
    user_idx = int(input("원하시는 인덱스를 입력해주세요: "))
    action = actions
    data = []

    ret, img = cap.read()
    img = cv2.flip(img, 1)

    # Draw a black dot at the center of the screen
    height, width, _ = img.shape
    center_x, center_y = width // 2, height // 2
    cv2.circle(img, (center_x, center_y), 5, (0, 0, 0), -1)

    cv2.putText(img, f'Waiting for collecting {action.upper()} action...', org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
    cv2.imshow('img', img)
    cv2.waitKey(3000)

    start_time = time.time()

    while time.time() - start_time < secs_for_action:
        ret, img = cap.read()

        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks is not None:
            h = 0
            d1 = np.empty(0)
            d2 = np.empty(0)
            for res in result.multi_hand_landmarks:
                h+=1
                joint = np.zeros((23, 3))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z]
                joint[21] = [0, 0, 0]
                joint[22] = [1, 1, 1]
                # Compute angles between joints
                v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19,21], :3] # Parent joint
                v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22], :3] # Child joint
                v = v2 - v1 # [20, 3]
                # Normalize v
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                # Get angle using arcos of dot product
                angle = np.arccos(np.einsum('nt,nt->n',
                    v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,0,16],:], 
                    v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19,20,20],:])) # [15,]

                angle = np.degrees(angle) # Convert radian to degree

                angle_label = np.array([angle], dtype=np.float32)
                if h==1:
                    d1 = np.concatenate([angle_label[0]])
                else:
                    d2 = np.concatenate([angle_label[0], [user_idx]])

                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            d=np.concatenate([d1, d2])
            if len(d)<=17:
                d=np.concatenate([d, np.zeros(len(d)), [user_idx]])
            data.append(d)
            print(data)

        # Draw a black dot at the center of the screen
        cv2.circle(img, (center_x, center_y), 5, (0, 0, 0), -1)

        cv2.imshow('img', img)
        if cv2.waitKey(1) == ord('q'):
            break

    # Save data
    file_name = str(user_idx) + '_' + str(action) + '_s_0.npy'
    script_directory = os.path.dirname(os.path.abspath(__file__))
    save_data = os.path.join(script_directory, "dataset", file_name)
    ######
    save_file_num = 0
    while os.path.exists(save_data):
        save_file_num += 1
        length_except_num = len(save_data.split('_')[-1])
        save_data = save_data[:-length_except_num] # 'label_action_'
        save_data += str(save_file_num) + '.npy'

    # Create sequence data
    full_seq_data = []
    for seq in range(len(data) - seq_length):
        full_seq_data.append(data[seq:seq + seq_length])

    full_seq_data = np.array(full_seq_data)
    print(action, full_seq_data.shape)

    # Save the sequence data
    np.save(save_data, full_seq_data)
    file_name = str(user_idx) + '_' + str(action) + '_s_'+str(save_file_num)+'.npy'
    frame_data = os.path.join(script_directory, "dataset_frame", file_name)
    break

cap.release()
cv2.destroyAllWindows()
