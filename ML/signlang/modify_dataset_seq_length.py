import os
import numpy as np

seq_length = 30 # 저장할 시퀀스 데이터 길이

script_directory = os.path.dirname(os.path.abspath(__file__))
dataset_directory = os.path.join(script_directory, "dataset_frame")


data_file_list = os.listdir(dataset_directory)
# data_file_list = sorted(data_file_list, reverse=True)

# 프레임 단위 데이터셋일 때
for file_name in data_file_list:
    npy_file = os.path.join(script_directory, 'dataset', file_name)
    data = np.load(npy_file)
    # print(len(data))
    
    if len(data) - seq_length < 5:
        print("프레임이 너무 적어 데이터 생성에 실패했습니다.")
        continue

    # 시퀀스 데이터 생성
    full_seq_data = []
    for seq in range(len(data) - seq_length):
        full_seq_data.append(data[seq:seq + seq_length])

    full_seq_data = np.array(full_seq_data)
    print(data[0][-1], ": ", full_seq_data.shape)

    # # 데이터셋 새로 저장할 디렉터리 생성
    new_dir_name = f"dataset_seq_{seq_length}"
    os.mkdir(new_dir_name)

    new_dataset_directory = os.path.join(script_directory, new_dir_name)
    
    new_save_data = os.path.join(new_dataset_directory, file_name)
    
    np.save(new_save_data, full_seq_data)        



