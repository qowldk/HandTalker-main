import os
import numpy as np

# dataset 디렉터리 내의 모든 데이터를 하나의 npy 파일로 저장

dir_name_to_merge = "dataset" # 합칠 npy 파일들의 디렉터리 이름


script_directory = os.path.dirname(os.path.abspath(__file__))
dataset_directory = os.path.join(script_directory, dir_name_to_merge)

data_file_list = os.listdir(dataset_directory)

first_npy_file = os.path.join(script_directory, dir_name_to_merge, data_file_list[0])
total_data = np.load(first_npy_file).tolist()
data_file_list = data_file_list[1:]


for file_name in data_file_list:
    npy_file = os.path.join(script_directory, 'dataset', file_name)
    data = np.load(npy_file).tolist()
    for d in data:
        total_data.append(d)

total_data = np.array(total_data) # list -> numpy

# 저장
script_directory = os.path.dirname(os.path.abspath(__file__))
file_name = 'total_dataset'
save_data = os.path.join(script_directory, file_name)
np.save(save_data, total_data)

print("done: ", total_data.shape)
