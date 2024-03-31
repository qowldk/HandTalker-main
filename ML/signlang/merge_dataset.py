import os
import numpy as np


dir_name_to_merge = "dataset" # 합칠 npy 파일들의 디렉터리 이름


script_directory = os.path.dirname(os.path.abspath(__file__))
dataset_directory = os.path.join(script_directory, dir_name_to_merge)


data_file_list = os.listdir(dataset_directory)
# data_file_list = sorted(data_file_list, reverse=True)


first_npy_file = os.path.join(script_directory, dir_name_to_merge, data_file_list[0])
total_data = np.load(first_npy_file).tolist()
data_file_list = data_file_list[1:]

# print(total_data) # 첫 npy 
# print(len(total_data))

for file_name in data_file_list:
    npy_file = os.path.join(script_directory, 'dataset', file_name)
    data = np.load(npy_file).tolist()
    for d in data:
        total_data.append(d)

total_data = np.array(total_data) # list -> numpy
print("total_dataset: ", total_data.shape)

# 저장
script_directory = os.path.dirname(os.path.abspath(__file__))
file_name = 'total_dataset' + ''
save_data = os.path.join(script_directory, file_name)
np.save(save_data, total_data)

print("done")