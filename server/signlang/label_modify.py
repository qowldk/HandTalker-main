import os
import numpy as np

# a 라벨을 b 라벨로 변경
a = 1000
b = 1001

datset_dir = "dataset" # 합칠 npy 파일들의 디렉터리 이름


script_directory = os.path.dirname(os.path.abspath(__file__))
dataset_directory = os.path.join(script_directory, datset_dir)

count=0

data_file_list = os.listdir(dataset_directory) # 파일 리스트

for file_name in data_file_list:
    label = int(file_name.split('_')[0])
    if label==a:
        count+=1
        edit_file = os.path.join(dataset_directory, file_name)
        data = np.load(edit_file) # (1,30,n)

        for i in range(30):
            data[0][i][-1] = b

        new_file_name = str(b)

        new_file_name += file_name[len(str(a)):]


        os.remove(edit_file)
        np.save(os.path.join(dataset_directory, new_file_name), data)

print(a, "->", b, "done:", count, "개")
