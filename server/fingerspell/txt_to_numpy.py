import numpy as np
import os
# txt->numpy
sc_dir = os.path.dirname(os.path.abspath(__file__))
# print(type(sc_dir))
# print(type('ss'))
txt_file = os.path.join(sc_dir, "dataSet_ko.txt")
f = open(txt_file, 'r')
total_arr = []
while True:
    line = f.readline().rstrip()
    if not line: break
    # print(line)
    line_arr = []
    for d in line.split(','):
        line_arr.append(d)
    total_arr.append(line_arr)
f.close()

# print(total_arr)
total_arr = np.array(total_arr)

save_data = os.path.join(sc_dir, "fs_dataset.npy")

np.save(save_data, total_arr)
print('locate: ', save_data)