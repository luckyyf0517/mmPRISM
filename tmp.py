import numpy as np
from glob import glob
from tqdm import tqdm

datalist = glob('data/collected_base/poses/*.npy')

for data_path in tqdm(datalist):
    data = np.load(data_path)
    if not data.shape[0] == 99:
        print(data_path)