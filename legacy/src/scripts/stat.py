import numpy as np
from glob import glob
from tqdm import tqdm


if __name__ == '__main__':
    # file_pattern = '/root/autodl-tmp/datasets/csl-news/poses/archive_*/*.npy'
    # file_pattern = '/root/autodl-tmp/datasets/csl-daily/sentence/pred_poses_0602_rtm/*.npy'
    file_pattern = '/root/autodl-tmp/datasets/collected_base/poses/*.npy'
    # file_pattern = '/root/autodl-tmp/datasets/collected_csl/poses/*.npy'

    sum_x = np.zeros(3)
    sum_x2 = np.zeros(3)
    count = 0

    files = sorted(glob(file_pattern))
    for file in tqdm(files):
        data = np.load(file)

        nan = np.isnan(data)
        data[nan] = data[~nan].mean()

        sum_x += data.sum(axis=0).sum(axis=0).sum(axis=0)
        sum_x2 += (data ** 2).sum(axis=0).sum(axis=0).sum(axis=0)
        count += np.ones_like(data).sum() / 3

    mean = sum_x / count
    std = np.sqrt(sum_x2 / count - mean ** 2)

    print(file_pattern)
    print(f'mean: {mean}, std: {std}')
