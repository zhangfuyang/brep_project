import numpy as np
import argparse
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--checking_dir', type=str, default='output')
args = parser.parse_args()

codebook_size = 8192

codebook_count_ = np.zeros(codebook_size, dtype=np.int64)
for data_name in tqdm(os.listdir(os.path.join(args.checking_dir, 'quantization'))):
    data_path = os.path.join(args.checking_dir, 'quantization', data_name)
    if data_path.endswith('.npy') is False:
        continue
    data = np.load(data_path)
    data = data.reshape(-1)
    # add to codebook count
    for i in range(codebook_size):
        codebook_count_[i] += np.sum(data == i)
    
# make histogram
plt.bar(range(codebook_size), codebook_count_)
plt.savefig(os.path.join(args.checking_dir, 'codebook_histogram.png'))

# save statistics
with open(os.path.join(args.checking_dir, 'codebook_statistics.txt'), 'w') as f:
    f.write('max: {}\n'.format(np.max(codebook_count_)))
    f.write('min: {}\n'.format(np.min(codebook_count_)))
    f.write('mean: {}\n'.format(np.mean(codebook_count_)))
    f.write('std: {}\n'.format(np.std(codebook_count_)))
    f.write('median: {}\n'.format(np.median(codebook_count_)))
    f.write('nonzero count: {}\n'.format(np.count_nonzero(codebook_count_)))

# save codebook count
np.save(os.path.join(args.checking_dir, 'codebook_count.npy'), codebook_count_)

