import math
import pickle 
import argparse
from tqdm import tqdm
from hashlib import sha256
from convert_utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--bit",  type=int, help='Deduplicate precision', default=6)
parser.add_argument("--data_list", type=str, help="Data list path", default='data_list_path.txt')
parser.add_argument("--output", type=str, help="Output file path", default='deduplicated_data_path.txt')
args = parser.parse_args()

# Remove duplicate for the training set 
train_path = []
unique_hash = set()
total = 0

# load data list
with open(args.data_list, "r") as file:
    train = file.readlines()

for path_idx, path in tqdm(enumerate(train)):
    path = path.strip()
    total += 1

    # Load pkl data
    with open(path, "rb") as file:
        data = pickle.load(file) 

    # Hash the surface sampled points
    surfs_wcs = data['surf_wcs']
    surf_hash_total = []
    for surf in surfs_wcs:
        np_bit = real2bit(surf, n_bits=args.bit).reshape(-1, 3)  # bits
        data_hash = sha256(np_bit.tobytes()).hexdigest()
        surf_hash_total.append(data_hash)
    surf_hash_total = sorted(surf_hash_total)
    data_hash = '_'.join(surf_hash_total)

    # Save non-duplicate shapes
    prev_len = len(unique_hash)
    unique_hash.add(data_hash)  
    if prev_len < len(unique_hash):
        train_path.append(uid)
    else:
        continue
        
    if path_idx % 2000 == 0:
        print(len(unique_hash)/total)

# save data 
data_path = {
    'train':train_path,
    'val':val_path,
    'test':test_path,
}
with open(OUTPUT, "wb") as tf:
    pickle.dump(data_path, tf)
