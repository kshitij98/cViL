import torch
from tqdm import tqdm
import os

files = ['train_img_frcnn_feats.pt', 'val_img_frcnn_feats.pt', 'test-dev2015_img_frcnn_feats.pt']

for name in files:
	dir_name = name.split('_')[0]
	data = torch.load(name)
	print(f'{dir_name} has {len(data)} images')

	os.makedirs(dir_name, exist_ok=True)

	n_saved = len(os.listdir(dir_name))

	if n_saved == len(data):
		continue

	for key in tqdm(data.keys()):
		features = data[key]
		torch.save(features, f'{dir_name}/{key}.pt')
