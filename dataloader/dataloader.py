#!/usr/bin/env python

import os
from os import listdir
from os.path import join, isfile
from PIL import Image
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms

def loadImagesForTraining(args):
	mask_files = [f for f in listdir(args["root"]) if isfile(join(args["root"], f)) and f.endswith('.png')]
	print(mask_files)
	mask_files = mask_files[:args["train_size"]]
	celeb_files = [f.split('mask_')[1] for f in mask_files]

	return mask_files, celeb_files

def loadImagesForTesting(args):
	mask_files = [f for f in listdir(args["root"]) if isfile(join(args["root"], f)) and f.startswith('mask_') and f.endswith('.png')]
	mask_files = mask_files[args["train_size"]:]
	celeb_files = [f.split('mask_')[1] for f in mask_files]

	return mask_files, celeb_files

class CelebDataset(Dataset):
	def __init__(self, args, train=True, transform=None):
		self.args = args
		if train == True:
			self.mask_files, self.celeb_files = loadImagesForTraining(args)
		else:
			self.mask_files, self.celeb_files = loadImagesForTesting(args)
		self.data_size = len(self.mask_files)

	def __getitem__(self, index):
		mask = Image.open(join(self.args["root"], self.mask_files[index]))
		indices = np.where(mask == [255])
		for (i,j) in (indices[0], indices[1]):
			mask[i,j] = 1
		mask.show()
		celeb = Image.open(self.args["celeb"] + self.celeb_files[index]).convert('RGB')
				
		if self.transform is not None:
			mask = self.transform(mask)
			celeb = self.transform(celeb)

		return (mask, celeb)

	def __len__(self):
		return self.data_size