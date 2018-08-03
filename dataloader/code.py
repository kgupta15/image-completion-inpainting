#!/usr/bin/env python

import os
from os import listdir
from os.path import join, isfile
from PIL import Image
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt

root = "/home/kapil/Documents/DeepLearning/image-completion-inpainting/data/input"
celeb = "/home/kapil/Documents/DeepLearning/image-completion-inpainting/data/celebA" 
train_size = 10

def loadImagesForTraining():
	mask_files = [f for f in listdir(root) if isfile(join(root, f)) and f.endswith('.png') and f.startswith('mask_')]
	mask_files = mask_files[:train_size]
	celeb_files = [f.split('mask_')[1] for f in mask_files]

	return mask_files, celeb_files

def loadImagesForTesting():
	mask_files = [f for f in listdir(root) if isfile(join(root, f)) and f.startswith('mask_') and f.endswith('.png')]
	mask_files = mask_files[train_size:]
	celeb_files = [f.split('mask_')[1] for f in mask_files]

	return mask_files, celeb_files

class CelebDataset(Dataset):
	def __init__(self, train=True, transform=None):
		# self.args = args
		if train == True:
			self.mask_files, self.celeb_files = loadImagesForTraining()
		else:
			self.mask_files, self.celeb_files = loadImagesForTesting()
		self.transform = transform
		self.data_size = len(self.mask_files)
		self.root = "/home/kapil/Documents/DeepLearning/image-completion-inpainting/data/input"
		self.celeb = "/home/kapil/Documents/DeepLearning/image-completion-inpainting/data/celebA"

	def __getitem__(self, index):
		mask = cv2.imread(join(self.root, self.mask_files[index]))
		(b,g,r) = cv2.split(mask)
		indices = np.where(b == [255])
		for (i,j) in zip(indices[0], indices[1]):
			b[i,j] = 1
		final_mask = Image.fromarray(b)
		celeb_image = Image.open(join(self.celeb, self.celeb_files[index])).convert('RGB')

		i,j = indices
		bbox = ((min(i), min(j)), (max(i), max(j)))

		if self.transform is not None:
			final_mask = self.transform(final_mask)
			celeb_image = self.transform(celeb_image)

		return (celeb_image, final_mask, bbox)

	def __len__(self):
		return self.data_size


train_dataset = CelebDataset(True, transforms.Compose([
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor()
]))
train_loader = torch.utils.data.DataLoader(
				train_dataset, batch_size=10, shuffle=False)

for i, images in enumerate(train_loader):
	faces, masks, bboxes = images
	print(masks.shape)
	print(faces.shape)
	image_4ch = torch.cat((faces, masks), 1)
	print(image_4ch.shape)
	