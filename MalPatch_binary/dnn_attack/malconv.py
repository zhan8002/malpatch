"""
Malware Detection by Eating a Whole EXE
Edward Raff, Jon Barker, Jared Sylvester, Robert Brandon, Bryan Catanzaro, Charles Nicholas
https://arxiv.org/abs/1710.09435
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from secml_malware.models.basee2e import End2EndModel
from secml.settings import SECML_PYTORCH_USE_CUDA

use_cuda = torch.cuda.is_available() and SECML_PYTORCH_USE_CUDA


class MalConv(End2EndModel):
	"""
	Architecture implementation.
	"""

	def __init__(self, pretrained_path=None, embedding_size=8, max_input_size=2 ** 20):
		super(MalConv, self).__init__(embedding_size, max_input_size, 256, False)
		self.embedding_1 = nn.Embedding(num_embeddings=257, embedding_dim=embedding_size)
		self.conv1d_1 = nn.Conv1d(in_channels=embedding_size, out_channels=128, kernel_size=(500,), stride=(500,),
								  groups=1, bias=True)
		self.conv1d_2 = nn.Conv1d(in_channels=embedding_size, out_channels=128, kernel_size=(500,), stride=(500,),
								  groups=1, bias=True)
		self.dense_1 = nn.Linear(in_features=128, out_features=128, bias=True)
		self.dense_2 = nn.Linear(in_features=128, out_features=1, bias=True)
		if pretrained_path is not None:
			self.load_simplified_model(pretrained_path)
		if use_cuda:
			self.cuda()

	def embed(self, input_x, transpose=True):
		if isinstance(input_x, torch.Tensor):
			x = input_x.clone().detach().requires_grad_(True).type(torch.LongTensor)
		else:
			x = torch.from_numpy(input_x).type(torch.LongTensor)
		x = x.squeeze(dim=1)
		if use_cuda:
			x = x.cuda()
		emb_x = self.embedding_1(x)
		if transpose:
			emb_x = torch.transpose(emb_x, 1, 2)
		return emb_x

	def embedd_and_forward(self, x):
		conv1d_1 = self.conv1d_1(x)
		conv1d_2 = self.conv1d_2(x)
		conv1d_1_activation = torch.relu(conv1d_1)
		conv1d_2_activation = torch.sigmoid(conv1d_2)
		multiply_1 = conv1d_1_activation * conv1d_2_activation
		global_max_pooling1d_1 = F.max_pool1d(input=multiply_1, kernel_size=multiply_1.size()[2:])
		global_max_pooling1d_1_flatten = global_max_pooling1d_1.view(global_max_pooling1d_1.size(0), -1)
		dense_1 = self.dense_1(global_max_pooling1d_1_flatten)
		dense_1_activation = torch.relu(dense_1)
		dense_2 = self.dense_2(dense_1_activation)
		dense_2_activation = torch.sigmoid(dense_2)
		return dense_2_activation

class DNN_Net(End2EndModel):
	"""
	Architecture implementation.
	"""

	def __init__(self, pretrained_path=None, embedding_size=16, max_input_size=4096):
		super(DNN_Net, self).__init__(embedding_size, max_input_size, 256, False)

		self.embed_1 = nn.Embedding(257, 16)
		self.conv1d_1 = nn.Conv1d(16, 48, 32, 4, padding=14)
		self.conv1d_2 = nn.Conv1d(48, 96, 32, 4, padding=14)
		self.conv1d_3 = nn.Conv1d(96, 128, 16, 8, padding=4)
		self.conv1d_4 = nn.Conv1d(128, 192, 16, 8, padding=4)

		self.conv_dropout = nn.Dropout(0.2)
		self.dense_dropout = nn.Dropout(0.5)

		self.dense_1 = nn.Linear(192, 64)
		self.dense_2 = nn.Linear(64, 1)

		if pretrained_path is not None:
			self.load_simplified_model(pretrained_path)
		if use_cuda:
			self.cuda()

	def embed(self, input_x, transpose=True):
		if isinstance(input_x, torch.Tensor):
			x = input_x.float().clone().detach().requires_grad_(True).type(torch.LongTensor)
		else:
			x = torch.from_numpy(input_x).type(torch.LongTensor)
		x = x.squeeze(dim=1)
		if use_cuda:
			x = x.cuda()
		emb_x = self.embed_1(x)
		if transpose:
			emb_x = torch.transpose(emb_x, -1, -2)
		return emb_x

	def embedd_and_forward(self, x):

		x = self.conv_dropout(x)

		x = F.relu(self.conv1d_1(x))
		x = self.conv_dropout(x)

		x = F.relu(self.conv1d_2(x))
		x = self.conv_dropout(x)

		x = F.max_pool1d(x, 4, stride=4, padding=0)

		x = F.relu(self.conv1d_3(x))
		x = self.conv_dropout(x)
		x = F.relu(self.conv1d_4(x))
		x = self.conv_dropout(x)

		x = x.view(-1, 192)

		x = F.selu(self.dense_1(x))
		x = self.dense_2(x)

		x = torch.sigmoid(x)
		return x