import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from secml_malware.models.basee2e import End2EndModel
from secml.settings import SECML_PYTORCH_USE_CUDA

use_cuda = torch.cuda.is_available() and SECML_PYTORCH_USE_CUDA

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