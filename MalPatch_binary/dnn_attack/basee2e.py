from abc import abstractmethod
from os.path import isfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import struct
from secml.settings import SECML_PYTORCH_USE_CUDA
from torch.nn import Module

from secml_malware.utils.exceptions import FileNotExistsException


use_cuda = torch.cuda.is_available() and SECML_PYTORCH_USE_CUDA


class End2EndModel(Module):

	def __init__(self, embedding_size: int, max_input_size: int, embedding_value: int, shift_values: bool):
		"""
		Basic end to end model wrapper

		Parameters
		----------
		embedding_size : int
			the size of the embedding space
		max_input_size : int
			the input window size
		embedding_value : int
			the value used as padding
		shift_values : bool
			True if values are shifted by one
		"""
		super(End2EndModel, self).__init__()
		self.embedding_size = embedding_size
		self.max_input_size = max_input_size
		self.embedding_value = embedding_value
		self.shift_values = shift_values

	@classmethod
	def path_to_exe_vector(cls, path: str, max_length: int, padding_value: int, shift_values: bool) -> np.ndarray:
		"""
		Creates a numpy array from the bytes contained in file

		Parameters
		----------
		path : str
			the path of the file
		max_length : int
			the max input size of the network
		padding_value : int
			the value used as padding
		shift_values : bool
			True if values are shifted by one

		Returns
		-------
		numpy array
			the sample as numpy array, cropped or padded
		"""
		exe = cls.load_sample_from_file(path, max_length, has_shape=True, padding_value=padding_value,
										shift_values=shift_values)
		return exe.reshape(max_length)

	@classmethod
	@abstractmethod
	def bytes_to_numpy(cls, bytez: bytes, max_length: int, padding_value: int, shift_values: bool) -> np.ndarray:
		"""
		It creates a numpy array from bare bytes. The vector is max_length long.

		Parameters
		----------
		bytez : bytes
			byte string containing the sample
		max_length : int
			the max input size of the network
		padding_value : int
			the value used as padding
		shift_values : bool
			True if values are shifted by one

		Returns
		-------
		numpy array
			the sample as numpy array, cropped or padded

		"""
		# b = np.ones((max_length,), dtype=np.uint16) * padding_value
		# bytez = np.frombuffer(bytez[:max_length], dtype=np.uint8)
		# bytez = bytez.astype(np.uint16) + shift_values
		# b[: len(bytez)] = bytez
		# return np.array(b, dtype=float)


		b = np.ones((max_length,), dtype=np.uint16) * 0

		bytez = np.frombuffer(bytez, dtype=np.uint8)

		pe_position = struct.unpack("<I", bytez[0x3C:0x40])
		pe_position = pe_position[0]

		if pe_position > len(bytez):
			bytez = bytez[:4096]
			b[: min(4096, len(bytez))] = bytez[: min(4096, len(bytez))]
			return np.array(b, dtype=float)


		else:
			optional_header_size = bytez[pe_position + 20]

			coff_header_size = 24

			content_offset = pe_position + optional_header_size + coff_header_size + 12

			first_content_offset = struct.unpack("<I", bytez[content_offset:content_offset + 4])

			bytez = bytez[:first_content_offset[0]]

			b[: min(4096, len(bytez))] = bytez[: min(4096, len(bytez))]
			return np.array(b, dtype=float)

	@classmethod
	@abstractmethod
	def list_to_numpy(cls, x, max_length, padding_value, shift_values):
		"""
		It creates a numpy array from bare bytes. The vector is max_length long.
		"""
		b = np.ones((max_length,), dtype=np.uint16) * padding_value
		bytez = np.array(x[:max_length], dtype=np.uint8)
		bytez = bytez.astype(np.uint16) + shift_values
		b[: len(bytez)] = bytez
		return np.array(b, dtype=float)

	@classmethod
	@abstractmethod
	def load_sample_from_file(cls, path, max_length, has_shape, padding_value, shift_values):
		"""
		It creates a numpy array containing a sample. The vector is max_length long.
		If shape is true, then the path is supposed to be a (1,1) matrix.
		Hence, item() is called.
		"""
		file_path = path.item() if has_shape else path
		with open(file_path, "rb") as malware:
			code = cls.bytes_to_numpy(malware.read(), max_length, padding_value, shift_values)
		return code

	@abstractmethod
	def embed(self, input_x, transpose=True):
		"""
		It embeds an input vector into MalConv embedded representation.
		"""
		pass

	# @abstractmethod
	def compute_embedding_gradient(self, numpy_x: np.ndarray) -> torch.Tensor:
		"""
		It computes the gradient w.r.t. the embedding layer.

		Parameters
		----------
		numpy_x : numpy array
			the numpy array containing a sample
		Returns
		-------
		torch.Tensor
			the gradient w.r.t. the embedding layer
		"""
		emb_x = self.embed(numpy_x)
		y = self.embedd_and_forward(emb_x)
		g = torch.autograd.grad(y, emb_x)[0]
		g = torch.transpose(g, 1, 2)[0]
		return g

	@abstractmethod
	def embedd_and_forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Compute the embedding for sample x and returns the prediction.

		Parameters
		----------
		x : torch.Tensor
			the sample as torch tensor
		Returns
		-------
		torch.Tensor
			the result of the forward pass
		"""
		pass

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Forward pass.

		Parameters
		----------
		x : torch.Tensor
			the sample to test
		Returns
		-------
		torch.Tensor
			the result of the forward pass
		"""
		embedding_1 = self.embed(x.contiguous())
		output = self.embedd_and_forward(embedding_1)
		return output

	def load_simplified_model(self, path: str):
		"""
		Load the model weights.

		Parameters
		----------
		path : str
			the path to the model
		"""
		if not isfile(path):
			raise FileNotExistsException('{} path not pointing to regular file!'.format(path))
		f = torch.load(path) if use_cuda else torch.load(path, map_location="cpu")
		self.load_state_dict(f)
		self.eval()
