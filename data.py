import torch

from torch.utils.data import Dataset



class MyDataset(Dataset):
	def __init__(self, phase='train'):
		"""
		Initialize your dataset, e.g. by loading data from a csv.
		"""
		super().__init__()

	def __len__(self):
		"""
		Getter method for dataset length.
		"""
		pass

	def __getitem__(self, idx):
		"""
		Return data item with index 'idx'.
		"""
		pass

def my_collate_fn(batch):
	"""
	This function will be called by a dataloader -- given a batch of data instances,
	it should return a 'collated' version, i.e. so that given N total data instances
	with A attributes each, this function will return A objects, each with first
	dimension N. Read more here: https://pytorch.org/docs/stable/data.html
	"""
	pass