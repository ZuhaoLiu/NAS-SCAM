import torch
import torch.nn as nn
import torch.nn.functional as F

class Zero1D(nn.Module):

	def __init__(self, input_C, output_C):
		super(Zero1D, self).__init__()
		self.conv = nn.Conv1d(input_C, output_C, 1)

	def forward(self, x):
		x = torch.mul(self.conv(x), 0.)
		return x


class Zero2D(nn.Module):

	def __init__(self, input_C, output_C):
		super(Zero2D, self).__init__()
		self.conv = nn.Conv2d(input_C, output_C, 1)

	def forward(self, x):
		x = torch.mul(self.conv(x), 0.)
		return x

class DoubleConv(nn.Module):
	'''Conv => BN => ReLU => Dropout => Conv => BN => ReLU'''
	def __init__(self, in_ch, out_ch):
		super(DoubleConv, self).__init__()
		self.conv = nn.Sequential(
		nn.Conv2d(in_ch, out_ch, 3, padding=1),
		nn.GroupNorm(4, out_ch),
		nn.ReLU(inplace=True),
		nn.Dropout(p = 0.2),
		nn.Conv2d(out_ch, out_ch, 3, padding=1),
		nn.GroupNorm(4, out_ch),
		nn.ReLU(inplace=True)
		)

	def forward(self, x):
		x = self.conv(x)
		return x


class Down(nn.Module):
	def __init__(self, in_ch, out_ch):
		super(Down, self).__init__()
		self.conv = nn.Sequential(
			DoubleConv(in_ch, out_ch),
			nn.MaxPool2d(2)
			)
	def forward(self, x):
		x = self.conv(x)
		return x

class Up(nn.Module):
	def __init__(self, in_ch, out_ch):
		super(Up, self).__init__() 
		self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1)
		self.conv = DoubleConv(in_ch, out_ch)

	def forward(self, x1, x2):
		x1 = self.up(x1)
		x = torch.cat([x2, x1], dim=1)
		x = self.conv(x)
		return x














