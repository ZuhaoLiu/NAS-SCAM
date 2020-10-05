import torch
from torch import sigmoid
import torch.nn.functional as F
from torch.nn.functional import relu
from primitives import SAM_PRIMITIVES, SAM_OPS, CAM_PRIMITIVES, CAM_OPS
import math
from conv_modules import * 

class MixedOp(nn.Module):

	def __init__(self, input_C, output_C, PRIMITIVE, OPS):
		super(MixedOp, self).__init__()

		self.ops = nn.ModuleList()

		for primitive in PRIMITIVE:
			op = OPS[primitive](input_C, output_C)
			self.ops.append(op)

	def forward(self, x, weights, function, ops_number):
		for i in range(ops_number):
			if i == 0:
				output = weights[i] * function(self.ops[i](x))
			else:
				output.add_(weights[i] * function(self.ops[i](x)))
		return output



class Cell(nn.Module):
	def __init__(self, C, operation_number, ops_number, cell_type = 'SAM'):
		super(Cell, self).__init__()
		self.operation_number = operation_number
		self.ops_number = ops_number
		self.cell_type = cell_type
		channel_list = [32]*(operation_number+1)
		channel_list[0] = C
		channel_list[-1] = 1
		self.mixop_list = nn.ModuleList()
		for i in range(operation_number):
			for j in range(i+1):
				self.mixop_list.append(MixedOp(channel_list[j], channel_list[i+1], 
								globals()[cell_type+'_PRIMITIVES'], 
								globals()[cell_type+'_OPS']))
		self.function_list = ['relu']*math.factorial(operation_number)
		self.function_list[-operation_number:] = ['sigmoid']*operation_number
	def forward(self, input_x, weights):
		x = input_x
		if self.cell_type == 'SAM':
			pass
		else:
			batch_size, channels, height, width = x.size()
			x = x.mean(dim=(2,3))
			x = torch.unsqueeze(x, dim = 1)
		total_x = list()
		total_index = 0
		add_x = x
		for i in range(self.operation_number):
			total_x.append(add_x)
			now_x = 0
			for j in range(i+1):
				now_x = torch.add(self.mixop_list[total_index](total_x[j],
							weights[total_index],
							globals()[self.function_list[total_index]],
							self.ops_number), now_x)
				total_index += 1
			add_x = now_x
		x = torch.div(add_x,self.operation_number)
		if self.cell_type == 'SAM':
			return torch.mul(input_x, x)
		else:
			return torch.mul(input_x, x.view(batch_size, channels, 1, 1))



class SCAM_P(nn.Module):
	def __init__(self, C):
		super(SCAM_P, self).__init__()
		self.SAM = Cell(C, 3, 6, cell_type = 'SAM')
		self.CAM = Cell(1, 3, 8, cell_type = 'CAM')
	def forward(self, x, SAM_weights, CAM_weights):
		x = torch.max(self.SAM(x, SAM_weights), self.CAM(x, CAM_weights))
		return x


class SCAM_S(nn.Module):
	def __init__(self, C):
		super(SCAM_S, self).__init__()
		self.SAM = Cell(C, 3, 6, cell_type = 'SAM')
		self.CAM = Cell(1, 3, 8, cell_type = 'CAM')
	def forward(self, x, SAM_weights, CAM_weights):
		x = self.SAM(x, SAM_weights)
		x = self.CAM(x, CAM_weights)
		return x



class MakeDownLayers(nn.Module):
	def __init__(self, in_ch, layers_number, attention_type = 'P'):
		super(MakeDownLayers, self).__init__()
		self.layers_number = layers_number
		self.down_list = nn.ModuleList()
		self.SCAM_list = nn.ModuleList()
		for i in range(self.layers_number):
			self.down_list.append(Down(in_ch, in_ch*2))
			self.SCAM_list.append(globals()['SCAM_'+attention_type](in_ch*2))
			in_ch *= 2

	def forward(self, x, SAM_weights, CAM_weights):
		output_list = list()
		output_list.append(x)
		for i in range(self.layers_number):
			x = self.down_list[i](x)
			x = self.SCAM_list[i](x, SAM_weights[i], CAM_weights[i])
			output_list.append(x)
		return output_list
		

class MakeUpLayers(nn.Module):
	def __init__(self, in_ch, layers_number, attention_type = 'P'):
		super(MakeUpLayers, self).__init__()
		self.layers_number = layers_number
		self.up_list = nn.ModuleList()
		self.SCAM_list = nn.ModuleList()
		for i in range(self.layers_number):
			self.up_list.append(Up(in_ch, in_ch // 2))
			self.SCAM_list.append(globals()['SCAM_'+attention_type](in_ch // 2))
			in_ch //= 2
	def forward(self, x_list, SAM_weights, CAM_weights):
		x = x_list[-1]
		for i in range(self.layers_number):
			x = self.up_list[i](x, x_list[-i-2])
			x = self.SCAM_list[i](x, SAM_weights[i], CAM_weights[i])
		return x

class SearchNet(nn.Module):
	def __init__(self, NChannels = 4, NClasses = 4, FeatureRoot = 16, SamplingNumber = 4):
		super(SearchNet, self).__init__()
		self.SamplingNumber = SamplingNumber
		self.InConv = DoubleConv(NChannels, FeatureRoot)
		self.DownLayers = MakeDownLayers(FeatureRoot, SamplingNumber)
		self.Middle = DoubleConv(2**SamplingNumber*FeatureRoot, 2**SamplingNumber*FeatureRoot)
		self.UpLayers = MakeUpLayers(2**SamplingNumber*FeatureRoot, SamplingNumber)
		self.OutConv = nn.Conv2d(FeatureRoot, NClasses, 1)
		self.initialize_alphas()
	def forward(self, x):
		SAM_weights = F.softmax(self.SAM_alphas, dim = -1)
		CAM_weights = F.softmax(self.CAM_alphas, dim = -1)
		x = self.InConv(x)
		x = self.DownLayers(x, SAM_weights[0:self.SamplingNumber], CAM_weights[0:self.SamplingNumber])
		x[-1] = self.Middle(x[-1])
		x = self.UpLayers(x, SAM_weights[self.SamplingNumber:], CAM_weights[self.SamplingNumber:])
		x = torch.sigmoid(self.OutConv(x))
		return x, SAM_weights, CAM_weights
	def initialize_alphas(self, SAMOperationNumber = 3, CAMOperationNumber = 3, SAMOpsNumber = 6, CAMOpsNumber = 8):
		self.SAM_alphas = nn.Parameter(1e-3*torch.randn(self.SamplingNumber*2, 
								math.factorial(SAMOperationNumber), SAMOpsNumber).cuda())
		self.CAM_alphas = nn.Parameter(1e-3*torch.randn(self.SamplingNumber*2, 
								math.factorial(CAMOperationNumber), CAMOpsNumber).cuda())
		

