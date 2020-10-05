import torch
import torch.nn as nn
from conv_modules import Zero1D, Zero2D

SAM_PRIMITIVES = [
	'None',
	'Conv1',
	'Conv3',
	'Conv5',
	'DilaConv3',
	'DilaConv5',
]

SAM_OPS = {
	'None': lambda input_C, output_C: Zero2D(input_C, output_C),

	'Conv1' : lambda input_C, output_C: nn.Sequential(
							nn.Conv2d(input_C, output_C, 1),
							nn.BatchNorm2d(output_C)),

	'Conv3' : lambda input_C, output_C: nn.Sequential(
							nn.Conv2d(input_C, output_C, 3, padding = 1),
							nn.BatchNorm2d(output_C)),
							
	'Conv5' : lambda input_C, output_C: nn.Sequential(
							nn.Conv2d(input_C, output_C, 5, padding = 2),
							nn.BatchNorm2d(output_C)),
						
	'DilaConv3' : lambda input_C, output_C: nn.Sequential(
							nn.Conv2d(input_C, output_C, 3, padding = 2, dilation=2),
							nn.BatchNorm2d(output_C)),

	'DilaConv5': lambda input_C, output_C: nn.Sequential(
							nn.Conv2d(input_C, output_C, 5, padding = 4, dilation=2),
							nn.BatchNorm2d(output_C))
}

CAM_PRIMITIVES = [
	'None',
	'Conv1',
	'Conv3',
	'Conv5',
	'Conv9',
	'Conv15',
	'DilaConv3',
	'DilaConv5',
]

CAM_OPS = {
	'None': lambda input_C, output_C: Zero1D(input_C, output_C),

	'Conv1' : lambda input_C, output_C: nn.Sequential(
							nn.Conv1d(input_C, output_C, 1),
							nn.BatchNorm1d(output_C)),

	'Conv3' : lambda input_C, output_C: nn.Sequential(
							nn.Conv1d(input_C, output_C, 3, padding = 1),
							nn.BatchNorm1d(output_C)),
							
	'Conv5' : lambda input_C, output_C: nn.Sequential(
							nn.Conv1d(input_C, output_C, 5, padding = 2),
							nn.BatchNorm1d(output_C)),
	'Conv9' : lambda input_C, output_C: nn.Sequential(
							nn.Conv1d(input_C, output_C, 9, padding = 4),
							nn.BatchNorm1d(output_C)),

	'Conv15' : lambda input_C, output_C: nn.Sequential(
							nn.Conv1d(input_C, output_C, 15, padding = 7),
							nn.BatchNorm1d(output_C)),
						
	'DilaConv3' : lambda input_C, output_C: nn.Sequential(
							nn.Conv1d(input_C, output_C, 3, padding = 2, dilation=2),
							nn.BatchNorm1d(output_C)),

	'DilaConv5': lambda input_C, output_C: nn.Sequential(
							nn.Conv1d(input_C, output_C, 5, padding = 4, dilation=2),
							nn.BatchNorm1d(output_C))
}
