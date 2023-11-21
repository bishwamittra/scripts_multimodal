import sys
# Appending path to import layers from rescalenet (change accordingly)
# sys.path.append("/nvmedata/scratch/shrirajp/SambaNova_Astar/classification_1.16.5-38")

server_name = 'SambaNova'
import pickle
import socket
import struct
import time
from statistics import mean
from typing import List, Tuple
import sambaflow.samba.utils as sn_utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sambaflow import samba
from tqdm import tqdm

from classification.rescalenet.layers import Bias2DMean

# SP DEUBG: For generating model summary
from torchinfo import summary

# def get_inputs(map_size):
# 	images_sn = samba.randn(map_size,
# 	                        name = 'input',
# 	                        batch_dim = 0).bfloat16()
#
# 	return (images_sn,)

def get_inputs(batch_size) -> Tuple[samba.SambaTensor, ...]:
	tensor = samba.randn(batch_size, 64, 8, 8, name='input', batch_dim=0).bfloat16()
	# tensor = torch.randn(batch_size, 64, 8, 8) # SP DEBUG: Used for torch summary
	# Following KT Debug and adding tensor.requires_grad_(False)
    # tensor.requires_grad_(args.compute_input_grad)
	tensor.requires_grad_(False)
	return (tensor, )

def send_msg(sock, msg):
	# prefix each message with a 4-byte length in network byte order
	msg = pickle.dumps(msg)
	l_send = len(msg)
	msg = struct.pack('>I', l_send) + msg
	sock.sendall(msg)
	return l_send


def recv_msg(sock):
	# read message length and unpack it into an integer
	raw_msg_len = recv_all(sock, 4)
	if not raw_msg_len:
		return None
	msg_len = struct.unpack('>I', raw_msg_len)[0]
	# read the message data
	msg = recv_all(sock, msg_len)
	msg = pickle.loads(msg)
	return msg, msg_len


def recv_all(sock, n):
	# helper function to receive n bytes or return None if EOF is hit
	data = b''
	while len(data) < n:
		packet = sock.recv(n - len(data))
		if not packet:
			return None
		data += packet
	return data


class BasicBlock(nn.Module):
	expansion: int = 1

	def __init__(self,
	             inplanes: int,
	             planes: int,
	             block_idx: int,
	             max_block: int,
	             stride: int = 1,
	             groups: int = 1,
	             base_width: int = 64,
	             drop_conv = 0.0) -> None:

		super(BasicBlock, self).__init__()

		if groups != 1 or base_width != 64:
			raise ValueError('BasicBlock only supports groups=1 and base_width=64')

		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size = 3, padding = 1, stride = stride, groups = groups, bias = False)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size = 3, padding = 1, stride = 1, groups = groups, bias = False)

		self.addbias1 = Bias2DMean(inplanes)
		self.addbias2 = Bias2DMean(planes)

		# Both self.conv1 and self.downsample layers downsample the input when stride != 1
		self.stride = stride
		self._scale = nn.Parameter(torch.ones(1))
		multiplier = (block_idx + 1) ** -(1 / 6) * max_block ** (1 / 6)
		multiplier = multiplier * (1 - drop_conv) ** .5

		for m in self.modules():
			if isinstance(m, nn.Conv2d) and m.weight.requires_grad:
				_, C, H, W = m.weight.shape
				stddev = (C * H * W / 2) ** -.5
				nn.init.normal_(m.weight, std = stddev * multiplier)

		self.residual = max_block ** -.5
		self.identity = block_idx ** .5 / (block_idx + 1) ** .5

		self.downsample = nn.Sequential()
		if stride != 1 or inplanes != self.expansion * planes:
			if stride == 1:
				avgpool = nn.Sequential()
			else:
				avgpool = nn.AvgPool2d(stride)

			self.downsample = nn.Sequential(avgpool, Bias2DMean(num_features = inplanes),
			                                nn.Conv2d(inplanes, self.expansion * planes, kernel_size = 1, bias = False))

			nn.init.kaiming_normal_(self.downsample[2].weight, a = 1)

		self.drop = nn.Sequential()
		if drop_conv > 0.0:
			self.drop = nn.Dropout2d(drop_conv)

	def forward(self, x):
		# Not adding dropout here.
		out = F.relu(self.drop(self.conv1(self.addbias1(x))))
		out = self.drop(self.conv2(self.addbias2(out)))
		out = out * self.residual * self._scale + self.identity * self.downsample(x)
		out = F.relu(out)
		return out

	def init_pass(self, x, count):
		out = F.relu(self.drop(self.conv1(self.addbias1.init_pass(x, count))))
		out = self.drop(self.conv2(self.addbias2.init_pass(out, count)))
		out = out * self.residual * self._scale + self.identity * self.downsample(x)
		out = F.relu(out)
		return out


class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, inplanes, planes, block_idx, max_block, stride = 1, groups = 1, base_width = 64, drop_conv = 0.0):
		super(Bottleneck, self).__init__()
		width = int(planes * (base_width / 64.)) * groups
		self.conv1 = nn.Conv2d(inplanes, width, kernel_size = 1, bias = False)
		self.conv2 = nn.Conv2d(width, width, kernel_size = 3, padding = 1, stride = stride, groups = groups, bias = False)
		self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size = 1, bias = False)

		self.addbias1 = Bias2DMean(inplanes)
		self.addbias2 = Bias2DMean(width)
		self.addbias3 = Bias2DMean(width)

		self._scale = nn.Parameter(torch.ones(1))
		multiplier = (block_idx + 1) ** -(1 / 6) * max_block ** (1 / 6)
		multiplier = multiplier * (1 - drop_conv) ** .5

		for m in self.modules():
			if isinstance(m, nn.Conv2d) and m.weight.requires_grad:
				_, C, H, W = m.weight.shape
				stddev = (C * H * W / 2) ** -.5
				nn.init.normal_(m.weight, std = stddev * multiplier)

		self.residual = max_block ** -.5
		self.identity = block_idx ** .5 / (block_idx + 1) ** .5

		self.downsample = nn.Sequential()
		if stride != 1 or inplanes != self.expansion * planes:
			if stride == 1:
				avgpool = nn.Sequential()
			else:
				avgpool = nn.AvgPool2d(stride)

			self.downsample = nn.Sequential(avgpool, Bias2DMean(num_features = inplanes),
			                                nn.Conv2d(inplanes, self.expansion * planes, kernel_size = 1, bias = False))
			nn.init.kaiming_normal_(self.downsample[2].weight, a = 1)

		self.drop = nn.Sequential()
		if drop_conv > 0.0:
			self.drop = nn.Dropout2d(drop_conv)

	def forward(self, x):
		out = F.relu(self.drop(self.conv1(self.addbias1(x))))
		out = F.relu(self.drop(self.conv2(self.addbias2(out))))
		out = self.drop(self.conv3(self.addbias3(out)))
		out = out * self.residual * self._scale + self.identity * self.downsample(x)
		out = F.relu(out)
		return out

	def init_pass(self, x, count):
		out = F.relu(self.drop(self.conv1(self.addbias1.init_pass(x, count))))
		out = F.relu(self.drop(self.conv2(self.addbias2.init_pass(out, count))))
		out = self.drop(self.conv3(self.addbias3.init_pass(out, count)))
		out = out * self.residual * self._scale + self.identity * self.downsample(x)
		out = F.relu(out)
		return out


class ReScale(nn.Module):
	def __init__(self,
	             layers,
	             num_classes = 1000,
	             groups = 1,
	             width_per_group = 64,
	             drop_conv = 0.0,
	             drop_fc = 0.0,
	             block = Bottleneck,
	             input_shapes = (None, None),
	             num_flexible_classes = -1):
		super(ReScale, self).__init__()

		self.inplanes = 64
		self.num_classes = num_classes
		self.input_shapes = input_shapes
		self.groups = groups
		self.base_width = width_per_group
		self.block_idx = sum(layers) - 1
		self.max_depth = sum(layers)
		self.num_flexible_classes = num_flexible_classes

		# KT TEST SPLIT LEARNING
		# self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
		# self.addbias1 = Bias2DMean(self.inplanes)
		# self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride = 2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride = 2, drop_conv = drop_conv)
		self.layer4 = self._make_layer(block, 512, layers[3], stride = 2, drop_conv = drop_conv)
		self.addbias2 = Bias2DMean(512 * block.expansion)
		self.drop = nn.Dropout(drop_fc)
		self.fc = nn.Linear(512 * block.expansion, num_classes)

		# KT DEBUG
		#self.mean_pool = nn.AvgPool2d((input_shapes[0] // 8, input_shapes[1] // 8))
		self.mean_pool = nn.AvgPool2d((input_shapes[0] // 32, input_shapes[1] // 32))

		# KT TEST SPLIT LEARNING
		# nn.init.kaiming_normal_(self.conv1.weight)
		nn.init.kaiming_normal_(self.fc.weight, a = 1)

		if self.num_flexible_classes != -1:
			_fixed_sum_layer = torch.zeros(num_classes)
			num_unused_classes = num_classes - self.num_flexible_classes
			if num_unused_classes > 0:
				_fixed_sum_layer[self.num_flexible_classes:] = torch.ones(num_unused_classes) * -10000.0
				# initialize bias and weight of unused to 0
				self.fc.bias.data[self.num_flexible_classes:] = 0
				self.fc.weight.data[self.num_flexible_classes:, :] = 0

			# make the fixed_mask not trainable
			self.register_buffer("fixed_sum_layer", _fixed_sum_layer)

	def _make_layer(self, block, planes, num_blocks, stride = 1, drop_conv = 0.0):
		strides = [stride] + [1] * (num_blocks - 1)
		layers = []
		for stride in strides:
			layers.append(
				block(self.inplanes,
				      planes,
				      block_idx = self.block_idx,
				      max_block = self.max_depth,
				      stride = stride,
				      groups = self.groups,
				      base_width = self.base_width,
				      drop_conv = drop_conv))
			self.inplanes = planes * block.expansion
			self.block_idx += 1
		return nn.Sequential(*layers)

	def forward(self, x):
		# KT TEST SPLIT LEARNING
		# x = self.conv1(x)
		# x = self.addbias1(x)
		# x = self.relu(x)
		# x = self.maxpool(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.addbias2(x)

		x = self.mean_pool(x)
		x = x.squeeze(-1).squeeze(-1)
		x = self.drop(x)
		x = self.fc(x)
		if self.num_flexible_classes != -1:
			x = x + self.fixed_sum_layer

		return x

def rescale18(num_classes = 10, drop_conv = 0.0, drop_fc = 0.0, **kwargs):
	return ReScale([2, 2, 2, 2],
	               num_classes = num_classes,
	               drop_conv = drop_conv,
	               input_shapes = (32, 32),  #Following the KT debug, Orig values: (8, 8) Values passed during compilation: (32, 32)
	               drop_fc = drop_fc,
	               groups = 1,
	               width_per_group = 64,
	               block = BasicBlock)

resnet18_server = rescale18()
# KT DEBUG: Comment out temp for torch summary
samba.from_torch_model_(resnet18_server.bfloat16())

criterion = nn.CrossEntropyLoss()
lr = 0.001
# KT DEBUG
# optimizer = samba.optim.SGD(resnet18_server.parameters(), lr = lr, momentum = 0.9)

def init_optim(model: torch.nn.Module, optimizer_type: str, lr: float, weight_decay: float) -> torch.optim:
	"""
	Initialize optimizer based on the model type.
	"""
	
	params_w_decay = []
	params_wo_decay = []
	for name, p in model.named_parameters():
		if p.requires_grad:
			if 'addbias' in name or '_scale' in name:
				params_wo_decay.append(p)
			else:
				params_w_decay.append(p)
	if optimizer_type == 'adamw':
		optim = [
			samba.optim.AdamW(params_wo_decay, lr=lr, betas=(0.9, 0.997), weight_decay=0),
			samba.optim.AdamW(
				params_w_decay, lr=lr, betas=(0.9, 0.997), weight_decay=weight_decay)
		] 
	elif optimizer_type == 'sgd':
		optim = [
			samba.optim.SGD(params_wo_decay, lr=lr, weight_decay=0, momentum=0.9),
			samba.optim.SGD(
				params_w_decay, lr=lr, weight_decay=weight_decay, momentum=0.9)
		] 

	return optim
optimizer = init_optim(resnet18_server, optimizer_type='adamw', lr=lr, weight_decay=0.0)
optimizer = optimizer[0] #SP DEBUG
print(optimizer)

#KT DEBUG: Comment out temporariy
host = '10.19.64.32' #SN30_Node1 - 10.19.64.32, A*Star using IP - 10.9.240.14
# host = '10.2.16.246'
port = 8889
seed = 777
sn_utils.set_seed(seed)
s = socket.socket()
s.bind((host, port))
s.listen(5)

print('Waiting for client')

conn, addr = s.accept()
print("Connected to: ", addr)

# read epoch
rmsg, data_size = recv_msg(conn)  # receive total bach number and epoch from client.

epoch = rmsg['epoch']
batch_size = rmsg['batch_size']
total_batch = rmsg['total_batch']

print("received epoch: ", rmsg['epoch'], rmsg['total_batch'])

send_msg(conn, server_name)  # send server meta information.

# Start training
start_time = time.time()
print("Start training @ ", time.asctime())

# batch_size = 256 # SP DEBUG
inputs = get_inputs(batch_size)
'''
print("-------------------------------------Got Inputs---------------------------------", inputs[0].shape)

# SP Debug to extract the model defination and input summary to comapre it with the compile's .pef model. 
summary(resnet18_server, input_data=inputs[0], col_names=["input_size", "output_size", "num_params", "kernel_size"], verbose=1)

print('--------------------------------------Ok Summary Done-----------------------------------')
'''
# SP DEBUG : Have added inputs[0] instead of just inputs access the tensors itself rather than tuple containing the tensors.
sn_utils.trace_graph(resnet18_server, inputs[0], optimizer, pef = "./rescale18_split/rescale18_split.pef")
print("Traced graph successfully")
# sys.exit()


for epc in range(epoch):
	init = 0
	loss_list = []
	acc_list = []
	for i in tqdm(range(total_batch), ncols = 100, desc = 'Training with {}'.format(server_name)):
		# optimizer.zero_grad()

		msg, data_size = recv_msg(conn)  # receives label and feature from client.

		# label
		labels = msg['label']
		labels = labels.clone().detach().long()  # conversion between gpu and cpu.
		# feature
		client_output = msg['client_output']

		labels = samba.from_torch_tensor(labels, name = 'label', batch_dim = 0)
		client_output = samba.from_torch_tensor(client_output, name = 'input', batch_dim = 0).bfloat16()
		sn_utils.trace_graph(resnet18_server, (client_output, labels), optimizer, pef = "/nvmedata/scratch/shrirajp/SambaNova_Astar/rescale18_split/rescale18_split.pef")
		#pef = "/home/liuw2/python-projects/test/rescale18_split/rescale18_split.pef")
		

		loss, outputs = samba.session.run(input_tensors = [client_output, labels],
		                                  output_tensors = resnet18_server.output_tensors)
		# Convert SambaTensors back to Torch Tensors to calculate accuracy
		loss, outputs = samba.to_torch(loss), samba.to_torch(outputs)

		loss_list.append(loss.item())
		# track accuracy
		total = labels.size(0)
		_, predicted = torch.max(outputs.data.float(), 1)
		correct = (predicted == labels).sum().item()
		acc_list.append(correct / total)

		print(f"Epoch {epc + 1} / {epoch}: Avg. loss {mean(loss_list):.1f} / Accuracy {mean(acc_list):.3f}")
		# send gradient to client
		msg = client_output.grad.clone().detach()
		data_size = send_msg(conn, msg)

print('Contribution from {} is done'.format(server_name))
print('Contribution duration is: {} seconds'.format(time.time() - start_time))
