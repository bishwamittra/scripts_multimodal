#!/usr/bin/python
# encoding: utf-8

# Copyright © 2020 by SambaNova Systems, Inc. Disclosure, reproduction,
# reverse engineering, or any other use made without the advance written
# permission of SambaNova Systems, Inc. is unauthorized and strictly
# prohibited. All rights of ownership and enforcement are reserved.
from torchinfo import summary
import argparse
import sambaflow.samba.utils as sn_utils
# Imports for compiling the .pef
import sys

# Appending path to import certain files/modules like (rescale, layers, rescale_estimator, schema, etc).
# sys.path.append("/nvmedata/scratch/karent/aston/shriraj_0815/SambaNova_Astar/classification_1.16.5-38")

import torch
import torch.nn as nn
# KT DEBUG
# To include meta data in pef
from sambaflow import __version__ as sambaflow_version
from sambaflow import samba
from sambaflow.samba.env import (enable_addbias_grad_accum_stoc,
                                 enable_conv_grad_accum_stoc)
from sambaflow.samba.test import consistency_test
from sambaflow.samba.utils.argparser import parse_app_args
from sambaflow.samba.utils.numeric import MAX_BF16_TRUNCATE_NORMAL_REL
from sambaflow.samba.utils.pef_utils import get_pefmeta
from tqdm import tqdm
from typing import List, Tuple

from classification.schema import schema
from rescale_estimator import RescaleEstimator


# imports for running/training server_split side.
server_name = 'SambaNova'
import pickle
import socket
import struct
import time
from statistics import mean


def add_args(parser: argparse.ArgumentParser):
	# --compute-input-grad and --hot-layers conflict.
	# input_grad cannot be computed when intermediate layers do not require grad.
	grad_group = parser.add_mutually_exclusive_group()

	# Model arguments
	parser.add_argument('--in-height', type = int, default = 512, help = 'Height of the input image')
	parser.add_argument('--in-width', type = int, default = 512, help = 'Width of the input image')
	parser.add_argument('--channels', type = int, default = 3, help = 'Width of the input image')

	# TODO: Remove once addbias is supported for tiling
	grad_group.add_argument('--compute-input-grad', action = "store_true", help = 'Compute input grad.')
	parser.add_argument('--num-classes', type = int, default = 2, help = 'Number of classes')
	parser.add_argument('--num-flexible-classes',
	                    type = int,
	                    default = -1,
	                    help = 'Number of eligible classes. This feature is turned off by default')
	parser.add_argument("--optimizer",
	                    type = str,
	                    default = "adamw",
	                    choices = ["sgd", "adamw"],
	                    help = "pick between adamw and sgd")
	parser.add_argument('--learning-rate', type = float, default = 0.0001, help = 'initial lr (default: 0.001)')
	parser.add_argument('--weight-decay', type = float, default = 0.0, help = 'weight decay for optimizer (default: 0.)')
	parser.add_argument('--momentum', type = float, default = 0.9, help = 'momentum for SGD optimizer')
	parser.add_argument('--drop-conv', type = float, default = 0.03)
	parser.add_argument('--drop-fc', type = float, default = 0.3)
	# KT DEBUG
	#parser.add_argument('--command', type = str, default = "run")

	# TODO: Add support for more models.
	parser.add_argument('--model',
	                    type = str,
	                    default = "rescale18",
	                    choices = ["rescale18", "rescale50", "resnet18", "resnet50"])
	grad_group.add_argument(
		'--hot-layers',
		type = int,
		nargs = '*',
		default = None,  # There are only 4 layers for Rescale.
		choices = [1, 2, 3, 4],
		help = 'Layers to unfreeze for training. Default is all layers.')
	parser.add_argument('--weighted-cross-entropy', action = 'store_true', help = 'Use weighted cross entropy')
	parser.add_argument('--class-weights',
	                    type = float,
	                    nargs = '*',
	                    help = 'Use weighted cross entropy, using inverse class frequency, normalized to sum to 1.')

	# Additional testing/debug arguments
	parser.add_argument('--enable-stoc-rounding', action = "store_true", help = 'Enable SOTC Rounding')
	parser.add_argument("--device", type = str, default = "RDU", choices = ["RDU", "GPU", "CPU"], help = "Device to run on")
	# Benchmarking (measure-performance) args
	parser.add_argument("--min-throughput", type = float, default = -1)

	parser.add_argument('--use-sambaloader', action = 'store_true', help = 'Use Samba Data Loader. Supported in inference')
	parser.add_argument('--run-benchmark', action = 'store_true', help = 'Benchmark end-to-end performance')
	parser.add_argument('--pinned-memory', action = 'store_true')
	# parser.add_argument('--data-parallel', action = 'store_true')

	# Benchmarking inference arguments
	parser.add_argument("--benchmark-steps", type = int, default = 100)
	parser.add_argument("--benchmark-warmup-steps", type = int, default = 0)

	# KT: TEST SPLIT LEARNING
	parser.add_argument('--orig-in-height', type = int, default = 512, help = 'ORIGINAL Height of the input image')
	parser.add_argument('--orig-in-width', type = int, default = 512, help = 'ORIGINAL Width of the input image')


@consistency_test()
def run_consistency_test(args: argparse.Namespace, model: nn.Module, inputs: Tuple[samba.SambaTensor],
                         outputs: Tuple[samba.SambaTensor]) -> List[Tuple[str, samba.SambaTensor]]:
	model.bfloat16().float()
	model.zero_grad()
	inputs[0].grad = None

	idx = 0
	inputs_gold = inputs[0].float()
	output_gold = model(inputs_gold)
	output_samba = samba.session.run(input_tensors = inputs,
	                                 output_tensors = outputs,
	                                 section_types = ['fwd'],
	                                 data_parallel = args.data_parallel,
	                                 reduce_on_rdu = args.reduce_on_rdu)
	output_samba = output_samba[idx]
	outputs_vars = [("output_samba", output_samba), ("output_gold", output_gold)]

	if not args.inference:
		output_grad = samba.randn_like(output_samba.data).bfloat16().float()
		section_types = ['bckwd'] if not args.data_parallel else ['bckwd', 'reduce']
		outputs[0].sn_grad = output_grad
		samba_outputs = samba.session.run(input_tensors = inputs,
		                                  section_types = section_types,
		                                  output_tensors = [output_samba],
		                                  data_parallel = args.data_parallel,
		                                  reduce_on_rdu = args.reduce_on_rdu)[0]
		output_gold.backward(samba.to_torch(output_grad))

		if (args.compute_input_grad):
			# check input grad
			gold_input_grad = inputs[0].grad
			samba_input_grad = inputs[0].sn_grad
			outputs_vars += [("samba_input_grad", samba_input_grad), ("gold_input_grad", gold_input_grad)]

		# check weight grads of all params.
		for name, param in model.named_parameters():
			if not param.requires_grad:
				continue
			gold_weight_grad = param.grad
			samba_weight_grad = param
			outputs_vars += [("samba_weight_grad", samba_weight_grad), ("gold_weight_grad", gold_weight_grad)]
	return outputs_vars


def run_functional_test(args: argparse.Namespace, model: nn.Module, inputs: Tuple[samba.SambaTensor],
                        outputs: Tuple[samba.SambaTensor]) -> List[Tuple[str, samba.SambaTensor]]:
	model.bfloat16().float()
	model.zero_grad()
	inputs[0].grad = None

	idx = 0
	inputs_gold = inputs[0].float()
	output_gold = model(inputs_gold)

	output_samba = samba.session.run(input_tensors = inputs,
	                                 output_tensors = outputs,
	                                 section_types = ['fwd'],
	                                 data_parallel = args.data_parallel,
	                                 reduce_on_rdu = args.reduce_on_rdu)
	output_samba = output_samba[idx]

	# check that all samba and torch outputs1match numerically
	print(f'samba output abs sum: {output_samba.abs().sum()}')
	print(f'gold output abs sum: {output_gold.abs().sum()}')
	sn_utils.assert_close(output_samba.float(),
	                      output_gold,
	                      f'output',
	                      threshold = 0.05,
	                      rtol = MAX_BF16_TRUNCATE_NORMAL_REL,
	                      visualize = False)

	if not args.inference:
		output_grad = samba.randn_like(output_samba.data).bfloat16().float()
		section_types = ['bckwd'] if not args.data_parallel else ['bckwd', 'reduce']
		outputs[0].sn_grad = output_grad
		samba_outputs = samba.session.run(input_tensors = inputs,
		                                  section_types = section_types,
		                                  output_tensors = [output_samba],
		                                  data_parallel = args.data_parallel,
		                                  reduce_on_rdu = args.reduce_on_rdu)[0]
		output_gold.backward(samba.to_torch(output_grad))

		if (args.compute_input_grad):
			# check input grad
			gold_input_grad = inputs[0].grad
			samba_input_grad = inputs[0].sn_grad
			print(f'\ngold_input_grad_abs_sum: {gold_input_grad.abs().sum()}')
			print(f'\nsamba_input_grad_abs_sum: {samba_input_grad.abs().sum()}')
			sn_utils.assert_close(samba_input_grad,
			                      gold_input_grad,
			                      'input_grad',
			                      threshold = 0.1,
			                      rtol = 0.01,
			                      visualize = False)

		# compute the thresholds based on image sizes.
		threshold = 0.05
		scale_threshold = 0.05

		if args.in_height == 32:
			threshold = 0.2

		if args.in_height == 64:
			threshold = 0.25

		if args.in_height == 512:
			threshold = 0.6
			scale_threshold = 0.6

		if args.in_height == 1024:
			threshold = 0.3

		if args.in_height == 2048:
			threshold = 0.6

		# check weight grads of all params.
		for name, param in model.named_parameters():
			if not param.requires_grad:
				continue

			gold_weight_grad = param.grad
			samba_weight_grad = param.sn_grad

			print(f'gold_{name}: {gold_weight_grad.abs().sum()}')
			print(f'samba_{name}: {samba_weight_grad.abs().sum()}')
			thresh = scale_threshold if "scale" in name else threshold

			sn_utils.assert_close(samba_weight_grad,
			                      gold_weight_grad,
			                      name,
			                      threshold = thresh,
			                      rtol = 1.5,
			                      visualize = False)


def get_inputs(args: argparse.Namespace) -> Tuple[samba.SambaTensor, ...]:
	tensor = samba.randn(args.batch_size, args.channels, args.in_height, args.in_width, name = 'input', batch_dim = 0).bfloat16()

	if not args.inference:
		tensor.requires_grad_(args.compute_input_grad)
	return (tensor,)


# Client-Server connection functions to send/recieve tensors.
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


def main(argv: List[str]):
	# Set random seed for reproducibility.
	sn_utils.set_seed(256)

	# Get common args and any user added args.
	args = parse_app_args(argv = argv, common_parser_fn = add_args)
	cfg = schema.convert_compile_args_to_pydantic(args)
	# KT DEBUG
	# cfg = schema.convert_compile_args_to_pydantic(args, True)
	estimator_mode = "compile"

	if args.enable_stoc_rounding:
		enable_conv_grad_accum_stoc()
		enable_addbias_grad_accum_stoc()

	# Get the estimator mode. We use estimator_mode="compile" for setting up tests to measure performance.
	# NOTE: the estimator mode and the script command are not necessary the same. e.g. the estimator might be in
	#       compile mode but the script might be running performance tests
	# If we're actually running performance tests, we will need to trace the graph.
	trace_graph = True
	estimator = RescaleEstimator.from_pydantic(cfg, estimator_mode)

	if args.inference:
		estimator.model.eval()

	# Instantiated get_inputs
	inputs = get_inputs(args)


	if args.command == 'compile':
		# If all were doing is compiling, don't trace the graph
		trace_graph = False
		# do not trace the graph when compiling
		# KT DEBUG: Print torch summary
		# summary(estimator.model, input_data=(torch.rand(args.batch_size, args.channels, args.in_height, args.in_width), ), verbose=1)
		# summary(estimator.model, input_size=(inputs[0].shape, ), col_names=["input_size", "output_size", "num_params", "kernel_size"], verbose=1)
		# To include meta data in pef
		pef_metadata = get_pefmeta(args, estimator.model)
		pef_metadata['sambaflow_version'] = sambaflow_version
		pef_metadata['input_shapes'] = (inputs[0].shape,)

		samba.session.compile(estimator.model,
		                      inputs,
		                      estimator.optimizers,
		                      name = 'rescale',
		                      init_output_grads = not args.inference,
		                      squeeze_bs_dim = True,
		                      app_dir = sn_utils.get_file_dir(__file__),
		                      config_dict = vars(args),
		                      pef_metadata = pef_metadata)  # KT DEBUG: To include meta data in pef

	# KT DEBUG
	estimator.setup(trace_graph=trace_graph) # SP DEBUG

	if args.command == "test":
		# We are testing datamismatch and consistency test separately. This is because stoc rounding cases ND.
		if args.enable_stoc_rounding:
			run_functional_test(args, estimator.model, inputs, estimator.model.output_tensors)
		else:
			run_consistency_test(args, estimator.model, inputs, estimator.model.output_tensors)

	elif args.command == 'measure-performance':
		throughput, latency = sn_utils.measure_performance(estimator.model,
		                                                   inputs,
		                                                   args.batch_size,
		                                                   args.n_chips,
		                                                   args.inference,
		                                                   run_graph_only = args.run_graph_only,
		                                                   n_iterations = args.num_iterations,
		                                                   json = args.bench_report_json,
		                                                   compiled_stats_json = args.compiled_stats_json,
		                                                   data_parallel = args.data_parallel,
		                                                   reduce_on_rdu = args.reduce_on_rdu,
		                                                   min_duration = args.min_duration,
		                                                   use_sambaloader = args.use_sambaloader,
		                                                   json_counters = args.json_counters)
		assert throughput > args.min_throughput, \
			f'Expected throughput to be at least {args.min_throughput}, instead found {throughput}'

	elif args.command == "measure-sections":
		sn_utils.measure_sections(estimator.model,
		                          inputs,
		                          num_sections = args.num_sections,
		                          n_iterations = args.num_iterations,
		                          batch_size = args.batch_size,
		                          data_parallel = args.data_parallel,
		                          reduce_on_rdu = args.reduce_on_rdu,
		                          json = args.bench_report_json,
		                          min_duration = args.min_duration)

	elif args.command == "run":
		# This is the server-side training code.
		print("Inside 'run'")


		# KT DEBUG: Comment out temporariy
		# host = '10.19.64.32'  # SN30_Node1 - 10.19.64.32, A*Star using IP - 10.9.240.14
		host = '10.9.240.14'
		port = 8890
		seed = 777
		# sn_utils.set_seed(seed) # Already assigned the seed value at the start of main fn.
		s = socket.socket()
		s.bind((host, port))
		s.listen(5)

		print('Waiting for the client')

		conn, addr = s.accept()
		print("Connected to: ", addr)

		# read epoch
		rmsg, data_size = recv_msg(conn)  # receive total bach number and epoch from client.

		epoch = rmsg['epoch']
		batch_size = rmsg['batch_size']
		total_batch = rmsg['total_batch']

		print("\nReceived epoch: ", epoch,
		      "\nRecieved batch_size: ", batch_size,
		      "\nRecieved total_batch: ", total_batch, "\n")

		# Passing Batch Size through the command line argument
		if batch_size != args.batch_size:
			print("\n\nNOTE: Make sure to pass the same batch size from the client side and through the CLI (using example '-b 256')\n\n")
			sys.exit()

		send_msg(conn, server_name)  # send server meta information.

		# Start training
		start_time = time.time()
		print("Start training @ ", time.asctime())

		# print(estimator.model)

		print("----------------------------------------Got Inputs---------------------------------------", inputs[0].shape)
		#
		# SP Debug to check/verify the model defination and input summary and to comapre it with the compile's .pef model summary.
		# NOTE: To get the summary make sure you pass --device CPU as an argument.
		# summary(estimator.model, input_size=(inputs[0].shape, ), col_names=["input_size", "output_size", "num_params", "kernel_size"], verbose=1)
		#print('--------------------------------------Ok Summary Done-----------------------------------')
		# sys.exit()

		# SP DEBUG : Have added inputs[0] instead of just inputs access the tensors itself rather than tuple containing the tensors.
		# KT DEBUG
		#sn_utils.trace_graph(estimator.model, inputs, estimator.optimizers, pef = args.pef, init_output_grads=True)
		print("Traced graph successfully")
		print("\n\n Printing Output Tensor after trace_graph:", estimator.model.output_tensors, "\n\n")
		# sys.exit()

		# pef = "/nvmedata/scratch/shrirajp/Astar_updated_code/rescale18_split/rescale18_split.pef"

		# ----------Have made changes till here-------------

		for epc in range(epoch):
			init = 0
			loss_list = []
			acc_list = []
			for i in tqdm(range(total_batch), ncols = 100, desc = 'Training with {}'.format(server_name)):
				# optimizer.zero_grad() # Not needed for SambaFlow specific training.

				msg, data_size = recv_msg(conn)  # receives label and feature from client.

				# label
				# labels = msg['label']
				# labels = labels.clone().detach().long()  # conversion between gpu and cpu.
				# feature
				client_output = msg['client_output']

				# KT DEBUG
				#labels = samba.from_torch_tensor(labels, name = 'label', batch_dim = 0)
				client_output = samba.from_torch_tensor(client_output, name = 'input', batch_dim = 0).bfloat16()
				client_output.requires_grad_(True)
				# sn_utils.trace_graph(estimator.model, inputs, estimator.optimizers, pef = args.pef)
				# pef = "/home/liuw2/python-projects/test/rescale18_split/rescale18_split.pef")

				# output = estimator.model.output_tensors
				# print("output is: ", output)
				# loss, outputs = samba.session.run(input_tensors = (inputs, ), output_tensors = labels)
				# KT DEBUG
				#outputs = samba.session.run(input_tensors = (client_output,), output_tensors = estimator.model.output_tensors)[0]
				partial_output = samba.session.run(input_tensors=(client_output,),
                                                   #input_tensors = inputs,
                                                   output_tensors=estimator.model.output_tensors,
                                                   section_types=['fwd'])[0]
				
				# Make PyTorch return the grads
				partial_output = samba.to_torch(partial_output)
				partial_output.requires_grad = True
				
				
				partial_output_torch = samba.to_torch(partial_output)
				send_msg(conn, msg = {"server_output": partial_output_torch})
				
				
				rmsg, _ = recv_msg(conn)
				server_grad = rmsg['server_grad']

				# loss = nn.CrossEntropyLoss()(partial_output, labels)
				# loss.backward()
				# partial_output.backward(server_grad)
				# estimator.model.output_tensors[0].sn_grad = partial_output.grad
				estimator.model.output_tensors[0].sn_grad = samba.from_torch_tensor(server_grad)
				# estimator.model.output_tensors[0].sn_grad = server_grad
				#print("\n\n Printing Output Tensor in the for loop before smaba.session.run:", estimator.model.output_tensors, "\n\n")
				outputs = samba.session.run(input_tensors=(client_output,),
											#input_tensors=inputs,
											output_tensors=estimator.model.output_tensors,
											#grad_of_outputs=(samba.from_torch_tensor(partial_output.grad),),
										    section_types=['bckwd', 'opt'])[0]
		
				# outputs = samba.to_torch(outputs)
				# KT DEBUG
				#labels = samba.to_torch(labels)
				# outputs, loss = estimator.training_step(client_output, labels, epc)
				# Convert SambaTensors back to Torch Tensors to calculate accuracy
				# loss, outputs = samba.to_torch(loss), samba.to_torch(outputs)
				# print(outputs)
				# print(labels)
				#loss = nn.CrossEntropyLoss()(outputs, labels)
				# loss_list.append(loss.item())
				# KT DEBUG



				client_output.grad = client_output.sn_grad
				client_output = samba.to_torch(client_output)

				# track accuracy
				# total = labels.size(0)
				# _, predicted = torch.max(outputs.data.float(), 1)
				# correct = (predicted == labels).sum().item()
				# acc_list.append(correct / total)

				# print(f"Epoch {epc + 1} / {epoch}: Avg. loss {mean(loss_list):.1f} / Accuracy {mean(acc_list):.3f}")
				# print(client_output.requires_grad)
				# send gradient to client
				# KT DEBUG
				msg = {
					"client_output_grad": client_output.grad.clone().detach()
				}
				#msg = client_output.sn_grad.clone().detach()
				#msg = inputs[0].grad.clone().detach()
				#msg = torch.randn(client_output.size())
				send_msg(conn, msg)
			
			print(estimator.model.state_dict().keys())
			print(estimator.model.state_dict()['layer3.0.conv1.weight'])
			send_msg(conn, {"server model": {k: samba.to_torch(v) for k, v in estimator.model.state_dict().items()}}) # send model to client.
			


		print('Contribution from {} is done'.format(server_name))
		print('Contribution duration is: {} seconds'.format(time.time() - start_time))
		conn.close()

if __name__ == '__main__':
	main(sys.argv[1:])
	# main(sys.argv)

'''
# Command used to "run" Compile.py file.
# NOTE: Make sure to change the path to your pef file
python compile.py run \
-b 256 \
-p /nvmedata/scratch/karent/aston/shriraj_0815/SambaNova_Astar/out/rescale18_split/rescale18_split.pef \
-v \
--in-height 8 \
--model rescale18 \
--drop-conv 0.0 \
--drop-fc 0.0 \
--mac-v2 \
--in-width 8 \
--channels 64 \
--num-classes 10 \
--orig-in-height 32 \
--orig-in-width 32 \
--device CPU # for printing the summary of the model using torchinfo
'''
