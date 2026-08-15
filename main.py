import sys
import os
import argparse
import torch
import torch.nn as nn
import math
import numpy as np
import random
import time
sys.path.append('src')
import Models
import ppi_data
import utils

def str2bool(v):
	"""
	Converts string to bool type; enables command line 
	arguments in the format of '--arg1 true --arg2 false'
	"""
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')
	return


def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)

#graph: embed1: data, edge1: local , edge2 global edges 
def train(model,data,loss_fn,optimizer,device,result_prefix=None,batch_size=512,epochs=100,scheduler=None,global_best_f1=0.0,args=None):
	with open(result_prefix+'.txt','w') as f:
		f.write('')
	#torch.backends.cudnn.benchmark =  True
	#torch.backends.cudnn.enabled =  True
	print('begin training')
	best_f1,best_epoch = 0.0,0
	result = None
	scaler = torch.cuda.amp.GradScaler()
	val_size = len(data.val_mask)
	aly_data = None
	encode_chunks = int(getattr(args,'encode_chunks',0) or 0)
	if encode_chunks > 0:
		effective_bs = math.ceil(len(data.train_mask)/encode_chunks)
		print(f'[phase-B] encode_chunks={encode_chunks}, train edges={len(data.train_mask)}, effective batch_size={effective_bs} (was {batch_size})')
		batch_size = effective_bs
	for epoch in range(epochs):
		f1_sum,loss_sum ,recall_sum,precision_sum = 0.0,0.0,0.0,0.0
		steps = math.ceil(len(data.train_mask)/batch_size)
		model.train()
		random.shuffle(data.train_mask)
		train_loss_sum = 0.0
		for step in range(steps):
			if step == steps-1:
				train_edge_id = data.train_mask[step*batch_size:]
			else:
				train_edge_id = data.train_mask[step*batch_size:(step+1)*batch_size]
			optimizer.zero_grad(set_to_none=True)
			output = model(data=data,edge_id=train_edge_id)
			label = data.edge_attr[train_edge_id]
			label = label.type(torch.FloatTensor).to(device)
			loss = loss_fn(output,label)
			if not torch.isfinite(loss):
				raise FloatingPointError(
					f'non-finite train loss at epoch={epoch+1} step={step+1}; '
					f'pair_head={getattr(args,"pair_head","unknown")}, '
					f'riemann_residual={getattr(args,"riemann_residual","unknown")}, '
					f'riemann_detach={getattr(args,"riemann_detach_encoder","unknown")}'
				)
			train_loss_sum += loss.item()
			scaler.scale(loss).backward()
			if getattr(args,'pair_head','gated') != 'gated':
				clip_norm = float(getattr(args,'grad_clip_norm',5.0) or 0.0)
				if clip_norm > 0:
					scaler.unscale_(optimizer)
					torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
			scaler.step(optimizer)
			scaler.update()
		#validation
		model.eval()
		valid_pre_result_list = []
		valid_label_list = []
		valid_loss_sum = 0.0
		#torch.save()#save model
		steps = math.ceil(len(data.val_mask) / batch_size)
		saved_pred = []
		val_use_cached_encode = hasattr(model,'_encode') and hasattr(model,'_classify')

		with torch.no_grad(): #validation set
			if val_use_cached_encode:
				x_cache = model._encode(data)
			for step in range(steps):
				if step == steps-1:
					valid_edge_id = data.val_mask[step*batch_size:]
				else:
					valid_edge_id = data.val_mask[step*batch_size:(step+1)*batch_size]

				if val_use_cached_encode:
					output = model._classify(x_cache,data.edge2,valid_edge_id)
				else:
					output = model(data=data,edge_id=valid_edge_id)

				label = data.edge_attr[valid_edge_id]
				label = label.type(torch.FloatTensor).to(device)
				loss = loss_fn(output,label)
				if not torch.isfinite(loss):
					raise FloatingPointError(
						f'non-finite valid loss at epoch={epoch+1} step={step+1}; '
						f'pair_head={getattr(args,"pair_head","unknown")}, '
						f'riemann_residual={getattr(args,"riemann_residual","unknown")}, '
						f'riemann_detach={getattr(args,"riemann_detach_encoder","unknown")}'
					)
				valid_loss_sum += loss.item()

				m = nn.Sigmoid()
				pre_result = (m(output) > 0.5).type(torch.FloatTensor).to(device)

				valid_pre_result_list.append(pre_result.cpu().data)
				valid_label_list.append(label.cpu().data)
				saved_pred.append(m(output).to(device).cpu().data )


		valid_pre_result_list = torch.cat(valid_pre_result_list, dim=0)
		valid_label_list = torch.cat(valid_label_list, dim=0)

		saved_pred = torch.cat(saved_pred,dim=0)
		#print(saved_pred.size())

		metrics = utils.Metrictor_PPI(valid_pre_result_list, valid_label_list, prob_y=saved_pred)
		record = metrics.append_result(result_prefix+'.txt',epoch+1,train_loss_sum,valid_loss_sum)
		print(record)

		recall_sum += metrics.recall
		precision_sum += metrics.pre
		f1_sum += metrics.microF1
		loss_sum += loss.item()
		valid_loss = valid_loss_sum / steps

		if best_f1 < metrics.microF1: #epoch == epochs -1 :
			best_f1 = metrics.microF1
			best_epoch = epoch
			result =  {'pred':saved_pred,'actual':valid_label_list}
			if args.aly:
				torch.save(model.state_dict(),result_prefix+'_weighs.pt' )
			#torch.save()
		if scheduler is not None:
			scheduler.step()


	if global_best_f1 < best_f1:
		global_best_f1 = best_f1
		torch.save(result,result_prefix+'.pt')

	if args.aly:
		#model.load_state_dict(torch.load(result_prefix+'_weighs.pt', weights_only=True))
		aly_data = model.compute_hiearchical_level(data=data)


	return global_best_f1, aly_data



def get_args_parser():
	parser = argparse.ArgumentParser('PwPPI',add_help=False)
	parser.add_argument('-m',default=None,type=str,help='mode, optinal value: bfs,dfs,rand,read,data')
	# parser.add_argument('-m',default='s',type=str,help='mode')
	parser.add_argument('-o',default='output',type=str)
	parser.add_argument('-t', default='default', type=str,help='for test distintct models')
	# parser.add_argument('-sf', default=None,type=str,help='optional input, contains path for sequence and relation file')
	parser.add_argument('-i',default=None,type=str,help='path for sequnce and relation file')
	parser.add_argument('-i1',default=None,type=str,help='sequence file')
	parser.add_argument('-i2',default=None,type=str,help='relation file')
	parser.add_argument('-i3',default=None,type=str,help='file path of test set indices (for read mode)')
	parser.add_argument('-i4',default=None,type=str,help='prefix for the pre generated structure')
	parser.add_argument('-s1',default='/home/user1/code/PPI4/data/structure_data',type=str,help='file path for structure file')
	#parser.add_argument('-i4',default='../data/map1.csv',type=str,help='file path for map STRING id to uniref id')
	parser.add_argument('-e',default=50,type=int,help='epochs')
	parser.add_argument('-b', default=256, type=int,help='batch size')
	parser.add_argument('-ln', default=2, type=int,help='graph layer num')
	parser.add_argument('-L', default=128, type=int,help='length for sequence padding')
	parser.add_argument('-Loss', default='CE', type=str,help='loss function')
	parser.add_argument('-ff', default='CnM', type=str,help='feature fusion option, default mul')
	parser.add_argument('-hl', default=512, type=int,help='hidden layer')
	parser.add_argument('-sv',default=False,type=str2bool,help='if save dataset path')
	parser.add_argument('-cuda',default=True,type=str2bool,help='if use cuda')
	parser.add_argument('-force',default=True,type=str2bool,help='if write to existed output file')
	parser.add_argument('-mainfold',default='Hyperboloid',type=str,help='any of the following: Euclidean, Hyperboloid, PoincareBall')
	parser.add_argument('-pr',default=0.0,type=float,help='perturbation ratio')
	parser.add_argument('-seed',default=7,type=int,help='random seed for training')
	parser.add_argument('-sf',default=None,type=str,help='pfolder that contain pdb file')
	parser.add_argument('-aly', default=False, type=str2bool, help='analyze hierarchical level')
	parser.add_argument('-seq_encoder', default='esm2', choices=['esm2','handcrafted'], type=str, help='sequence encoder: esm2 replaces handcrafted PAAC/CTDT/position descriptors')
	parser.add_argument('-esm_cache', default=None, type=str, help='path to cached ESM2 protein embeddings')
	parser.add_argument('-esm_project_dim', default=80, type=int, help='ESM2 projection dimension before concatenating with structure features')
	parser.add_argument('-esm_zscore', default=False, type=str2bool, help='z-score cached ESM2 embeddings using train proteins only')
	parser.add_argument('-fusion_dropout', default=0.3, type=float, help='dropout inside ESM2 projection/fusion layers')
	parser.add_argument('-typed_edge_mode', default='original', choices=['original','sym'], type=str, help='whether typed relation graphs remain original or are symmetrized for message passing')
	parser.add_argument('-pair_head', default='riemann', choices=['gated','riemann','riemann_gate','riemann_branch_gate'], type=str, help='pair scoring head: original gated baseline, concat riemann residual, gated feature residual, relation-preserving gated residual')
	parser.add_argument('-riemann_residual', default=True, type=str2bool, help='when pair_head=riemann, concatenate the legacy gated pair feature as a residual skip')
	parser.add_argument('-riemann_detach_encoder', default=True, type=str2bool, help='stop gradients from the riemannian pair head into the hyperbolic encoder for stability')
	parser.add_argument('-grad_clip_norm', default=5.0, type=float, help='global grad clipping norm; applied to riemann runs by default')
	parser.add_argument('-encode_chunks', default=0, type=int, help='Phase B: number of encoder forwards per epoch. 0 disables (use -b). When K>0, train_mask is split into K super-batches, so encoder runs K times/epoch (vs len(train_mask)/b times). Fewer optimizer steps per epoch; may need lr/epoch retuning')
	return parser

#cd /home/user1/code/PPI4/HI-PPI && source /home/user1/code/PPIKG/env/bin/activate
#python3 main.py -m bfs -t HI-PPI -i data/27K.txt -i4 features/27K -o test -e 100 -mainfold Hyperboloid
#python3 main.py -m bfs -t HI-PPI -i 27K.txt -i4 features/27K -o test -e 100 -mainfold Hyperboloid
def main(args):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(torch.cuda.is_available())
	print(device)
	set_seed(args.seed)

	if args.i:
		with open(args.i,'r') as f:
			args.i1 = f.readline().strip()
			args.i2 = f.readline().strip()
	
	if not args.force:
		if os.path.isfile(args.o+'.txt'):
			print('output name already exists')
			exit()

	if args.m == 'data':
		ppi_data.generate_structure_feature(args)
		return

	PPIData = ppi_data.PPIData(args)
	data = PPIData.data
	data.to(device)
	if args.t == 'HI-PPI':
		sequence_dim = None
		model_input_dim = data.embed1.shape[-1]
		if args.seq_encoder == 'esm2':
			structure_dim = data.structure_embed.shape[-1]
			sequence_dim = data.sequence_embed.shape[-1]
			model_input_dim = structure_dim + args.esm_project_dim
		model = Models.HIPPI(model_input_dim,args=args,layer_num=args.ln,in_len=args.L,sequence_dim=sequence_dim).to(device)
	elif args.t == 'ab1':
		model = Models.ablation1(data.embed1.shape[-1],args=args,layer_num=args.ln,in_len=args.L).to(device)
	elif args.t == 'ab2':
		model = Models.ablation2(data.embed1.shape[-1],args=args,layer_num=args.ln,in_len=args.L).to(device)

	if args.Loss=='CE':
		loss_fn = nn.BCEWithLogitsLoss().to(device)
	elif args.Loss=='AS':
		loss_fn = Models.AsymmetricLossOptimized(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True).to(device)
	if args.seq_encoder == 'esm2':
		projector_params = []
		other_params = []
		for name, param in model.named_parameters():
			if not param.requires_grad:
				continue
			if name.startswith('initial_feature.sequence_proj'):
				projector_params.append(param)
			else:
				other_params.append(param)
		optimizer = torch.optim.AdamW(
			[
				{'params': projector_params, 'lr': 5e-5},
				{'params': other_params, 'lr': 1e-4},
			],
			weight_decay=1e-2,
		)
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
			optimizer,
			T_max=args.e,
			eta_min=1e-5,
		)
	else:
		optimizer = torch.optim.Adam(model.parameters(),lr=0.001, weight_decay=5e-4)
		scheduler = None


	start = time.time()
	best_f1,aly_data = train(model,data,loss_fn,optimizer,device,args.o,args.b,args.e,scheduler,0.0,args)
	end = time.time()
	mins = (end - start)/60

	print(f'\nbest F1 score: {best_f1:.4f}, running time: {mins:.2f} minutes')
	print(f'output save to {args.o}.txt')
	if args.o:
		with open(args.o+'.txt','r+') as file:
			file_data = file.read()
			file.seek(0,0)
			command = ' '.join(arg for arg in sys.argv)
			line = f'command: {command}\n'
			line += f'best F1 score: {best_f1:.4f}\n'
			line += f'training time: {mins:.2f} minutes\n'
			line += f'mode: {args.m}\n'
			line += f'layer num: {args.ln}\n'
			line += f'seed: {args.seed}\n'
			line += f'filePath: {args.i1} {args.i2}\n'
			if args.i3:
				line += f'valid set path: {args.i3}\n'
			line += f'model: {args.t}\n'
			line += f'Loss function: {args.Loss}\n'
			line += f'max length of seqs: {args.L}\n'
			line += f'sequence encoder: {args.seq_encoder}\n'
			if args.seq_encoder == 'esm2':
				line += f'ESM2 cache: {args.esm_cache}\n'
				line += f'ESM2 projection dim: {args.esm_project_dim}\n'
				line += f'ESM2 z-score: {args.esm_zscore}\n'
			line += f'epoch: {args.e}\n'
			line += f'feature fusion mode: {args.ff}\n'
			line += f'mainfold: {args.mainfold}\n'
			line += f'typed edge mode: {args.typed_edge_mode}\n'
			line += f'pair head: {args.pair_head}\n'
			line += f'riemann residual: {args.riemann_residual}\n'
			line += f'riemann detach encoder: {args.riemann_detach_encoder}\n'
			line += f'grad clip norm: {args.grad_clip_norm}\n'

			file.write(line + '\n' + file_data)


	if args.aly:
		PPIData.analyze_hiearchical(aly_data)

	return best_f1

#python3 main.py -m read -t HPPI19 -i 27K.txt -i3 /home/user1/code/PPIKG/multiSet/o2/27k_bfs10.data -i4 features/27K -o ../result/test -e 100 -mainfold Hyperboloid
if __name__ == "__main__":
	parser = argparse.ArgumentParser('PPIM', parents=[get_args_parser()])
	args = parser.parse_args()	
	best_f1 = main(args)

