#!/usr/bin/env python3

import argparse
import os
import re

import torch


AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWYXBZUOJ")



def str2bool(value):
	if isinstance(value,bool):
		return value
	if value.lower() in ('yes','true','t','y','1'):
		return True
	if value.lower() in ('no','false','f','n','0'):
		return False
	raise argparse.ArgumentTypeError('Boolean value expected.')


def read_sequences(seq_path):
	names = []
	seqs = []
	with open(seq_path,'r') as file:
		for line in file:
			tmp = re.split(',|\t',line.strip('\n'))
			if len(tmp) < 2:
				continue
			names.append(tmp[0])
			seqs.append(clean_sequence(tmp[1]))
	return names, seqs


def clean_sequence(seq):
	return ''.join(aa if aa in AMINO_ACIDS else 'X' for aa in seq)


def chunk_sequence(seq,chunk_size=1022,overlap=128):
	if chunk_size <= 0:
		raise ValueError('ESM2 chunk size must be positive')
	if overlap >= chunk_size:
		raise ValueError('ESM2 chunk overlap must be smaller than chunk size')
	if len(seq) <= chunk_size:
		return [seq]
	chunks = []
	step = chunk_size - overlap
	for start in range(0,len(seq),step):
		chunk = seq[start:start+chunk_size]
		if chunk:
			chunks.append(chunk)
		if start + chunk_size >= len(seq):
			break
	return chunks


def normalize_model_name(model_name):
	return model_name.split('/')[-1].replace('-','_')


def load_esm_model(model_name,device):
	try:
		import esm
	except ImportError as exc:
		raise ImportError('ESM2 precompute requires fair-esm. Install it with: pip install fair-esm') from exc

	model_key = normalize_model_name(model_name)
	if not hasattr(esm.pretrained,model_key):
		raise ValueError(f'Unsupported fair-esm model: {model_name} ({model_key})')
	model, alphabet = getattr(esm.pretrained,model_key)()
	model = model.to(device)
	model.eval()
	return model, alphabet


def model_hidden_size(model):
	return model.embed_dim


def pooled_hidden_size(model,pooling):
	hidden_size = model_hidden_size(model)
	if pooling == 'mean_max':
		return hidden_size * 2
	return hidden_size


def embed_one_sequence(seq,batch_converter,model,repr_layer,device,chunk_size,overlap,pooling):
	if len(seq) == 0:
		return torch.zeros(pooled_hidden_size(model,pooling),dtype=torch.float)

	total = None
	total_count = 0
	max_values = None
	for i, chunk in enumerate(chunk_sequence(seq,chunk_size,overlap)):
		_, _, tokens = batch_converter([('protein',chunk)])
		tokens = tokens.to(device)
		output = model(tokens,repr_layers=[repr_layer],return_contacts=False)
		hidden = output['representations'][repr_layer][0].detach().cpu()
		residue_hidden = hidden[1:len(chunk)+1]
		if residue_hidden.numel() == 0:
			continue
		# Skip the overlapping prefix on all chunks after the first so each
		# original sequence position is counted exactly once in the mean.
		skip = overlap if i > 0 else 0
		unique_hidden = residue_hidden[skip:]
		if unique_hidden.numel() == 0:
			continue
		chunk_sum = unique_hidden.sum(dim=0)
		chunk_max = unique_hidden.max(dim=0).values
		total = chunk_sum if total is None else total + chunk_sum
		max_values = chunk_max if max_values is None else torch.maximum(max_values,chunk_max)
		total_count += unique_hidden.shape[0]

	if total is None or total_count == 0:
		return torch.zeros(pooled_hidden_size(model,pooling),dtype=torch.float)
	mean_values = total / total_count
	if pooling == 'mean':
		return mean_values
	if pooling == 'max':
		return max_values
	if pooling == 'mean_max':
		return torch.cat([mean_values,max_values],dim=0)
	raise ValueError(f'Unsupported pooling mode: {pooling}')


def precompute(args):
	if os.path.isfile(args.output) and not args.overwrite:
		print(f'ESM2 cache already exists: {args.output}')
		print('use --overwrite true to regenerate it')
		return

	names, seqs = read_sequences(args.input)
	device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
	model, alphabet = load_esm_model(args.model,device)
	batch_converter = alphabet.get_batch_converter()
	repr_layer = model.num_layers

	embeddings = []
	with torch.no_grad():
		for start in range(0,len(seqs),args.batch_size):
			batch = seqs[start:start+args.batch_size]
			for seq in batch:
				embeddings.append(embed_one_sequence(seq,batch_converter,model,repr_layer,device,args.chunk_size,args.chunk_overlap,args.pooling))
			print(f'encoded ESM2 proteins: {min(start+args.batch_size,len(seqs))}/{len(seqs)}')

	embeddings = torch.stack(embeddings,dim=0).cpu().float()
	output_dir = os.path.dirname(args.output)
	if output_dir:
		os.makedirs(output_dir,exist_ok=True)
	torch.save({
		'model_name': args.model,
		'model_backend': 'fair-esm',
		'repr_layer': repr_layer,
		'sequence_file': args.input,
		'protein_names': names,
		'protein_count': len(seqs),
		'chunk_size': args.chunk_size,
		'chunk_overlap': args.chunk_overlap,
		'pooling': args.pooling,
		'embeddings': embeddings,
	},args.output)
	print(f'saved ESM2 embeddings to {args.output}')


def get_args():
	parser = argparse.ArgumentParser(description='Precompute frozen ESM2 protein embeddings for HI-PPI.')
	parser.add_argument('-i','--input',required=True,help='sequence dictionary file')
	parser.add_argument('-o','--output',required=True,help='output .pt cache path')
	parser.add_argument('--model',default='facebook/esm2_t6_8M_UR50D',help='fair-esm ESM2 model name')
	parser.add_argument('--batch-size',default=8,type=int,help='number of proteins to encode before progress logging')
	parser.add_argument('--chunk-size',default=1022,type=int,help='ESM2 residue chunk size for long proteins')
	parser.add_argument('--chunk-overlap',default=128,type=int,help='overlap between ESM2 chunks for long proteins')
	parser.add_argument('--pooling',default='mean',choices=['mean','max','mean_max'],help='residue-to-protein pooling mode')
	parser.add_argument('--cuda',default=True,type=str2bool,help='use CUDA when available')
	parser.add_argument('--overwrite',default=False,type=str2bool,help='overwrite an existing output cache')
	return parser.parse_args()


if __name__ == '__main__':
	precompute(get_args())
