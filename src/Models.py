import torch
import torch.nn as nn
import math
import random
import torch
import torch_geometric
import torch.nn.functional as F
import torch_geometric.nn.conv as Conv
from torch_geometric.typing import OptTensor
import numpy as np
import dgl
from torch.nn import Parameter
from dgl.nn.pytorch import GraphConv, GINConv, HeteroGraphConv
import mainfold



class ESMSequenceReplacement(nn.Module):
	def __init__(self,sequence_dim,sequence_project_dim=80,dropout=0.3):
		super(ESMSequenceReplacement, self).__init__()
		hidden_dim = max(256,sequence_project_dim)
		self.sequence_proj = nn.Sequential(
			nn.Linear(sequence_dim,hidden_dim),
			nn.LayerNorm(hidden_dim),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(hidden_dim,sequence_project_dim),
			nn.LayerNorm(sequence_project_dim),
		)

	def forward(self,data):
		sequence_embed = self.sequence_proj(data.sequence_embed)
		return torch.cat([data.structure_embed,sequence_embed],dim=1)


class RawInitialFeature(nn.Module):
	def forward(self,data):
		return data.embed1


class HIPPI(nn.Module):
	def __init__(self,input_dim,args=None,act='relu',layer_num=2,radius=None,dropout=0.0,if_bias=True,use_att=0,local_agg=0,feature_fusion='CnM',class_num =7,in_len=512,device=None,sequence_dim=None):
		super(HIPPI, self).__init__()
		self.models = torch.nn.ModuleList()#seven independent GNN models
		self.layer_num = layer_num
		self.class_num = class_num
		self.feature_fusion = feature_fusion
		#self.f1_transform = 64
		self.layer_num = layer_num
		self.in_len = in_len
		self.input_dim = input_dim
		#self.long_conv = hyena.HyenaOperator(d_model=input_dim,l_max=in_len)#
		#self.fc1 = nn.Linear(math.floor( in_len / pool_size),self.f1_transform )
		self.hyper_dim = int(self.input_dim/2)
		self.mainfold_name = args.mainfold
		self.mainfold = getattr(mainfold,self.mainfold_name)()#mainfold.Hyperboloid()#
		self.feature_fusion = feature_fusion
		self.layer_num = layer_num
		self.device = device
		self.pair_head_name = getattr(args,'pair_head','riemann')
		self.use_riemann_residual = bool(getattr(args,'riemann_residual',True))
		self.riemann_detach_encoder = bool(getattr(args,'riemann_detach_encoder',True))
		if sequence_dim is not None:
			self.initial_feature = ESMSequenceReplacement(
				sequence_dim,
				getattr(args,'esm_project_dim',80),
				getattr(args,'fusion_dropout',dropout),
			)
		else:
			self.initial_feature = RawInitialFeature()
		dims = [self.input_dim] + ([self.hyper_dim] * (layer_num))

		if self.mainfold.name == 'Hyperboloid':
			dims[0] += 1		
		n_curvatures = class_num + 1
		self.radius = radius
		if radius is None:
			self.curvatures = nn.ParameterList(
				[nn.Parameter(torch.ones(1), requires_grad=False) for _ in range(n_curvatures)]
			)
		else:
			self.curvatures = nn.ParameterList(
				[nn.Parameter(torch.tensor([radius], dtype=torch.float32), requires_grad=False) for _ in range(n_curvatures)]
			)

		act = getattr(torch.nn.functional, act)
		acts = [act] * (layer_num)
		for c in range(class_num):
			graph_layers = []
			i = 0
			c_in, c_out = self.curvatures[0], self.curvatures[c+1]
			in_dim, out_dim = dims[i], dims[i+1]
			graph_layers.append(HyperbolicGCN(self.mainfold,in_dim,out_dim,c_in, c_out,dropout,acts[i],if_bias,use_att,local_agg))
			graph_layers.append(HyperbolicDecoder(self.mainfold_name,dims[-2],dims[-1],if_bias,dropout,self.curvatures[c+1]))
			graph_layers.append(torch_geometric.nn.models.GIN(dims[-1],dims[-1],1,out_dim,act='tanh',norm=nn.BatchNorm1d(dims[-1])))			
			self.models.append(nn.Sequential(*graph_layers))

		hidden3 = dims[0]+1*class_num*dims[-1]
		self.GatedNetwork = GatedInteractionNetwork(hidden3,hidden3,hidden3)
		fc2_dim = hidden3
		self.fc2 = nn.Sequential(
		  nn.Linear(fc2_dim,int(fc2_dim/2)),
		  nn.ReLU(),
		  nn.Linear(int(fc2_dim/2),int(fc2_dim/4)),
		  nn.ReLU(),
		  nn.Linear(int(fc2_dim/4),class_num),
		)
		if self.pair_head_name == 'riemann':
			v1_dim = hidden3 if self.use_riemann_residual else 0
			self.riemannian_head = ReviewerSafeRiemannianHead(dims[-1], class_num=class_num, v1_dim=v1_dim)
		elif self.pair_head_name == 'riemann_gate':
			self.riemannian_adapter = ReviewerSafeRiemannianAdapter(
				dims[-1],
				base_dim=hidden3,
				class_num=class_num,
			)
			self.riemannian_head = None
		elif self.pair_head_name == 'riemann_branch_gate':
			self.riemannian_adapter = ReviewerSafeRiemannianBranchAdapter(
				dims[-1],
				base_dim=hidden3,
				class_num=class_num,
			)
			self.riemannian_head = None
		else:
			self.riemannian_head = None
		return

	def _encode(self,data):
		f1 = self.initial_feature(data)
		sparse_adj = data.sparse_adj1
		edges = data.edge1

		if self.mainfold_name == 'Hyperboloid':
			o = torch.zeros_like(f1)
			f1 = torch.cat([o[:, 0:1], f1], dim=1)
		output = [f1]

		x_tan = self.mainfold.proj_tan0(f1, self.curvatures[0])
		x_hyp = self.mainfold.expmap0(x_tan, c=self.curvatures[0])
		x_hyp = self.mainfold.proj(x_hyp, c=self.curvatures[0])

		x_hyp_list = []
		for i,m in enumerate(self.models):
			tmp = x_hyp
			tmp, _ = m[0]((tmp,sparse_adj[i]))
			x_hyp_list.append(tmp)
			tmp = m[1].forward(tmp)
			tmp = m[2](tmp,edges[i])
			output.append(tmp)

		return torch.cat(output,dim=1), x_hyp_list

	def _classify(self,encoded,edge_index,edge_id):
		x_v1, x_hyp_list = encoded
		node_id = edge_index[:, edge_id]
		x1_v1 = x_v1[node_id[0]]
		x2_v1 = x_v1[node_id[1]]
		x_v1_pair = self.GatedNetwork(x1_v1, x2_v1)
		if self.pair_head_name == 'gated':
			return self.fc2(x_v1_pair)
		if self.riemann_detach_encoder:
			x_hyp_list = [x.detach() for x in x_hyp_list]
		if self.pair_head_name in {'riemann_gate', 'riemann_branch_gate'}:
			fused_pair = self.riemannian_adapter(
				x_v1_pair,
				x_hyp_list,
				edge_index,
				edge_id,
				self.mainfold,
				self.curvatures,
			)
			return self.fc2(fused_pair)
		return self.riemannian_head(
			x_v1_pair if self.use_riemann_residual else None,
			x_hyp_list,
			edge_index,
			edge_id,
			self.mainfold,
			self.curvatures,
		)

	def forward(self,data,edge_id=None):
		x = self._encode(data)
		return self._classify(x,data.edge2,edge_id)


class HyperbolicDecoder(nn.Module):
	"""
	Decoder abstract class for node classification tasks.
	"""

	def __init__(self, mainfold_name,input_dim,output_dim,if_bias,dropout,radius):
		super(HyperbolicDecoder, self).__init__()
		self.mainfold = getattr(mainfold, mainfold_name)()
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.if_bias = if_bias
		self.linear = nn.Linear(self.input_dim, self.output_dim,bias=self.if_bias)
#, dropout, lambda x: x, 
		self.radius = radius

	def forward(self, x):

		x = self.mainfold.proj_tan0(self.mainfold.logmap0(x, c=self.radius), c=self.radius)
		x = self.linear(x)
		return x


#GIN for ablation study
class ablation1(nn.Module):
	def __init__(self,input_dim,args=None,act='relu',layer_num=2,radius=None,dropout=0.0,bias=1,use_att=0,local_agg=0,feature_fusion='CnM',class_num =7,in_len=512):
		super(ablation1, self).__init__()
		self.models = torch.nn.ModuleList()#seven independent GNN models
		self.layer_num = layer_num
		self.class_num = class_num
		self.feature_fusion = feature_fusion
		#self.f1_transform = 64
		self.layer_num = layer_num
		self.in_len = in_len
		self.input_dim = input_dim
		#self.long_conv = hyena.HyenaOperator(d_model=input_dim,l_max=in_len)#
		#self.fc1 = nn.Linear(math.floor( in_len / pool_size),self.f1_transform )
		self.hyper_dim = int(self.input_dim)
		self.mainfold_name = args.mainfold
		self.mainfold = getattr(mainfold,self.mainfold_name)()
		self.feature_fusion = feature_fusion
		self.layer_num = layer_num

		dims = [self.input_dim] + ([self.hyper_dim] * (layer_num))

		if self.mainfold.name == 'Hyperboloid':
			dims[0] += 1		
		n_curvatures = len(dims)
		self.radius = radius
		if radius is None:
			self.curvatures = [nn.Parameter(torch.Tensor([1.])) for _ in range(n_curvatures)]
		else:
			self.curvatures = [torch.tensor([radius]) for _ in range(n_curvatures)]         # fixed curvature
		self.curvatures.append(self.radius)

		act = getattr(torch.nn.functional, act)
		acts = [act] * (layer_num)
		for c in range(class_num):
			graph_layers = []
			for i in range(layer_num):
				in_dim, out_dim = dims[i], dims[i+1]
				graph_layers.append(torch_geometric.nn.models.GIN(in_dim,out_dim,1,out_dim,act='tanh',norm=nn.BatchNorm1d(out_dim)))
			self.models.append(nn.Sequential(*graph_layers))

		hidden3 = dims[0]+class_num*sum(dims[1:])	
		self.merge = GatedInteractionNetwork(hidden3,hidden3,hidden3)
		#self.fc2 = get_classifier(hidden3,class_num,feature_fusion)
		fc2_dim = hidden3*1
		self.fc2 = nn.Sequential(
		  nn.Linear(fc2_dim,int(fc2_dim/2)),
		  nn.ReLU(),
		  nn.Linear(int(fc2_dim/2),int(fc2_dim/4)),
		  nn.ReLU(),
		  nn.Linear(int(fc2_dim/4),class_num),
		)
		return

	def forward(self,data,edge_id=None):
		f1 = data.embed1 #f1 = data.encode1
		sparse_adj = data.sparse_adj1
		edges = data.edge1
		edge_index = data.edge2
		#f1 = self.fc1(f1)

		if self.mainfold_name == 'Hyperboloid':
			o = torch.zeros_like(f1)
			f1 = torch.cat([o[:, 0:1], f1], dim=1)
		output = [f1]

		for i,m in enumerate(self.models):
			tmp = f1
			for j in range(self.layer_num):
				tmp =  m[-1](tmp,edges[i])
				output.append(tmp)


		x = torch.cat(output,dim=1)
		node_id = edge_index[:, edge_id]
		x1 = x[node_id[0]]
		x2 = x[node_id[1]]

		x = torch.cat([self.merge(x1, x2)], dim=1) #torch.mul(x1, x2)
		x = self.fc2(x)
		return x

#abltion2
class ablation2(nn.Module):
	def __init__(self,input_dim,args=None,act='relu',layer_num=2,radius=None,dropout=0.0,bias=1,use_att=0,local_agg=0,feature_fusion='CnM',class_num =7,in_len=512):
		super(ablation2, self).__init__()
		self.models = torch.nn.ModuleList()#seven independent GNN models
		self.layer_num = layer_num
		self.class_num = class_num
		self.feature_fusion = feature_fusion
		#self.f1_transform = 64
		self.layer_num = layer_num
		self.in_len = in_len
		self.input_dim = input_dim
		#self.long_conv = hyena.HyenaOperator(d_model=input_dim,l_max=in_len)#
		#self.fc1 = nn.Linear(math.floor( in_len / pool_size),self.f1_transform )
		self.hyper_dim = int(self.input_dim)
		self.mainfold_name = args.mainfold
		self.mainfold = getattr(mainfold,self.mainfold_name)()
		self.feature_fusion = feature_fusion
		self.layer_num = layer_num

		dims = [self.input_dim] + ([self.hyper_dim] * (layer_num))

		if self.mainfold.name == 'Hyperboloid':
			dims[0] += 1		
		n_curvatures = len(dims)
		self.radius = radius
		if radius is None:
			self.curvatures = [nn.Parameter(torch.Tensor([1.])) for _ in range(n_curvatures)]
		else:
			self.curvatures = [torch.tensor([radius]) for _ in range(n_curvatures)]         # fixed curvature
		self.curvatures.append(self.radius)

		act = getattr(torch.nn.functional, act)
		acts = [act] * (layer_num)
		for c in range(class_num):
			graph_layers = []
			for i in range(layer_num-1):
				c_in, c_out = self.curvatures[0], self.curvatures[1]
				in_dim, out_dim = dims[i], dims[i+1]
				graph_layers.append(HyperbolicGCN(self.mainfold,in_dim,out_dim,c_in, c_out,dropout,acts[i],bias,use_att,local_agg))
			in_dim, out_dim = dims[-2], dims[-1]
			graph_layers.append(torch_geometric.nn.models.GIN(in_dim,out_dim,1,out_dim,act='tanh',norm=nn.BatchNorm1d(out_dim)))
			self.models.append(nn.Sequential(*graph_layers))
		
		hidden3 = dims[0]+class_num*sum(dims[1:])	

		#self.fc2 = get_classifier(hidden3,class_num,feature_fusion)
		fc2_dim = hidden3*2
		self.fc2 = nn.Sequential(
		  nn.Linear(fc2_dim,int(fc2_dim/2)),
		  nn.ReLU(),
		  nn.Linear(int(fc2_dim/2),int(fc2_dim/4)),
		  nn.ReLU(),
		  nn.Linear(int(fc2_dim/4),class_num),
		)
		return

	def forward(self,data,edge_id=None):
		f1 = data.embed1 #f1 = data.encode1
		sparse_adj = data.sparse_adj1
		edges = data.edge1
		edge_index = data.edge2
		#f1 = self.fc1(f1)

		if self.mainfold_name == 'Hyperboloid':
			o = torch.zeros_like(f1)
			f1 = torch.cat([o[:, 0:1], f1], dim=1)
		output = [f1]

		for i,m in enumerate(self.models):
			tmp = f1
			for j in range(self.layer_num-1):
				input = (tmp,sparse_adj[i])	
				tmp, _ = m[j](input)
				output.append(tmp)
			tmp =  m[-1](tmp,edges[i])
			output.append(tmp)

		x = torch.cat(output,dim=1)
		node_id = edge_index[:, edge_id]
		x1 = x[node_id[0]]
		x2 = x[node_id[1]]

		x = torch.cat([x1, x2], dim=1) #torch.mul(x1, x2)
		x = self.fc2(x)
		return x

	def forward(self,data,edge_id=None):
		f1 = data.embed1 #f1 = data.encode1
		sparse_adj = data.sparse_adj1
		edges = data.edge1
		edge_index = data.edge2
		#f1 = self.fc1(f1)

		if self.mainfold_name == 'Hyperboloid':
			o = torch.zeros_like(f1)
			f1 = torch.cat([o[:, 0:1], f1], dim=1)
		output = [f1]

		for i,m in enumerate(self.models):
			tmp = f1
			for j in range(self.layer_num):
				tmp =  m[-1](tmp,edges[i])
				output.append(tmp)


		x = torch.cat(output,dim=1)
		node_id = edge_index[:, edge_id]
		x1 = x[node_id[0]]
		x2 = x[node_id[1]]

		x = torch.cat([x1, x2], dim=1) #torch.mul(x1, x2)
		x = self.fc2(x)
		return x

class GatedInteractionNetwork(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim):
		super(GatedInteractionNetwork, self).__init__()
		self.fc_interaction = nn.Linear(input_dim, hidden_dim)
		self.fc_gate = nn.Linear(input_dim, hidden_dim)
		self.fc_output = nn.Linear(hidden_dim, output_dim)
		
	def forward(self, x1, x2):
		interaction = x1 * x2 
		# Gating mechanism
		gate = torch.sigmoid(self.fc_gate(x1 + x2)) 
		gated_interaction = gate * F.relu(self.fc_interaction(interaction))
		output = self.fc_output(gated_interaction)
		
		return output


class ReviewerSafeRiemannianEncoder(nn.Module):
	"""
	Extract relation-wise local tangent interaction features and fuse them.
	"""

	def __init__(self, branch_dim, class_num=7):
		super(ReviewerSafeRiemannianEncoder, self).__init__()
		self.class_num = class_num
		self.branch_dim = branch_dim
		self.total_riemann_dim = branch_dim * class_num
		input_dim = 2 * branch_dim
		self.feature_clip = 50.0

		self.pre_norm = nn.ModuleList([nn.LayerNorm(input_dim) for _ in range(class_num)])
		self.W_gate = nn.ModuleList([nn.Linear(input_dim, branch_dim) for _ in range(class_num)])
		self.W_feat = nn.ModuleList([nn.Linear(input_dim, branch_dim) for _ in range(class_num)])

	def _origin_like(self, x, c, manifold):
		if getattr(manifold, 'name', None) == 'Hyperboloid':
			sqrtK = torch.sqrt(torch.clamp(1.0 / c, min=1e-15)).to(device=x.device, dtype=x.dtype)
			origin = torch.zeros_like(x)
			origin[..., 0] = sqrtK
			return manifold.proj(origin, c)
		return torch.zeros_like(x)

	def sanitize(self, x):
		x = torch.nan_to_num(x, nan=0.0, posinf=self.feature_clip, neginf=-self.feature_clip)
		return torch.clamp(x, min=-self.feature_clip, max=self.feature_clip)

	def forward(self, x_hyp_list, edge_index, edge_id, manifold, curvatures, return_stacked=False):
		node_id = edge_index[:, edge_id]
		src_id, dst_id = node_id[0], node_id[1]
		riemannian_outputs = []

		for r, x_branch in enumerate(x_hyp_list):
			branch_curvature = curvatures[r + 1]
			x_i = x_branch[src_id]
			x_j = x_branch[dst_id]

			u_ij = manifold.logmap(x_i, x_j, branch_curvature)
			u_ji = manifold.logmap(x_j, x_i, branch_curvature)

			origin = self._origin_like(x_i, branch_curvature, manifold)
			v_ij = self.sanitize(manifold.ptransp(x_i, origin, u_ij, branch_curvature))
			v_ji = self.sanitize(manifold.ptransp(x_j, origin, u_ji, branch_curvature))

			v_sym = v_ij + v_ji
			v_antisym = torch.abs(v_ij - v_ji)
			f_ij = self.pre_norm[r](self.sanitize(torch.cat([v_sym, v_antisym], dim=-1)))

			gate = torch.sigmoid(self.W_gate[r](f_ij))
			feat = torch.tanh(self.W_feat[r](f_ij))
			riemannian_outputs.append(gate * feat)

		stacked = torch.stack(riemannian_outputs, dim=1)
		if return_stacked:
			return self.sanitize(stacked)
		return self.sanitize(torch.cat(riemannian_outputs, dim=-1))


class ReviewerSafeRiemannianHead(nn.Module):
	"""
	Local tangent-space pair head with transport to a shared origin anchor.
	"""

	def __init__(self, branch_dim, class_num=7, v1_dim=0):
		super(ReviewerSafeRiemannianHead, self).__init__()
		self.v1_dim = v1_dim
		self.logit_clip = 30.0
		self.encoder = ReviewerSafeRiemannianEncoder(branch_dim, class_num=class_num)

		total_classifier_dim = self.encoder.total_riemann_dim + v1_dim
		hidden1 = max(class_num, int(total_classifier_dim / 2))
		hidden2 = max(class_num, int(total_classifier_dim / 4))
		self.fc2 = nn.Sequential(
			nn.Linear(total_classifier_dim, hidden1),
			nn.ReLU(),
			nn.Linear(hidden1, hidden2),
			nn.ReLU(),
			nn.Linear(hidden2, class_num),
		)

	def _sanitize_logits(self, x):
		x = torch.nan_to_num(x, nan=0.0, posinf=self.logit_clip, neginf=-self.logit_clip)
		return torch.clamp(x, min=-self.logit_clip, max=self.logit_clip)

	def forward(self, x_v1_pair, x_hyp_list, edge_index, edge_id, manifold, curvatures):
		h_riemann = self.encoder(x_hyp_list, edge_index, edge_id, manifold, curvatures)
		if self.v1_dim > 0:
			h_final = torch.cat([h_riemann, self.encoder.sanitize(x_v1_pair)], dim=-1)
		else:
			h_final = h_riemann
		return self._sanitize_logits(self.fc2(h_final))


class ReviewerSafeRiemannianAdapter(nn.Module):
	"""
	Adapt local tangent features into a residual feature correction before the
	baseline classifier.
	"""

	def __init__(self, branch_dim, base_dim, class_num=7):
		super(ReviewerSafeRiemannianAdapter, self).__init__()
		self.base_dim = base_dim
		self.encoder = ReviewerSafeRiemannianEncoder(branch_dim, class_num=class_num)
		self.base_norm = nn.LayerNorm(base_dim)
		self.riem_proj = nn.Linear(self.encoder.total_riemann_dim, base_dim)
		self.riem_norm = nn.LayerNorm(self.encoder.total_riemann_dim)
		self.fusion_gate = nn.Linear(base_dim * 2, base_dim)
		
		# Zero-initialized projection
		nn.init.zeros_(self.riem_proj.weight)
		nn.init.zeros_(self.riem_proj.bias)
		nn.init.constant_(self.fusion_gate.bias, -2.0)

	def forward(self, x_v1_pair, x_hyp_list, edge_index, edge_id, manifold, curvatures):
		base = self.base_norm(self.encoder.sanitize(x_v1_pair))
		h_riemann = self.encoder(x_hyp_list, edge_index, edge_id, manifold, curvatures)
		# Standardize input before zero-initialized projection
		riem_residual = self.riem_proj(self.riem_norm(h_riemann))
		gate = torch.sigmoid(self.fusion_gate(torch.cat([base, riem_residual], dim=-1)))
		fused = base + gate * riem_residual
		return self.encoder.sanitize(fused)


class ReviewerSafeRiemannianBranchAdapter(nn.Module):
	"""
	Keep relation-wise geometric branches separate longer, then inject them as a
	gated residual over the baseline pair feature.
	"""

	def __init__(self, branch_dim, base_dim, class_num=7):
		super(ReviewerSafeRiemannianBranchAdapter, self).__init__()
		self.class_num = class_num
		self.base_dim = base_dim
		self.encoder = ReviewerSafeRiemannianEncoder(branch_dim, class_num=class_num)
		self.base_norm = nn.LayerNorm(base_dim)
		self.branch_proj = nn.ModuleList([nn.Linear(branch_dim, base_dim) for _ in range(class_num)])
		self.branch_norm = nn.ModuleList([nn.LayerNorm(branch_dim) for _ in range(class_num)])
		self.branch_gate = nn.ModuleList([nn.Linear(base_dim * 2, base_dim) for _ in range(class_num)])
		self.branch_scale = nn.Parameter(torch.ones(class_num))
		
		# Zero-initialized projection
		for proj in self.branch_proj:
			nn.init.zeros_(proj.weight)
			nn.init.zeros_(proj.bias)
		for gate in self.branch_gate:
			nn.init.constant_(gate.bias, -2.0)

	def forward(self, x_v1_pair, x_hyp_list, edge_index, edge_id, manifold, curvatures):
		base = self.base_norm(self.encoder.sanitize(x_v1_pair))
		h_branches = self.encoder(
			x_hyp_list,
			edge_index,
			edge_id,
			manifold,
			curvatures,
			return_stacked=True,
		)
		residual = torch.zeros_like(base)
		scale = math.sqrt(float(self.class_num))

		for r in range(self.class_num):
			branch_feat = h_branches[:, r, :]
			# Standardize input before zero-initialized projection
			proj = self.branch_proj[r](self.branch_norm[r](branch_feat))
			gate = torch.sigmoid(self.branch_gate[r](torch.cat([base, proj], dim=-1)))
			residual = residual + self.branch_scale[r] * gate * proj

		fused = base + residual / scale
		return self.encoder.sanitize(fused)

class FactorizedBilinearPooling(nn.Module):
	def __init__(self, input_dim1, input_dim2, output_dim, factor_dim=256):
		super(FactorizedBilinearPooling, self).__init__()
		self.W1 = nn.Linear(input_dim1, factor_dim, bias=False)
		self.W2 = nn.Linear(input_dim2, factor_dim, bias=False)
		self.fc = nn.Linear(factor_dim, output_dim)
		
	def forward(self, v1, v2):
		v1_transformed = self.W1(v1)  
		v2_transformed = self.W2(v2)  
		factorized_interaction = v1_transformed * v2_transformed 
		output = self.fc(factorized_interaction)  
		return output

class GatedBilinearPooling(nn.Module):
	def __init__(self, input_dim1, input_dim2, output_dim):
		super(GatedBilinearPooling, self).__init__()
		# Bilinear weight matrix
		self.bilinear_layer = nn.Bilinear(input_dim1, input_dim2, output_dim)
		self.gate_layer1 = nn.Linear(input_dim1, output_dim)
		self.gate_layer2 = nn.Linear(input_dim2, output_dim)
		
	def forward(self, v1, v2):

		bilinear_output = self.bilinear_layer(v1, v2)
		gate_v1 = self.gate_layer1(v1)  # Linear transformation of v1
		gate_v2 = self.gate_layer2(v2)  # Linear transformation of v2
		gate = torch.sigmoid(gate_v1 + gate_v2)
		gated_bilinear_output = bilinear_output * gate

		return gated_bilinear_output





class CodeBook(nn.Module):
	def __init__(self, param, data_loader):
		super(CodeBook, self).__init__()
		self.param = param
		self.Protein_Encoder = GCN_Encoder(param, data_loader)
		self.Protein_Decoder = GCN_Decoder(param)
		self.vq_layer = VectorQuantizer(param['prot_hidden_dim'], param['num_embeddings'], param['commitment_cost'])

	def forward(self, batch_graph):
		z = self.Protein_Encoder.encoding(batch_graph)
		e, e_q_loss, encoding_indices = self.vq_layer(z)

		x_recon = self.Protein_Decoder.decoding(batch_graph, e)
		recon_loss = F.mse_loss(x_recon, batch_graph.ndata['x'])

		mask = torch.bernoulli(torch.full(size=(self.param['num_embeddings'],), fill_value=self.param['mask_ratio'])).bool().to(device)
		mask_index = mask[encoding_indices]
		e[mask_index] = 0.0

		x_mask_recon = self.Protein_Decoder.decoding(batch_graph, e)


		x = F.normalize(x_mask_recon[mask_index], p=2, dim=-1, eps=1e-12)
		y = F.normalize(batch_graph.ndata['x'][mask_index], p=2, dim=-1, eps=1e-12)
		mask_loss = ((1 - (x * y).sum(dim=-1)).pow_(self.param['sce_scale']))
		
		return z, e, e_q_loss, recon_loss, mask_loss.sum() / (mask_loss.shape[0] + 1e-12)





def get_classifier(hidden_layer,class_num,feature_fusion):
	fc = None
	if feature_fusion == 'CnM':
		fc = nn.Linear(3*hidden_layer,class_num)
	elif feature_fusion == 'concat':
		fc = nn.Linear(2*hidden_layer,class_num)
	elif feature_fusion == 'mul':
		fc = nn.Linear(1*hidden_layer,class_num)
	return fc


class HyperbolicGCN(nn.Module):
	"""
	Hyperbolic graph convolution layer.
	"""

	def __init__(self, manifold, in_features, out_features, c_in, c_out, dropout, act, use_bias, use_att, local_agg):
		super(HyperbolicGCN, self).__init__()
		self.mainfold = mainfold
		self.linear = HypLinear(manifold, in_features, out_features, c_in, dropout, use_bias)
		self.agg = HypAgg(manifold, c_in, out_features, dropout, use_att, local_agg)
		self.hyp_act = HypAct(manifold, c_in, c_out, act)

	def forward(self, input):
		x, adj = input

		h = self.linear.forward(x)
		h = self.agg.forward(h, adj)
		h = self.hyp_act.forward(h)
		output = h, adj
		return output



class HypLinear(nn.Module):
	"""
	Hyperbolic linear layer.
	"""

	def __init__(self, manifold, in_features, out_features, c, dropout, use_bias):
		super(HypLinear, self).__init__()
		self.manifold = manifold
		self.in_features = in_features
		self.out_features = out_features
		self.c = c
		self.dropout = dropout
		self.use_bias = use_bias
		self.bias = nn.Parameter(torch.Tensor(out_features))
		self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
		self.reset_parameters()

	def reset_parameters(self):
		torch.nn.init.xavier_uniform_(self.weight, gain=math.sqrt(2))
		torch.nn.init.constant_(self.bias, 0)

	def forward(self, x):
		drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
		mv = self.manifold.mobius_matvec(drop_weight, x, self.c)
		res = self.manifold.proj(mv, self.c)
		if self.use_bias:
			bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
			hyp_bias = self.manifold.expmap0(bias, self.c)
			hyp_bias = self.manifold.proj(hyp_bias, self.c)
			res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
			res = self.manifold.proj(res, self.c)
		return res

	def extra_repr(self):
		return 'in_features={}, out_features={}, c={}'.format(
			self.in_features, self.out_features, self.c
		)


class HypAgg(nn.Module):
	"""
	Hyperbolic aggregation layer.
	"""

	def __init__(self, manifold, c, in_features, dropout, use_att, local_agg):
		super(HypAgg, self).__init__()
		self.manifold = manifold
		self.c = c

		self.in_features = in_features
		self.dropout = dropout
		self.local_agg = local_agg
		self.use_att = use_att
		if self.use_att:
			self.att = DenseAtt(in_features, dropout)

	def forward(self, x, adj):
		x_tangent = self.manifold.logmap0(x, c=self.c)
		if self.use_att:
			if self.local_agg:
				x_local_tangent = []
				for i in range(x.size(0)):
					x_local_tangent.append(self.manifold.logmap(x[i], x, c=self.c))
				x_local_tangent = torch.stack(x_local_tangent, dim=0)
				adj_att = self.att(x_tangent, adj)
				att_rep = adj_att.unsqueeze(-1) * x_local_tangent
				support_t = torch.sum(adj_att.unsqueeze(-1) * x_local_tangent, dim=1)
				output = self.manifold.proj(self.manifold.expmap(x, support_t, c=self.c), c=self.c)
				return output
			else:
				adj_att = self.att(x_tangent, adj)
				support_t = torch.matmul(adj_att, x_tangent)
		else:
			support_t = torch.spmm(adj, x_tangent)
		output = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)
		return output

	def extra_repr(self):
		return 'c={}'.format(self.c)


class HypAct(nn.Module):
	"""
	Hyperbolic activation layer.
	"""

	def __init__(self, manifold, c_in, c_out, act):
		super(HypAct, self).__init__()
		self.manifold = manifold
		self.c_in = c_in
		self.c_out = c_out
		self.act = act

	def forward(self, x):
		xt = self.act(self.manifold.logmap0(x, c=self.c_in))

		xt = self.manifold.proj_tan0(xt, c=self.c_out)
		return self.manifold.proj(self.manifold.expmap0(xt, c=self.c_out), c=self.c_out)

	def extra_repr(self):
		return 'c_in={}, c_out={}'.format(
			self.c_in, self.c_out
		)


def get_mainfold(mainfold_name):
	if mainfold_name == 'Euclidean':
		mainfold = mainfold.Euclidean()
	elif mainfold_name == 'Hyperboloid':
		mainfold = mainfold.Hyperboloid(e)
	elif mainfold_name == 'PoincareBall':
		mainfold = mainfold.PoincareBall()
	else:
		print(f'error, unrecognzied mainfold_name {mainfold_name}')

	return mainfold

class GCN_Encoder(nn.Module):
	def __init__(self, param, data_loader):
		super(GCN_Encoder, self).__init__()        
		self.data_loader = data_loader
		self.num_layers = param['prot_num_layers']
		self.dropout = nn.Dropout(param['dropout_ratio'])
		self.layers = nn.ModuleList()
		self.norms = nn.ModuleList()
		self.fc = nn.ModuleList()

		self.norms.append(nn.BatchNorm1d(param['prot_hidden_dim']))
		self.fc.append(nn.Linear(param['prot_hidden_dim'], param['prot_hidden_dim']))
		self.layers.append(HeteroGraphConv({'SEQ' : GraphConv(param['input_dim'], param['prot_hidden_dim']), 
											'STR_KNN' : GraphConv(param['input_dim'], param['prot_hidden_dim']), 
											'STR_DIS' : GraphConv(param['input_dim'], param['prot_hidden_dim'])}, aggregate='sum'))

		for i in range(self.num_layers - 1):
			self.norms.append(nn.BatchNorm1d(param['prot_hidden_dim']))
			self.fc.append(nn.Linear(param['prot_hidden_dim'], param['prot_hidden_dim']))
			self.layers.append(HeteroGraphConv({'SEQ' : GraphConv(param['prot_hidden_dim'], param['prot_hidden_dim']), 
												'STR_KNN' : GraphConv(param['prot_hidden_dim'], param['prot_hidden_dim']), 
												'STR_DIS' : GraphConv(param['prot_hidden_dim'], param['prot_hidden_dim'])}, aggregate='sum'))

	def forward(self, vq_layer):
		prot_embed_list = []
		for iter, batch_graph in enumerate(self.data_loader):
			device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
			batch_graph.to(device)
			h = self.encoding(batch_graph)
			z, _, _ = vq_layer(h)
			batch_graph.ndata['h'] = torch.cat([h, z], dim=-1)
			prot_embed = dgl.mean_nodes(batch_graph, 'h').detach().cpu()
			prot_embed_list.append(prot_embed)

		return torch.cat(prot_embed_list, dim=0)

	def encoding(self, batch_graph):
		x = batch_graph.ndata['x']
		for l, layer in enumerate(self.layers):
			x = layer(batch_graph, {'amino_acid': x})
			x = self.norms[l](F.relu(self.fc[l](x['amino_acid'])))
			if l != self.num_layers - 1:
				x = self.dropout(x)

		return x
		


class GCN_Decoder(nn.Module):
	def __init__(self, param):
		super(GCN_Decoder, self).__init__()
		
		self.num_layers = param['prot_num_layers']
		self.dropout = nn.Dropout(param['dropout_ratio'])
		self.layers = nn.ModuleList()
		self.norms = nn.ModuleList()
		self.fc = nn.ModuleList()

		for i in range(self.num_layers - 1):
			self.norms.append(nn.BatchNorm1d(param['prot_hidden_dim']))
			self.fc.append(nn.Linear(param['prot_hidden_dim'], param['prot_hidden_dim']))
			self.layers.append(HeteroGraphConv({'SEQ' : GraphConv(param['prot_hidden_dim'], param['prot_hidden_dim']), 
												'STR_KNN' : GraphConv(param['prot_hidden_dim'], param['prot_hidden_dim']), 
												'STR_DIS' : GraphConv(param['prot_hidden_dim'], param['prot_hidden_dim'])}, aggregate='sum'))

		self.fc.append(nn.Linear(param['prot_hidden_dim'], param['input_dim']))
		self.layers.append(HeteroGraphConv({'SEQ' : GraphConv(param['prot_hidden_dim'], param['prot_hidden_dim']), 
											'STR_KNN' : GraphConv(param['prot_hidden_dim'], param['prot_hidden_dim']), 
											'STR_DIS' : GraphConv(param['prot_hidden_dim'], param['prot_hidden_dim'])}, aggregate='sum'))


	def decoding(self, batch_graph, x):

		for l, layer in enumerate(self.layers):
			x = layer(batch_graph, {'amino_acid': x})
			x = self.fc[l](x['amino_acid'])

			if l != self.num_layers - 1:
				x = self.dropout(self.norms[l](F.relu(x)))
			else:
				pass

		return x

class VectorQuantizer(nn.Module):
	"""
	VQ-VAE layer: Input any tensor to be quantized. 
	Args:
		embedding_dim (int): the dimensionality of the tensors in the
		quantized space. Inputs to the modules must be in this format as well.
		num_embeddings (int): the number of vectors in the quantized space.
		commitment_cost (float): scalar which controls the weighting of the loss terms.
	"""
	def __init__(self, embedding_dim, num_embeddings, commitment_cost):
		super().__init__()
		self.embedding_dim = embedding_dim
		self.num_embeddings = num_embeddings
		self.commitment_cost = commitment_cost
		
		# initialize embeddings
		self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
		
	def forward(self, x):    
		x = F.normalize(x, p=2, dim=-1)
		encoding_indices = self.get_code_indices(x)
		quantized = self.quantize(encoding_indices)

		q_latent_loss = F.mse_loss(quantized, x.detach())
		e_latent_loss = F.mse_loss(x, quantized.detach())
		loss = q_latent_loss + self.commitment_cost * e_latent_loss

		# Straight Through Estimator
		quantized = x + (quantized - x).detach().contiguous()

		return quantized, loss, encoding_indices
	
	def get_code_indices(self, x):

		distances = (
			torch.sum(x ** 2, dim=-1, keepdim=True) +
			torch.sum(F.normalize(self.embeddings.weight, p=2, dim=-1) ** 2, dim=1) -
			2. * torch.matmul(x, F.normalize(self.embeddings.weight.t(), p=2, dim=0))
		)
		
		encoding_indices = torch.argmin(distances, dim=1)
		
		return encoding_indices
	
	def quantize(self, encoding_indices):
		"""Returns embedding tensor for a batch of indices."""
		return F.normalize(self.embeddings(encoding_indices), p=2, dim=-1)
