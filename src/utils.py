import numpy as np
import random
import torch
import os
import math
import time
import Models
from collections import defaultdict
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader


class StepProfiler:
	def __init__(self,enabled=False,cuda_sync=True):
		self.enabled = enabled
		self.cuda_sync = cuda_sync and torch.cuda.is_available()
		self.totals = defaultdict(float)
		self.counts = defaultdict(int)

	def region(self,name):
		return _ProfileRegion(self,name)

	def reset(self):
		self.totals.clear()
		self.counts.clear()

	def report(self,header=None):
		if not self.enabled or not self.totals:
			return ''
		lines = []
		if header:
			lines.append(header)
		total = sum(self.totals.values())
		width = max(len(k) for k in self.totals.keys())
		for name,secs in sorted(self.totals.items(),key=lambda kv: -kv[1]):
			pct = (secs/total*100.0) if total>0 else 0.0
			n = self.counts[name]
			per = (secs/n*1000.0) if n>0 else 0.0
			lines.append(f'  {name:<{width}}  {secs:7.3f}s  ({pct:5.1f}%)  n={n:<5d}  avg={per:7.2f}ms')
		lines.append(f'  {"sum":<{width}}  {total:7.3f}s')
		return '\n'.join(lines)


class _ProfileRegion:
	def __init__(self,profiler,name):
		self.profiler = profiler
		self.name = name

	def __enter__(self):
		if not self.profiler.enabled:
			return self
		if self.profiler.cuda_sync:
			torch.cuda.synchronize()
		self.t0 = time.perf_counter()
		return self

	def __exit__(self,exc_type,exc,tb):
		if not self.profiler.enabled:
			return False
		if self.profiler.cuda_sync:
			torch.cuda.synchronize()
		self.profiler.totals[self.name] += time.perf_counter() - self.t0
		self.profiler.counts[self.name] += 1
		return False
def sorted_pair(id1,id2):
	if id1<id2:
		return [id1,id2]
	return [id2,id1]

def check_files_exist(fList:list):
	for fName in fList:
		if not os.path.isfile(fName):
			print(f'error,input file {fName} does not exist')
			exit()
	return
#encode binary vecotr into single num
def encode_inter(vec,num=7):
	result = 0
	for i,v in enumerate(vec):
		result += v*pow(2,num-1-i)
	return result
#		print([1,1,0,0,0,1,0])
#		a = utils.encode_inter([1,1,0,0,0,1,0])
def decode_inter(value,num=7):
	result = []
	for i in range(num):
		tmp = pow(2,num-1-i)
		v = math.floor(value/tmp)
		#print(f'{value} {tmp} {v}')
		value -= v*tmp
		result.append(v)
	return np.array(result,dtype=float)
def sort_dir_by_value(dict): #note: decesending order
	keys = list(dict.keys())
	values = list(dict.values())
	sorted_value_index = np.argsort(values)[::-1]
	sorted_dict = {keys[i]: values[i] for i in sorted_value_index}
	return sorted_dict


from sklearn.metrics import roc_auc_score, average_precision_score

class Metrictor_PPI:
	def __init__(self, pre_y, truth_y, prob_y=None, is_binary=False):
		self.prob_y = prob_y
		self.truth_y = truth_y

		pre_bool = pre_y.bool()
		truth_bool = truth_y.bool()
		tp = (pre_bool & truth_bool).sum().item()
		fp = (pre_bool & ~truth_bool).sum().item()
		tn = (~pre_bool & ~truth_bool).sum().item()
		fn = (~pre_bool & truth_bool).sum().item()

		self.TP = tp
		self.FP = fp
		self.TN = tn
		self.FN = fn
		self.num = pre_bool.numel()
	
	def append_result(self,path='test.txt',e=None,train_loss=0.0,valid_loss=0.0):
		self.acc = (self.TP + self.TN) / (self.num + 1e-10)
		self.pre = self.TP / (self.TP + self.FP + 1e-10)
		self.recall = self.TP / (self.TP + self.FN + 1e-10)
		self.microF1 = 2 * self.pre * self.recall / (self.pre + self.recall + 1e-10)

		self.auc = 0.0
		self.aupr = 0.0
		if self.prob_y is not None:
			try:
				self.auc = roc_auc_score(self.truth_y.cpu().numpy(), self.prob_y.cpu().numpy(), average='micro')
				self.aupr = average_precision_score(self.truth_y.cpu().numpy(), self.prob_y.cpu().numpy(), average='micro')
			except Exception:
				pass

		record = (
			f'epoch {e},acc {self.acc:.4f}, microF1 {self.microF1:.4f}, AUC {self.auc:.4f}, AUPR {self.aupr:.4f}, '
			f'precision {self.pre:.2f},recall {self.recall:.2f}, '
			f'train loss {train_loss:.4f}, valid loss {valid_loss:.4f}'
		)
		with open(path,'a') as f:
			f.write(record+'\n')
		return record

	def show_result(self, is_print=False, file=None):
		self.Accuracy = (self.TP + self.TN) / (self.num + 1e-10)
		self.Precision = self.TP / (self.TP + self.FP + 1e-10)
		self.Recall = self.TP / (self.TP + self.FN + 1e-10)
		self.F1 = 2 * self.Precision * self.Recall / (self.Precision + self.Recall + 1e-10)
		
		self.AUC = 0.0
		self.AUPR = 0.0
		if self.prob_y is not None:
			try:
				self.AUC = roc_auc_score(self.truth_y.cpu().numpy(), self.prob_y.cpu().numpy(), average='micro')
				self.AUPR = average_precision_score(self.truth_y.cpu().numpy(), self.prob_y.cpu().numpy(), average='micro')
			except Exception:
				pass

		if is_print:
			print_file("Accuracy: {}".format(self.Accuracy), file)
			print_file("Precision: {}".format(self.Precision), file)
			print_file("Recall: {}".format(self.Recall), file)
			print_file("F1-Score: {}".format(self.F1), file)
			if self.prob_y is not None:
				print_file("AUC: {}".format(self.AUC), file)
				print_file("AUPR: {}".format(self.AUPR), file)


