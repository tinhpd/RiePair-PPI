#!/usr/bin/env python3

import argparse
import os
import re
import statistics


METRIC_PATTERN = re.compile(
	r'^epoch\s+(?P<epoch>\d+),acc\s+(?P<acc>\d+\.\d+),\s+microF1\s+(?P<f1>\d+\.\d+),\s+'
	r'AUC\s+(?P<auc>\d+\.\d+),\s+AUPR\s+(?P<aupr>\d+\.\d+)'
)


def parse_seed_list(raw_value):
	return [item for item in raw_value.replace(',', ' ').split() if item]


def parse_best_metrics(path):
	best = None
	with open(path, 'r') as handle:
		for line in handle:
			match = METRIC_PATTERN.match(line.strip())
			if not match:
				continue
			row = {
				'epoch': int(match.group('epoch')),
				'acc': float(match.group('acc')),
				'f1': float(match.group('f1')),
				'auc': float(match.group('auc')),
				'aupr': float(match.group('aupr')),
			}
			if best is None or row['f1'] > best['f1']:
				best = row
	if best is None:
		raise ValueError(f'No epoch metrics found in {path}')
	return best


def mean_std(values):
	mean_value = statistics.mean(values)
	std_value = statistics.stdev(values) if len(values) > 1 else 0.0
	return mean_value, std_value


def collect_model_results(out_dir, model_name, seeds):
	results = []
	for seed in seeds:
		path = os.path.join(out_dir, f'{model_name}_seed{seed}.txt')
		if not os.path.isfile(path):
			raise FileNotFoundError(f'Missing result file: {path}')
		best = parse_best_metrics(path)
		best['seed'] = seed
		best['path'] = path
		results.append(best)
	return results


def summarize_results(results):
	summary = {
		'acc': mean_std([row['acc'] for row in results]),
		'f1': mean_std([row['f1'] for row in results]),
		'auc': mean_std([row['auc'] for row in results]),
		'aupr': mean_std([row['aupr'] for row in results]),
	}
	return summary


def print_model_summary(model_label, results):
	summary = summarize_results(results)
	print(f'[{model_label}]')
	for row in results:
		print(
			f"seed={row['seed']} best_epoch={row['epoch']} "
			f"ACC={row['acc']:.4f} F1={row['f1']:.4f} "
			f"AUC={row['auc']:.4f} AUPR={row['aupr']:.4f}"
		)
	summary_line = (
		f"mean+-std: ACC={summary['acc'][0]:.4f}+/-{summary['acc'][1]:.4f} "
		f"F1={summary['f1'][0]:.4f}+/-{summary['f1'][1]:.4f} "
		f"AUC={summary['auc'][0]:.4f}+/-{summary['auc'][1]:.4f} "
		f"AUPR={summary['aupr'][0]:.4f}+/-{summary['aupr'][1]:.4f}"
	)
	print(summary_line)
	print()


def main():
	parser = argparse.ArgumentParser(description='Summarize HI-PPI ablation results across multiple seeds.')
	parser.add_argument('--out-dir', required=True, help='Directory containing per-seed result txt files')
	parser.add_argument('--seeds', required=True, help='Comma or space separated seed list')
	parser.add_argument(
		'--models',
		default='hippi_handcrafted,hippi_esm2',
		help='Comma or space separated model filename prefixes to summarize',
	)
	args = parser.parse_args()

	seeds = parse_seed_list(args.seeds)
	if not seeds:
		raise ValueError('No seeds provided')
	models = parse_seed_list(args.models)
	if not models:
		raise ValueError('No model prefixes provided')

	print(f'out_dir={args.out_dir}')
	print(f'seeds={" ".join(seeds)}')
	print(f'models={" ".join(models)}')
	print()
	for model_name in models:
		results = collect_model_results(args.out_dir, model_name, seeds)
		print_model_summary(model_name, results)


if __name__ == '__main__':
	main()
