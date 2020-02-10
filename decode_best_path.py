import os
import json
import pickle
import time

import numpy as np
import toml
import click
import ctcdecode

from audiomate import annotations
import evalmate
from evalmate import evaluator


def get_logits(path):
    with open(path, 'rb') as f:
        return pickle.load(f, encoding='bytes')


def get_vocab(config_path):
    model_config = toml.load(config_path)
    vocab = model_config['labels']['labels']
    vocab.append('_')
    return vocab


def evaluate(labels, preds):
    ref = {k: annotations.LabelList.create_single(v) for k, v in labels.items()}
    hyp = {k: annotations.LabelList.create_single(v) for k, v in preds.items()}

    ref_out = evaluator.Outcome(ref)
    hyp_out = evaluator.Outcome(hyp)

    result = evalmate.ASREvaluator().evaluate(ref_out, hyp_out)
    return result


@click.command()
@click.argument('ref-corpus-path', type=click.Path(exists=True))
@click.argument('logits-path', type=click.Path(exists=True))
@click.argument('config-path', type=click.Path(exists=True))
@click.argument('output-path', type=click.Path())
@click.option('--num-workers', type=int, default=8)
def run(ref_corpus_path, logits_path, config_path, output_path, num_workers):

    logits = get_logits(logits_path)
    print('N Logits: {}'.format(len(logits)))
    print('Shape Logits 0: {}'.format(logits[0].shape))

    vocab = get_vocab(config_path)
    print('N Vocab: {}'.format(len(vocab)))

    print('Load refs')
    refs = []
    with open(ref_corpus_path, 'r') as f:
        for x in json.load(f):
            refs.append((x['utt_idx'], x['transcript']))

    refs_dict = {x[0]: x[1] for x in refs}
    print(len(refs))

    os.makedirs(output_path, exist_ok=True)

    decoder = ctcdecode.BestPathDecoder(vocab)
    result = decoder.decode_batch(logits)

    print(len(result))
    predictions = {}

    for i, pred in enumerate(result):
        predictions[refs[i][0]] = pred

    pred_path = os.path.join(output_path, 'predictions.txt')
    with open(pred_path, 'w') as f:
        outs = ['{} {}'.format(k, v) for k, v in predictions.items()]
        f.write('\n'.join(outs))

    report = evaluate(refs_dict, predictions)
    report_path = os.path.join(output_path, 'result.txt')
    report.write_report(report_path, template='asr_detail')


if __name__ == '__main__':
    run()
