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

from ctc_decoders import Scorer
from ctc_decoders import ctc_beam_search_decoder_batch, ctc_beam_search_decoder


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


def softmax(x):
  m = np.expand_dims(np.max(x, axis=-1), -1)
  e = np.exp(x - m)
  return e / np.expand_dims(e.sum(axis=-1), -1)


@click.command()
@click.argument('ref-corpus-path', type=click.Path(exists=True))
@click.argument('logits-path', type=click.Path(exists=True))
@click.argument('config-path', type=click.Path(exists=True))
@click.argument('output-path', type=click.Path())
@click.argument('lm-path', type=click.Path(exists=True))
@click.option('--num-workers', type=int, default=8)
@click.option('--beam-width', type=int, default=128)
@click.option('--alpha-start', type=float, default=0.5)
@click.option('--alpha-end', type=float, default=4.5)
@click.option('--alpha-step', type=float, default=0.5)
@click.option('--beta-start', type=float, default=-4.0)
@click.option('--beta-end', type=float, default=11.0)
@click.option('--beta-step', type=float, default=1.0)
def run(ref_corpus_path, logits_path, config_path, output_path, lm_path, 
        num_workers, beam_width,
        alpha_start, alpha_end, alpha_step,
        beta_start, beta_end, beta_step):

    print('Load refs')
    refs = []
    lengths = []
    with open(ref_corpus_path, 'r') as f:
        for x in json.load(f):
            refs.append((x['utt_idx'], x['transcript']))
            lengths.append(x['files'][0]['num_samples'])


    logits_raw = get_logits(logits_path)
    print('N Logits: {}'.format(len(logits_raw)))
    print('Shape Logits 0: {}'.format(logits_raw[0].shape))
    logits = []

    for i, l in enumerate(logits_raw):
        num_samples = lengths[i]
        num_frames = int(num_samples / 320) + 1
        logits.append(l[:num_frames])

    print('Shape Logits 0 (after trim): {}'.format(logits[0].shape))
    logits = [softmax(l) for l in logits]

    vocab = get_vocab(config_path)
    print('N Vocab: {}'.format(len(vocab)))

    refs_dict = {x[0]: x[1] for x in refs}
    print(len(refs))

    for alpha in np.arange(alpha_start, alpha_end, alpha_step):
        for beta in np.arange(beta_start, beta_end, beta_step):
            print('alpha: {}, beta: {}'.format(alpha, beta))
            target_folder = os.path.join(output_path, 'lm_{}_{}'.format(alpha, beta))
            os.makedirs(target_folder, exist_ok=True)

            # decoder = ctcdecode.BestPathDecoder(vocab)
            # scorer = ctcdecode.WordKenLMScorer(lm_path, alpha, beta)
            # decoder = ctcdecode.BeamSearchDecoder(
            #     vocab,
            #     num_workers=num_workers,
            #     beam_width=beam_width,
            #     scorers=[scorer],
            #     cutoff_prob=np.log(0.000001),
            #     cutoff_top_n=40
            # )
            # result = decoder.decode_batch(logits)
                
            start = time.time()
            scorer = Scorer(alpha, beta, model_path=lm_path, vocabulary=vocab[:-1])
            print('Scorer loaded')
            res = ctc_beam_search_decoder_batch(logits, vocab[:-1], 
                                                beam_size=beam_width, 
                                                num_processes=num_workers,
                                                ext_scoring_func=scorer)
            result = [[v for v in zip(*x)][1][0] for x in res]
            print('Took {}'.format(time.time() - start))

            print(len(result))
            predictions = {}

            for i, pred in enumerate(result):
                predictions[refs[i][0]] = pred

            pred_path = os.path.join(target_folder, 'predictions.txt')
            with open(pred_path, 'w') as f:
                outs = ['{} {}'.format(k, v) for k, v in predictions.items()]
                f.write('\n'.join(outs))

            report = evaluate(refs_dict, predictions)
            report_path = os.path.join(target_folder, 'result.txt')
            report.write_report(report_path, template='asr_detail')


if __name__ == '__main__':
    run()
