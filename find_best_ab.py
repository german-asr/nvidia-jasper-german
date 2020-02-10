import os
import re
import shutil

import click

WER_PATTERN = re.compile(r'Error Rate : \s*(.*)\%')


@click.command()
@click.argument('folder-path', type=click.Path(exists=True))
def run(folder_path):
    best_wer = 10000.0
    best_wer_path = None

    for folder in os.listdir(folder_path):
        ab_path = os.path.join(folder_path, folder)

        if os.path.isdir(ab_path):
            results_file = os.path.join(ab_path, 'result.txt')
            with open(results_file, 'r') as f:
                content = f.read()

            m = WER_PATTERN.search(content)
            wer = None

            if m is not None:
                wer = m.group(1)

                if float(wer) < best_wer:
                    best_wer = float(wer)
                    best_wer_path = ab_path

    res_path = os.path.join(folder_path, 'best_wer.txt')
    with open(res_path, 'w') as f:
        f.write('{} {}'.format(best_wer_path, best_wer))

    best_pred_path = os.path.join(best_wer_path, 'predictions.txt')
    best_res_path = os.path.join(best_wer_path, 'result.txt')
    shutil.copy(best_pred_path, os.path.join(folder_path, 'predictions.txt'))
    shutil.copy(best_res_path, os.path.join(folder_path, 'result.txt'))


if __name__ == '__main__':
    run()
