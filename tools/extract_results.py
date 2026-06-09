import os
import math
import argparse
import numpy as np
from tabulate import tabulate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--res-dir', type=str, default='', help='Path to the results')
    parser.add_argument('--shot-list', type=int, nargs='+', default=[10], help='')
    args = parser.parse_args()

    wf = open(os.path.join(args.res_dir, 'results.txt'), 'w')

    file_paths = []
    for fid, fname in enumerate(os.listdir(args.res_dir)):
        if 'episode' not in fname:
            continue
        _dir = os.path.join(args.res_dir, fname)
        if not os.path.isdir(_dir):
            continue
        n_epi = int(fname.split('episode')[-1])
        file_paths.append([n_epi, os.path.join(_dir, 'log.txt')])

    file_paths = sorted(file_paths)
    header, results = ['episode', 'acc'], []
    for fid, fpath in file_paths:
        lineinfos = open(fpath).readlines()
        res_info = lineinfos[-4].strip()  # TODO:-4表示log文件的倒数第四行，不使用PCB校正。
        results.append([fid] + [float(res_info.split('The classification accuracy of this episode is ')[-1])])

    results_np = np.array(results)
    avg = np.mean(results_np, axis=0).tolist()
    cid = [1.96 * s / math.sqrt(results_np.shape[0]) for s in np.std(results_np, axis=0)]
    results.append(['μ'] + avg[1:])
    results.append(['c'] + cid[1:])

    table = tabulate(
        results,
        tablefmt="pipe",
        floatfmt=".2f",
        headers=header,
        numalign="left",
    )

    # wf.write('--> {}-shot\n'.format(shot))
    wf.write('{}\n\n'.format(table))
    wf.flush()
    wf.close()

    print('Reformat all results -> {}'.format(os.path.join(args.res_dir, 'results.txt')))


if __name__ == '__main__':
    main()
