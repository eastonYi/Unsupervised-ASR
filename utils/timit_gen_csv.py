"""
"""
import logging
from argparse import ArgumentParser
from collections import Counter, defaultdict
from pathlib import Path
import os


class MyDict(dict):
    __missing__ = lambda self, key: key

dict_mapping = MyDict()
dict_mapping.update({'ao': 'aa',
                     'ax': 'ah', 'ax-h': 'ah',
                     'axr': 'er',
                     'hv': 'hh',
                     'ix': 'ih',
                     'el': 'l',
                     'em': 'm',
                     'en': 'n', 'nx': 'n',
                     'eng': 'ng',
                     'zh': 'sh',
                     'ux': 'uw',
                     'pcl': 'sil', 'tcl': 'sil', 'bcl': 'sil', 'kcl': 'sil', 'dcl': 'sil', 'gcl': 'sil', 'h#': 'sil', 'pau': 'sil', 'epi': 'sil',
                     'q': ''})

def pre_processing(fpath, f_new, unit, mapping):
    p = Path(fpath)

    with open(f_new, 'w') as fw:
        for wav in p.glob('*/*/*.WAV'):
            f_label = str(wav).split('.')[0] + '.' + unit
            f_wav = str(wav).split('.')[0] + '.wav'
            os.system('~/easton/projects/kaldi/tools/sph2pipe_v2.5/sph2pipe -f wav {} {}'.format(wav, f_wav))
            list_stamps = []
            list_labels = []
            with open(f_label) as f:
                for line in f:
                    _, stamp, label = line.strip().split()
                    list_stamps.append(stamp)
                    list_labels.append(label if not mapping else dict_mapping[label])
                    line_new = f_wav + ',' + ' '.join(list_labels) + ',' + ' '.join(list_stamps)
            fw.write(line_new + '\n')
            list_stamps = []
            list_labels = []


def get_phone_vocab(list_files, f_new):
    # print(list_files)
    list_files = list_files.split(',')
    word2cnt = Counter()
    for file in list_files:
        print(file)
        with open(file) as f:
            for l in f:
                words = l.strip().split(',')[1].split()
                word2cnt.update(Counter(words))

    with open(f_new, 'w') as fout:
        for word, cnt in word2cnt.most_common():
            fout.write(u"{}\t{}\n".format(word, cnt))
    logging.info('Vocab path: {}\t size: {}'.format(f_new, len(word2cnt)))


if __name__ == '__main__':
    """
    demo:
        python utils/timit_gen_csv.py --dir ~/easton/data2/TIMIT/TEST/ --unit PHN --output ~/easton/data2/TIMIT/test_phone39.csv -m
        python utils/timit_gen_csv.py --dir ~/easton/data2/TIMIT/test_phone39.csv,/mnt/lustre/xushuang2/easton/data2/TIMIT/train_phone39.csv  --output ~/easton/data2/TIMIT/phone39.list
    """
    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, dest='input_file', default=None, help='''
                        path/to/TIMIT/TRAIN or path/to/TIMIT/TEST ''')
    parser.add_argument('--output', type=str, dest='output_file', default=None)
    parser.add_argument('--unit', type=str, dest='unit', default='PHN')
    parser.add_argument('-m', action='store_true', dest='mapping', default=False)

    args = parser.parse_args()
    # Read config
    logging.basicConfig(level=logging.INFO)

    # pre_processing(args.input_file, args.output_file, args.unit, args.mapping)
    get_phone_vocab(args.input_file, args.output_file)

    logging.info("Done")
