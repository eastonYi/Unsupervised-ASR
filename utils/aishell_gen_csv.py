"""
"""
import logging
from argparse import ArgumentParser
from pathlib import Path
from utils.dataProcess import load_wavfile

def pre_processing(fpath, ftrans, fvocab, unit):
    # rate, sig = load_wavfile(f_wav)
    # ratio = len(sig)/len(align)
    ratio = 161
    p = Path(fpath)
    dict_trans = {}
    with open(ftrans) as f:
        for line in f:
            uttid, *phones = line.strip().split()
            dict_trans[uttid] = phones

    with open(fvocab) as f:
        vocab = []
        for line in f:
            phone = line.strip().split()[0]
            vocab.append(phone)

    for dir in ['dev', 'test', 'train']: #
        f_new = p / (dir + '_' + unit + '.csv')
        num_lost = 0
        print('generating ', f_new)
        with open(f_new, 'w') as fw:
            print('dir: ', len(list(p.glob('wav/' + dir + '/*/*.wav'))))
            for wav in p.glob('wav/' + dir + '/*/*.wav'):
                f_wav = str(wav).split('.')[0] + '.wav'
                try:
                    align = dict_trans[wav.name.split('.')[0]]
                except KeyError:
                    num_lost += 1
                    # print('not found ', wav.name.split('.')[0])
                    continue

                _id = align[0]
                list_labels = []
                list_stamps = []
                for i, id in enumerate(align+[99999]):
                    if id == _id:
                        continue
                    else:
                        list_labels.append(vocab[int(_id)])
                        list_stamps.append(str(int(i*ratio)))
                    _id = id
                line_new = f_wav + ',' + ' '.join(list_labels) + ',' + ' '.join(list_stamps)
                fw.write(line_new + '\n')
                list_stamps = []
                list_labels = []
            print(num_lost, ' lost.')
            num_lost = 0


if __name__ == '__main__':
    """
    demo:
        python utils/aishell_gen_csv.py --dir /home/user/easton/data/AISHELL --trans ~/easton/data/AISHELL/train.phone.frame --vocab /home/user/easton/data/AISHELL/phones.txt --unit phone
    """
    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, dest='dir', default=None, help='''
                        path/to/TIMIT/TRAIN or path/to/TIMIT/TEST ''')
    parser.add_argument('--trans', type=str, dest='trans', default=None)
    parser.add_argument('--vocab', type=str, dest='vocab', default=None)
    parser.add_argument('--unit', type=str, dest='unit', default='PHN')

    args = parser.parse_args()
    # Read config
    logging.basicConfig(level=logging.INFO)

    pre_processing(args.dir, args.trans, args.vocab, args.unit)

    logging.info("Done")
