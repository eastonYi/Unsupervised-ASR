"""
"""
import logging
from argparse import ArgumentParser
from pathlib import Path
from utils.dataProcess import load_wavfile
import re

def get_phones_vocab(file):
    phones = []
    idx2phone = []
    with open(file) as f:
        for line in f:
            if '#' not in line:
                _phone, *_ = line.strip().split('_')
                phone = re.sub('[ 0-9]*', '', _phone)
                idx2phone.append(phone)
    phones = list(set(idx2phone))
    phones.sort()
    phones[0] = '<blk>'

    new_phone_file = file.parent / 'phones_{}.txt'.format(len(phones))
    with open(new_phone_file, 'w') as fw:
        for i, phone in enumerate(phones):
            line = phone + ' ' + str(i)
            fw.write(line + '\n')
    return phones, idx2phone


def pre_processing(ali_file, phone_file, wav_dir, dir):
    dir = Path(dir)
    phones, idx2phone = get_phones_vocab(dir / phone_file)
    # rate, sig = load_wavfile(wav_file)
    # ratio = len(sig)/len(ids)
    ratio = 80
    csv_file = dir / 'train-clean-100.phone.csv'
    with open(dir / ali_file) as f, open(dir / csv_file, 'w') as fw:
        for num, line in zip(range(999999999), f):
            uttid, *ids = line.strip().split()
            wav_file = dir / wav_dir / (uttid + '.wav')
            # rate, sig = load_wavfile(wav_file)

            _idx = ids[0]
            list_phones = [idx2phone[int(_idx)]]
            align = []
            for i, idx in enumerate(ids):
                if idx != _idx:
                    list_phones.append(idx2phone[int(idx)])
                    align.append(i-1)
                    _idx = idx
            align.append(i)

            line = str(wav_file) + ',' + ' '.join(list_phones) + ',' + ' '.join(str(ali*ratio) for ali in align)
            fw.write(line + '\n')

            if num % 10000 == 0:
                print('processed {} utts'.format(num))

    # for dir in ['dev', 'test', 'train']: #
    #     f_new = p / (dir + '_' + unit + '.csv')
    #     num_lost = 0
    #     print('generating ', f_new)
    #     with open(f_new, 'w') as fw:
    #         print('dir: ', len(list(p.glob('wav/' + dir + '/*/*.wav'))))
    #         for wav in p.glob('wav/' + dir + '/*/*.wav'):
    #             f_wav = str(wav).split('.')[0] + '.wav'
    #             try:
    #                 align = dict_trans[wav.name.split('.')[0]]
    #             except KeyError:
    #                 num_lost += 1
    #                 # print('not found ', wav.name.split('.')[0])
    #                 continue
    #
    #             _id = align[0]
    #             list_labels = []
    #             list_stamps = []
    #             for i, id in enumerate(align+[99999]):
    #                 if id == _id:
    #                     continue
    #                 else:
    #                     list_labels.append(vocab[int(_id)])
    #                     list_stamps.append(str(int(i*ratio)))
    #                 _id = id
    #             line_new = f_wav + ',' + ' '.join(list_labels) + ',' + ' '.join(list_stamps)
    #             fw.write(line_new + '\n')
    #             list_stamps = []
    #             list_labels = []
    #         print(num_lost, ' lost.')
    #         num_lost = 0


if __name__ == '__main__':
    """
    demo:
        python utils/libri_gen_csv.py --dir /data/sxu/easton/data/LibriSpeech
    """
    parser = ArgumentParser()
    parser.add_argument('--align', type=str, dest='align', default='train.phone.frame')
    parser.add_argument('--phone', type=str, dest='phone', default='phones.txt')
    parser.add_argument('--wav', type=str, dest='wav', default='train-clean-100-wav')
    parser.add_argument('--dir', type=str, dest='dir', default=None)

    args = parser.parse_args()
    # Read config
    logging.basicConfig(level=logging.INFO)

    pre_processing(ali_file=args.align, phone_file=args.phone, wav_dir=args.wav, dir=args.dir)

    logging.info("Done")
