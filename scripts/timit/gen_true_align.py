import logging
from argparse import ArgumentParser
from pathlib import Path

from utils.arguments import args
from utils.dataProcess import get_alignRate


class MyDict(dict):
    __missing__ = lambda self, key: key


def main(dir_data, phone, length_file, output):
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
                         # 'q': ''})
                         'q': 'sil'})

    with open(phone) as f:
        dict_phone = {}
        for line in f:
            p, idx = line.split()
            dict_phone[p] = idx

    with open(length_file) as f:
        dict_length = {}
        for line in f:
            uttid, length = line.split()
            dict_length[uttid] = int(length)

    rate = get_alignRate('/data/sxu/easton/data/TIMIT/test_wavs/DR3/MJBR0/SX191.wav', args)
    print('MJBR0/SX191.wav align rate:', rate)

    with open(output, 'w') as fw:
        p = Path(dir_data)
        for f_stamps in p.glob('*/*/*.PHN'):
            list_align = []
            with open(f_stamps) as f:
                for line in f:
                    stamp_b, stamp_e, phone = line.strip().split()
                    dupli = int((int(stamp_e) - int(stamp_b)) / rate)
                    list_align.extend([dict_phone[dict_mapping[phone]]] * dupli)

                uttid = '_'.join(str(f_stamps).split('.')[0].split('/')[-2:])
                len_feat = dict_length[uttid]
                # print('len_feat:', len_feat, 'len(list_align):', len(list_align))
                if not 0 < len_feat-len(list_align) < 50:
                    print('len_feat:', len_feat, 'len(list_align):', len(list_align))
                list_align.extend([1] * (len_feat-len(list_align)))
                line = uttid + ' ' + ' '.join(map(str, list_align))
                fw.write(line + '\n')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dir_data', type=str, dest='dir_data', default=None)
    parser.add_argument('--length_file', type=str, dest='length_file', default=None)
    parser.add_argument('--phone', type=str, dest='phone', default=None)
    parser.add_argument('--output', type=str, dest='output_file', default=None)
    parser.add_argument('-c', type=str, dest='config', default=None)

    param = parser.parse_args()
    # Read config
    logging.basicConfig(level=logging.INFO)
    main(param.dir_data, param.phone, param.length_file, param.output_file)
    logging.info("Done")
