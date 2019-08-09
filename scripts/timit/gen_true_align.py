import logging
from argparse import ArgumentParser
from utils.arguments import args
from utils.dataProcess import get_alignRate


def main(csv_file, phone, length_file, output):
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

    file = open(csv_file).readline().strip().split(',')[0]
    rate = get_alignRate(file, args)

    with open(csv_file) as f, open(output, 'w') as fw:
        for line in f:
            file, phone, stamps = line.strip().split(',')
            uttid = file.split('/')[-2] + '_' + file.split('/')[-1].split('.')[0]
            length = dict_length[uttid]
            align = []
            _stamp = 0
            for phone, stamp in zip(phone.split(), stamps.split()):
                repetition = round((int(stamp) - _stamp)/rate)
                align.extend([dict_phone[phone]]*repetition)
                _stamp = int(stamp)
            align.extend(['1']*(length-len(align)))
            new_line = uttid + ' ' + ' '.join(align[:length])
            fw.write(new_line + '\n')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--csv_file', type=str, dest='csv_file', default=None)
    parser.add_argument('--length_file', type=str, dest='length_file', default=None)
    parser.add_argument('--phone', type=str, dest='phone', default=None)
    parser.add_argument('--output', type=str, dest='output_file', default=None)
    parser.add_argument('-c', type=str, dest='config', default=None)

    param = parser.parse_args()
    # Read config
    logging.basicConfig(level=logging.INFO)
    main(param.csv_file, param.phone, param.length_file, param.output_file)
    logging.info("Done")
