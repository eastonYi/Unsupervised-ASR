import logging
from argparse import ArgumentParser
import re


def pre_processing(vocab1_file, vocab2_file, input, output):
    vocab_b = {}
    vocab_s = {}
    phone_map = lambda x: re.sub('[0-9]', '', x)

    with open(vocab1_file) as f:
        for line in f:
            phone, id = line.strip().split()
            vocab_b[id] = phone

    with open(vocab2_file) as f:
        for idx, line in enumerate(f):
            phone = line.strip().split()[0]
            vocab_s[phone] = str(idx)

    with open(input) as f, open(output, 'w') as fw:
        for line in f:
            uttid, align = line.strip().split(maxsplit=1)
            new_line = uttid + ' ' + ' '.join(vocab_s[phone_map(vocab_b[id])] for id in align.split())
            fw.write(new_line+'\n')


if __name__ == '__main__':
    """
    demo:
        python gen_phone_align.py --vocab1 phones217.vocab --vocab2 phones67.vocab --input train.phone.frame --output train.phone67.frame
    """
    parser = ArgumentParser()
    parser.add_argument('--vocab1', type=str, dest='vocab1_file', default=None)
    parser.add_argument('--vocab2', type=str, dest='vocab2_file', default=None)
    parser.add_argument('--input', type=str, dest='input', default=None)
    parser.add_argument('--output', type=str, dest='output', default=None)

    args = parser.parse_args()
    # Read config
    logging.basicConfig(level=logging.INFO)

    pre_processing(args.vocab1_file, args.vocab2_file, args.input, args.output)

    logging.info("Done")
