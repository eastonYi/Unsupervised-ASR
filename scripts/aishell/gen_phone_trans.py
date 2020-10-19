import logging
from argparse import ArgumentParser


def align_shrink(align):
    _token = None
    list_tokens = []
    for token in align:
        if _token != token:
            list_tokens.append(token)
            _token = token

    return list_tokens


def pre_processing(vocab_file, input, output):
    vocab = {}
    with open(vocab_file) as f:
        for line in f:
            phone, id = line.strip().split()
            vocab[id] = phone

    with open(input) as f, open(output, 'w') as fw:
        for line in f:
            uttid, align = line.strip().split(maxsplit=1)

            new_line = uttid + ' ' + ' '.join(str(vocab[id]) for id in align_shrink(align.split()))
            fw.write(new_line+'\n')


if __name__ == '__main__':
    """
    demo:
        python utils/aishell_gen_csv.py --dir /data/sxu/easton/data/AISHELL --trans ~/easton/data/AISHELL/train.phone.frame --vocab /data/sxu/easton/data/AISHELL/phones.txt --unit phone
    """
    parser = ArgumentParser()
    parser.add_argument('--vocab', type=str, dest='vocab_file', default=None)
    parser.add_argument('--input', type=str, dest='input', default=None)
    parser.add_argument('--output', type=str, dest='output', default=None)

    args = parser.parse_args()
    # Read config
    logging.basicConfig(level=logging.INFO)

    pre_processing(args.vocab_file, args.input, args.output)

    logging.info("Done")
