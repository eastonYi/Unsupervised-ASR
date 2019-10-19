import logging
from argparse import ArgumentParser


def pre_processing(input1, input2, output):
    dict_trans = {}
    with open(input1) as f1, open(input2) as f2, open(output, 'w') as fw:
        for line in f1:
            key, value = line.strip().split(' ', maxsplit=1)
            dict_trans[key] = value
        for line in f2:
            key = line.split(',')[0].split('.')[0].split('/')[-1]
            fw.write(key + ' ' + dict_trans[key] + '\n')


if __name__ == '__main__':
    """
    demo:
        python utils/aishell_gen_csv.py --dir /data/sxu/easton/data/AISHELL --trans ~/easton/data/AISHELL/train.phone.frame --vocab /data/sxu/easton/data/AISHELL/phones.txt --unit phone
    """
    parser = ArgumentParser()
    parser.add_argument('--all', type=str, dest='all', default=None)
    parser.add_argument('--select', type=str, dest='select', default=None)
    parser.add_argument('--output', type=str, dest='output', default=None)

    args = parser.parse_args()
    # Read config
    logging.basicConfig(level=logging.INFO)

    pre_processing(args.all, args.select, args.output)

    logging.info("Done")
