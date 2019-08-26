import logging
from argparse import ArgumentParser
from statistics import mean

from utils.tools import R_value, align2stamp


def load_align(file):
    dict_stamps = {}
    with open(file) as f:
        for line in f:
            uttid, align = line.split(maxsplit=1)
            stamps = align2stamp(align.strip().split())
            dict_stamps[uttid] = stamps

    return dict_stamps


def pre_processing(ref, res):
    dict_ref = load_align(ref)
    dict_res = load_align(res)
    list_r = []
    for uttid, ref_stamps in dict_ref.items():
        res_stamps = dict_res[uttid]
        r = R_value(res_stamps=res_stamps, ref_stamps=ref_stamps, region=2)
        list_r.append(r)
    _r = mean(list_r)
    print('R-value:', _r)


if __name__ == '__main__':
    """
    demo:
        python utils/aishell_gen_csv.py --dir /data/sxu/easton/data/AISHELL --trans ~/easton/data/AISHELL/train.phone.frame --vocab /data/sxu/easton/data/AISHELL/phones.txt --unit phone
    """
    parser = ArgumentParser()
    parser.add_argument('--ref', type=str, dest='ref', default=None)
    parser.add_argument('--res', type=str, dest='res', default=None)

    args = parser.parse_args()
    # Read config
    logging.basicConfig(level=logging.INFO)

    pre_processing(args.ref, args.res)

    logging.info("Done")
