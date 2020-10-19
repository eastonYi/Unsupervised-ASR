import logging
from argparse import ArgumentParser


def pre_processing(input, output):
    with open(input) as f, open(output, 'w') as fw:
        for line in f:
            uttid, align = line.strip().split(maxsplit=1)
            m = [1,1,1,2,3,4,4,5,6,6,6]
            m_new = []
            token_cur = None
            for m1, m2, m3 in zip(m[1:] + [m[-1]], m, [m[0]] + m[:-1]):
                if m2 == m1:
                    m_new.append(m2)
                    token_cur = m2
                elif m2==m3:
                    m_new.append(m2)
                    token_cur = m2
                elif m1==m3:
                    m_new.append(m1)
                    token_cur = m1
                else:
                    m_new.append(token_cur)
            print(m)
            print(m_new)



if __name__ == '__main__':
    """
    demo:
        python utils/aishell_gen_csv.py --dir /data/sxu/easton/data/AISHELL --trans ~/easton/data/AISHELL/train.phone.frame --vocab /data/sxu/easton/data/AISHELL/phones.txt --unit phone
    """
    parser = ArgumentParser()
    parser.add_argument('--input', type=str, dest='input', default=None)
    parser.add_argument('--output', type=str, dest='output', default=None)

    args = parser.parse_args()
    # Read config
    logging.basicConfig(level=logging.INFO)

    pre_processing(args.input, args.output)

    logging.info("Done")
