#!/usr/bin/env
# coding=utf-8
from utils.tools import TFData
from utils.arguments import args
from utils.dataset import ASR_align_DataSet
from pathlib import Path


def main():
    dataset_train = ASR_align_DataSet(
        file=[args.dirs.train.data],
        args=args,
        _shuffle=False,
        transform=True)
    dataset_dev = ASR_align_DataSet(
        file=[args.dirs.dev.data],
        args=args,
        _shuffle=False,
        transform=True)
    tfdata_train = TFData(dataset=dataset_train,
                    dataAttr=['feature', 'label', 'align'],
                    dir_save=args.dirs.train.tfdata,
                    args=args)
    tfdata_dev = TFData(dataset=dataset_dev,
                    dataAttr=['feature', 'label', 'align'],
                    dir_save=args.dirs.dev.tfdata,
                    args=args)
    tfdata_train.save('0')
    tfdata_dev.save('0')
    # tfdata_train.get_bucket_size(100, True)
    # split_save()
    # for sample in tfdata_dev.read():
    # # for sample in dataset_train:
    #     # print(sample['feature'].shape)
    #     # print(sample['label'])
    #     print(sample[0].shape)
    #     import pdb; pdb.set_trace()
    dataset_train.get_dataset_ngram(n=args.data.ngram, k=10000, savefile=args.dirs.ngram)
    # import pdb; pdb.set_trace()
    # print()


def split_save(capacity=10000):
    fw = open(args.dirs.train.tfdata / '0.csv', 'w')
    with open(args.dirs.train.data) as f:
        for num, line in enumerate(f):
            if num%capacity == 0:
                idx_file = num//capacity
                print('package file ', idx_file)
                try:
                    fw.close()
                    fw = open(args.dirs.train.tfdata / (str(idx_file) +'.csv'), 'w')
                except:
                    pass
            fw.write(line)
    print('processed {} utts.'.format(num+1))
    fw.close()

    for i in Path(args.dirs.train.tfdata).glob('*.csv'):
        print('converting {}.csv to record'.format(i.name))
        dataset_train = ASR_align_DataSet(
            file=[i],
            args=args,
            _shuffle=False,
            transform=True)
        tfdata_train = TFData(dataset=dataset_train,
                        dataAttr=['feature', 'label', 'align'],
                        dir_save=args.dirs.train.tfdata,
                        args=args)

        tfdata_train.save(i.name.split('.')[0])

if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    main()
    # ngram()
