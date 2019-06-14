#!/usr/bin/env
# coding=utf-8
from eastonCode.tfTools.tfData import TFData
from utils.arguments import args
from utils.dataset import ASR_align_DataSet

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
    # for sample in tfdata_dev.read():
    # for sample in dataset_train:
    #     print(sample['align'])
    #     print(sample['label'])
    #     import pdb; pdb.set_trace()
    # dataset_train.get_dataset_ngram(n=args.data.ngram, k=10000, savefile=args.dirs.ngram)
    # import pdb; pdb.set_trace()
    # print()

if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    main()
    # ngram()
