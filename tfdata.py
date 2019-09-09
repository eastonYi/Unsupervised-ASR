#!/usr/bin/env
# coding=utf-8
from utils.tools import TFData
from utils.arguments import args
from utils.dataset import ASR_align_DataSet
from pathlib import Path


def main():
    dataset_train = ASR_align_DataSet(
        trans_file=args.dirs.train.trans,
        uttid2wav=args.dirs.train.wav_scp,
        align_file=None,
        feat_len_file=None,
        args=args,
        _shuffle=False,
        transform=True)
    dataset_train_supervise = ASR_align_DataSet(
        trans_file=args.dirs.train_supervise.trans,
        uttid2wav=args.dirs.train_supervise.wav_scp,
        align_file=None,
        feat_len_file=None,
        args=args,
        _shuffle=False,
        transform=True)
    dataset_dev = ASR_align_DataSet(
        trans_file=args.dirs.dev.trans,
        uttid2wav=args.dirs.dev.wav_scp,
        align_file=None,
        feat_len_file=None,
        args=args,
        _shuffle=False,
        transform=True)
    feature_train = TFData(dataset=dataset_train,
                    dir_save=args.dirs.train.tfdata,
                    args=args)
    feature_train_supervise = TFData(dataset=dataset_train_supervise,
                    dir_save=args.dirs.train_supervise.tfdata,
                    args=args)
    feature_dev = TFData(dataset=dataset_dev,
                    dir_save=args.dirs.dev.tfdata,
                    args=args)
    # feature_train.save('0')
    # feature_dev.save('0')
    # feature_train_supervise.save('0')

    dataset_train = ASR_align_DataSet(
        trans_file=args.dirs.train.trans,
        align_file=args.dirs.train.align,
        uttid2wav=args.dirs.train.wav_scp,
        feat_len_file=args.dirs.train.feat_len,
        args=args,
        _shuffle=False,
        transform=True)
    dataset_dev = ASR_align_DataSet(
        trans_file=args.dirs.dev.trans,
        uttid2wav=args.dirs.dev.wav_scp,
        align_file=args.dirs.dev.align,
        feat_len_file=args.dirs.dev.feat_len,
        args=args,
        _shuffle=False,
        transform=True)
    print(dataset_dev[0]['align'])
    print(dataset_dev[0]['trans'])
    print([dataset_dev.idx2token[i] for i in dataset_dev[0]['trans']])
    import pdb; pdb.set_trace()
    split_save()
    # # for uttid_feature in feature_dev.read():
    # #     uttid, feature = uttid_feature
    # #     uttid = uttid.numpy()
    # #     align = dataset_dev.get_attrs('align', dataset_dev.list_uttids[:10])
    # for sample in dataset_dev:
    #     if sample['align'] is None:
    #         print(sample['uttid'])
    #         print(sample['feature'].shape)

            # print(sample['align'].shape)
        # print(sample['stamps'].shape)
        # print(sample['trans'].shape)
        # import pdb; pdb.set_trace()
    # dataset_train.get_dataset_ngram(n=args.data.ngram, k=10000, savefile=args.dirs.ngram)
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
                        dir_save=args.dirs.train.tfdata,
                        args=args)

        tfdata_train.save(i.name.split('.')[0])


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    main()
