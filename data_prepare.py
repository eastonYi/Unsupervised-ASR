#!/usr/bin/env
# coding=utf-8
from utils.tools import TFData
from utils.arguments import args
from utils.dataset import ASR_align_ArkDataSet, get_bucket, ASR_align_DataSet
from pathlib import Path


def main():
    # dataset_train = ASR_align_ArkDataSet(
    #     scp_file=args.dirs.train.scp,
    #     trans_file=args.dirs.train.trans,
    #     align_file=None,
    #     feat_len_file=None,
    #     args=args,
    #     _shuffle=True,
    #     transform=False)
    # dataset_dev = ASR_align_ArkDataSet(
    #     scp_file=args.dirs.dev.scp,
    #     trans_file=args.dirs.dev.trans,
    #     align_file=None,
    #     feat_len_file=None,
    #     args=args,
    #     _shuffle=False,
    #     transform=False)
    # dataset_untrain = ASR_align_ArkDataSet(
    #     scp_file=args.dirs.untrain.scp,
    #     trans_file=None,
    #     align_file=None,
    #     feat_len_file=None,
    #     args=args,
    #     _shuffle=True,
    #     transform=False)
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
    feature_train = TFData(dataset=dataset_train,
                    dir_save=args.dirs.train.tfdata,
                    args=args)
    # feature_untrain = TFData(dataset=dataset_untrain,
    #                 dir_save=args.dirs.untrain.tfdata,
    #                 args=args)
    # feature_train_supervise = TFData(dataset=dataset_train_supervise,
    #                 dir_save=args.dirs.train_supervise.tfdata,
    #                 args=args)
    feature_dev = TFData(dataset=dataset_dev,
                    dir_save=args.dirs.dev.tfdata,
                    args=args)
    feature_train.split_save(capacity=100000)
    feature_dev.split_save(capacity=100000)
    # feature_untrain.split_save(capacity=100000)
    # feature_train_supervise.save('0')

    # get_bucket(args.dirs.train.tfdata / 'feature_length.txt', args.num_batch_tokens, 550)

    # dataset_train = ASR_align_DataSet(
    #     trans_file=args.dirs.train.trans,
    #     align_file=args.dirs.train.align,
    #     uttid2wav=args.dirs.train.wav_scp,
    #     feat_len_file=args.dirs.train.feat_len,
    #     args=args,
    #     _shuffle=False,
    #     transform=True)
    # dataset_dev = ASR_align_DataSet(
    #     trans_file=args.dirs.dev.trans,
    #     uttid2wav=args.dirs.dev.wav_scp,
    #     align_file=args.dirs.dev.align,
    #     feat_len_file=args.dirs.dev.feat_len,
    #     args=args,
    #     _shuffle=False,
    #     transform=True)
    # print(dataset_dev[0]['align'])
    # print(dataset_dev[0]['trans'])
    # print([dataset_dev.idx2token[i] for i in dataset_dev[0]['trans']])
    # import pdb; pdb.set_trace()
    # split_save()
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


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    main()
