# Unsupervised phone classification
Implementation of
[EODM](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwieh8LU68DsAhVMfSsKHaJdAEAQFjABegQIAhAC&url=https%3A%2F%2Fopenreview.net%2Fforum%3Fid%3DBylmkh05KX&usg=AOvVaw2yHfCik8RA7OlONom5IwiW).

## Environments
- Tensorflow == 2.2
- Python == 3.6

## EODM exps (timit)

======= update in 2020/10/22 ========

related files:
- models/EODM.py
- main_EODM.py
- utils/tools.py

data
- data/timit


### data prepare
All the files needed for timit demo are already in `data/timit`. You need to generate `tfrecord` first:
```
python data_prepare.py configs/timit/timit_EODM.yaml
```
`0.recode` and `tfdata.info` will appear in  `data/timit/train_feats/` and `data/timit/test_feats/` dirs.
**NOTE**

The `tfdata.info` is the sample info stored in `0.record`, where `dim_feature` should be the dim of processed feature (by `dataProcess/process_raw_feature`). `ASR_align_DataSet` should be created with `transform=True` in `data_prepare.py`.

### training
```bash
python main_EODM.py --gpu 1 -c configs/timit/timit_EODM.yaml
```
, where `--gpu` is the free gpu index in your machine.
I realize Eq.1 by applying Conv_1d on the model's output, i.e. the distribution sequence.
`ngram2kernel` is to convert ngram to a kernel and `P_Ngram` is the model to use the fixed ngram kernel to compute EODM loss.

You can check my inconplete log file in `logs/train_timit_EODM.log`.

**NOTE**

I found that pure unsupervised training is unstable and only converges at specific seeds. So I add few labelled samples (250) to make sure the convergence.
==========

Performance of one layer fully-connected classifier:

|Corpus| Supervised | EODM | GAN | GAN + 250 paired | EODM + 250 paired |
|:-----:|-------------|---|:-----:| :-----: | :-----: |
| TIMIT | 26.5 | 40 | 48 | 35.5 | 33.4 |
| AIShell-2 |  30.52 |  - | -  | 42.5  |   |
| LibriSpeech | 28.0  | -  | -  | 41.2  |   |
