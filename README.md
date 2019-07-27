# Unsupervised phone classification

## Environments
- Tensorflow == 2.0.0.beta
- Python == 3.7

## Exps
Performance of one layer fully-connected classifier:

|Corpus| Supervised | EODM | GAN | GAN + 250 paired | EODM + 250 paired |
|:-----:|-------------|---|:-----:| :-----: | :-----: |
| TIMIT | 26.5 | 40 | 48 | 35.5 | 33.4 |
| AIShell-2 |  30.52 |  - | -  | 42.5  |   |
| LibriSpeech | 28.0  | -  | -  | 41.2  |   |
