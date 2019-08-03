from pathlib import Path
from argparse import ArgumentParser
import os

from utils.tools import mkdirs


def main(timit_dir):
    for org_dir, new_dir, wav_scp in zip(['TRAIN', 'TEST'],
                                         ['train_wavs', 'test_wavs'],
                                         ['train_wav.scp', 'test_wav.scp']):
        print("Generating {} ...".format(wav_scp))
        p = Path(timit_dir+org_dir)
        with open(timit_dir+wav_scp, 'w') as fw:
            for i, org_wav in enumerate(p.glob('*/*/*.WAV')):
                f_wav = Path(str(org_wav).replace(org_dir, new_dir).replace('WAV', 'wav'))
                mkdirs(f_wav)
                os.system('~/easton/projects/kaldi/tools/sph2pipe_v2.5/sph2pipe -f wav {} {}'.format(org_wav, f_wav))
                # /data/sxu/easton/data/TIMIT/TRAIN/DR2/FAEM0/SI2022.WAV
                uttid = '_'.join(str(f_wav).split('.')[0].split('/')[-2:])
                fw.write(uttid + ' ' + str(f_wav) + '\n')
                if i % 999 == 0:
                    print('\tprocessed {} wavs'.format(i))
        print("Generate {} success".format(str(timit_dir+wav_scp)))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, dest='dir', default='/data/sxu/easton/data/TIMIT/')
    args = parser.parse_args()

    main(args.dir)

    print("Done")
