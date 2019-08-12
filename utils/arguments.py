import logging
import sys
import yaml
from pathlib import Path
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')

from .dataProcess import load_vocab
from .tools import TFData, mkdirs


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getattr__(self, item):
        try:
            if type(self[item]) is dict:
                self[item] = AttrDict(self[item])
            res = self[item]
        except:

            print('not found {}'.format(item))
            res = None
        return res

CONFIG_FILE = sys.argv[-1]
args = AttrDict(yaml.load(open(CONFIG_FILE), Loader=yaml.SafeLoader))

args.num_gpus = len(args.gpus.split(','))
args.list_gpus = ['/gpu:{}'.format(i) for i in range(args.num_gpus)]

# dirs
dir_dataInfo = Path.cwd() / 'data'
dir_exps = Path.cwd() / 'exps'
args.dir_exps = dir_exps / CONFIG_FILE.split('/')[-1].split('.')[0]
args.dir_log = args.dir_exps / 'log'
args.dir_checkpoint = args.dir_exps / 'checkpoint'
args.dirs.train.tfdata = Path(args.dirs.train.tfdata) if args.dirs.train.tfdata else None
args.dirs.dev.tfdata = Path(args.dirs.dev.tfdata) if args.dirs.dev.tfdata else None

if not dir_dataInfo.is_dir(): dir_dataInfo.mkdir()
if not dir_exps.is_dir(): dir_exps.mkdir()
if not args.dir_exps.is_dir(): args.dir_exps.mkdir()
if not args.dir_log.is_dir(): args.dir_log.mkdir()
if not args.dir_checkpoint.is_dir(): args.dir_checkpoint.mkdir()

# vocab
args.token2idx, args.idx2token = load_vocab(args.dirs.vocab)
args.dim_output = len(args.token2idx)

args.dirs.train.tfdata = Path(args.dirs.train.tfdata)
args.dirs.dev.tfdata = Path(args.dirs.dev.tfdata)
mkdirs(args.dirs.train.tfdata)
mkdirs(args.dirs.dev.tfdata)
args.dirs.train.feat_len = args.dirs.train.tfdata/'feature_length.txt'
args.dirs.dev.feat_len = args.dirs.dev.tfdata/'feature_length.txt'

try:
    args.dirs.train_supervise.tfdata = Path(args.dirs.train_supervise.tfdata)
    mkdirs(args.dirs.train_supervise.tfdata)
    args.dirs.train_supervise.feat_len = args.dirs.train_supervise.tfdata/'feature_length.txt'
except:
    print("not found train_supervise")

try:
    args.dim_input = TFData.read_tfdata_info(args.dirs.train.tfdata)['dim_feature']
    args.data.train_size = TFData.read_tfdata_info(args.dirs.train.tfdata)['size_dataset']
    args.data.dev_size = TFData.read_tfdata_info(args.dirs.dev.tfdata)['size_dataset']
except:
    print("have not converted to tfdata yet: ")
