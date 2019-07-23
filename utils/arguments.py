import logging
import sys
import yaml
from pathlib import Path
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')

from .dataProcess import load_vocab
from eastonCode.tfTools.tfData import TFData


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
args.list_bucket_boundaries = [int(i) for i in args.bucket_boundaries.split(',')]
args.list_batch_size = ([int(args.num_batch_tokens / boundary) * max(args.num_gpus, 1)
        for boundary in (args.list_bucket_boundaries)] + [max(args.num_gpus, 1)])

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


try:
    args.dim_input = TFData.read_tfdata_info(args.dirs.train.tfdata)['dim_feature']
    args.data.train_size = TFData.read_tfdata_info(args.dirs.train.tfdata)['size_dataset']
    args.data.dev_size = TFData.read_tfdata_info(args.dirs.dev.tfdata)['size_dataset']

    args.dirs.train.tfdata = Path(args.dirs.train.tfdata)
    args.dirs.dev.tfdata = Path(args.dirs.dev.tfdata)
    if not args.dirs.train.tfdata.is_dir(): args.dirs.train.tfdata.mkdir()
    if not args.dirs.dev.tfdata.is_dir(): args.dirs.dev.tfdata.mkdir()
except:
    print("have not converted to tfdata yet: ")

if args.dirs.lm_config:
    args.args_lm = AttrDict(yaml.load(open(args.dirs.lm_config), Loader=yaml.SafeLoader))
    args.args_lm.dim_output = len(args.token2idx)
