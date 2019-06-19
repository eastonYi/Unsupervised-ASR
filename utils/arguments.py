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
dir_models = Path.cwd() / 'models'
args.dir_model = dir_models / CONFIG_FILE.split('/')[-1].split('.')[0]
args.dir_log = args.dir_model / 'log'
args.dir_checkpoint = args.dir_model / 'checkpoint'
args.dirs.train.tfdata = Path(args.dirs.train.tfdata)
args.dirs.dev.tfdata = Path(args.dirs.dev.tfdata)

if not dir_dataInfo.is_dir(): dir_dataInfo.mkdir()
if not dir_models.is_dir(): dir_models.mkdir()
if not args.dir_model.is_dir(): args.dir_model.mkdir()
if not args.dir_log.is_dir(): args.dir_log.mkdir()
if not args.dir_checkpoint.is_dir(): args.dir_checkpoint.mkdir()
if not args.dirs.train.tfdata.is_dir(): args.dirs.train.tfdata.mkdir()
if not args.dirs.dev.tfdata.is_dir(): args.dirs.dev.tfdata.mkdir()

# vocab
args.token2idx, args.idx2token = load_vocab(args.dirs.vocab)
args.dim_output = len(args.token2idx)

try:
    args.dim_input = TFData.read_tfdata_info(args.dirs.train.tfdata)['dim_feature']
    args.data.train_size = TFData.read_tfdata_info(args.dirs.train.tfdata)['size_dataset']
    args.data.dev_size = TFData.read_tfdata_info(args.dirs.dev.tfdata)['size_dataset']
except:
    print("have not converted to tfdata yet: ")

if args.dirs.lm_config:
    args.args_lm = AttrDict(yaml.load(open(args.dirs.lm_config), Loader=yaml.SafeLoader))
    args.args_lm.dim_output = len(args.token2idx)

# model
if args.model.structure == 'fc':
    from utils.model import FC_Model as Model
elif args.model.structure == 'lstm':
    from utils.model import LSTM_Model as Model
elif args.model.structure == 'conv':
    from utils.model import Conv_Model as Model
args.Model = Model
