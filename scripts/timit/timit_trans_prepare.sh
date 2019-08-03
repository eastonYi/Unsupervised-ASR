. ./cmd.sh
[ -f path.sh ] && . ./path.sh
set -e

# Acoustic model parameters
feats_nj=10

echo ============================================================================
echo "                Data & Lexicon & Language Preparation                     "
echo ============================================================================

timit=/data/sxu/easton/data/TIMIT

local/timit_data_prep.sh $timit || exit 1

local/timit_prepare_dict.sh

# Caution below: we remove optional silence by setting "--sil-prob 0.0",
# in TIMIT the silence appears also as a word in the dictionary and is scored.
utils/prepare_lang.sh --sil-prob 0.0 --position-dependent-phones false --num-sil-states 3 \
 data/local/dict "sil" data/local/lang_tmp data/lang

local/timit_format_data.sh

for x in train dev test; do
  cp data/$x/text data/$x/trans_ground
  steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir
done

echo ============================================================================
echo "         MFCC Feature Extration & CMVN for Training and Test set          "
echo ============================================================================

# Now make MFCC features.
mfccdir=mfcc


for x in train dev test; do
  steps/make_mfcc.sh --cmd "$train_cmd" --nj $feats_nj data/$x exp/make_mfcc/$x $mfccdir
  steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir
done
