#!/bin/bash
"""
Before run this script, do these things first:
    - replace the `timit_data_prep.sh` with a modified one here.
"""

. ./cmd.sh
[ -f path.sh ] && . ./path.sh
set -e

feats_nj=10
train_nj=30
decode_nj=5

echo ============================================================================
echo "                Data & Lexicon & Language Preparation                     "
echo ============================================================================

#timit=/export/corpora5/LDC/LDC93S1/timit/TIMIT # @JHU
#timit=/mnt/matylda2/data/TIMIT/timit # @BUT
timit=/data/sxu/easton/data/TIMIT

local/timit_data_prep.sh $timit || exit 1

local/timit_prepare_dict.sh

# Caution below: we remove optional silence by setting "--sil-prob 0.0",
# in TIMIT the silence appears also as a word in the dictionary and is scored.
utils/prepare_lang.sh --sil-prob 0.0 --position-dependent-phones false --num-sil-states 3 \
 data/local/dict "sil" data/local/lang_tmp data/lang

local/timit_format_data.sh

echo ============================================================================
echo "         MFCC Feature Extration & CMVN for Training and Test set          "
echo ============================================================================

# Now make MFCC features.
mfccdir=mfcc

for x in train test; do
  steps/make_mfcc.sh --cmd "$train_cmd" --nj $feats_nj data/$x exp/make_mfcc/$x $mfccdir
  steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir
done
