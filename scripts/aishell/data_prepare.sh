#!/bin/bash
data=/data/sxu/easton/data/AISHELL

# Lexicon Preparation,
local/aishell_prepare_dict.sh $data || exit 1;

# Data Preparation,
local/aishell_data_prep.sh $data/wav $data/transcript || exit 1;

# Phone Sets, questions, L compilation
utils/prepare_lang.sh --position-dependent-phones false data/local/dict \
    "<SPOKEN_NOISE>" data/local/lang data/lang || exit 1;

# LM training
local/aishell_train_lms.sh || exit 1;

# G compilation, check LG composition
utils/format_lm.sh data/lang data/local/lm/3gram-mincount/lm_unpruned.gz \
    data/local/dict/lexicon.txt data/lang_test || exit 1;

# Now make MFCC plus pitch features.
# mfccdir should be some place with a largish disk where you
# want to store MFCC features.
mfccdir=mfcc
for x in train dev test; do
  steps/make_mfcc_pitch.sh --cmd "$train_cmd" --nj 10 data/$x exp/make_mfcc/$x $mfccdir || exit 1;
  steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir || exit 1;
  utils/fix_data_dir.sh data/$x || exit 1;
done
