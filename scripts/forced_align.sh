. ./cmd.sh
[ -f path.sh ] && . ./path.sh
set -e

feats_nj=10
train_nj=30

train_cmd="run.pl --mem 4G"

echo ============================================================================
echo "                     MonoPhone Training and Forced Align                  "
echo ============================================================================
steps/train_mono.sh  --nj "$train_nj" --cmd "$train_cmd" data/train data/lang exp/mono

steps/align_si.sh --boost-silence 1.25 --nj "$train_nj" --cmd "$train_cmd" \
 data/train data/lang exp/mono exp/mono_ali

cd exp/mono_ali
ali-to-phones --per-frame=true final.mdl "ark:gunzip -c ali.*.gz |" ark,t:train.phone.frame
