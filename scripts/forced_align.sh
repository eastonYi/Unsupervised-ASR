. ./cmd.sh
[ -f path.sh ] && . ./path.sh
set -e

feats_nj=10
train_nj=30

train_cmd="run.pl --mem 4G"

echo ============================================================================
echo "                     MonoPhone Training and Forced Align                  "
echo ============================================================================
steps/train_mono.sh  --nj "$train_nj" --cmd "$train_cmd" data/train_gen data/lang exp/iter1
# steps/train_mono.sh --nj 1 --cmd run.pl --mem 4G data/train_gen data/lang exp/iter1

steps/align_si.sh --boost-silence 1.25 --nj "$train_nj" --cmd "$train_cmd" \
 data/train_gen data/lang exp/iter1 exp/iter1_ali

cd exp/iter1_ali
ali-to-phones --per-frame=true final.mdl "ark:gunzip -c ali.*.gz |" ark,t:train.phone.frame
