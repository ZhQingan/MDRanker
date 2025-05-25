
# python eval_beam.py \
#   -e -c -r \
#   --cuda \
#   --config cnndm \
#   --gpuid 3 \
#   --name cnndm_sifted.Dyn_Norm_margin_hyper0.1.span_5.weight_1.rank_rg_only

python eval_beam.py \
  -e -c -r \
  --cuda \
  --config xsum \
  --gpuid 5 \
  --name xsum_sifted.Dyn_Norm_margin_hyper0.15.weight_1.rank_rg_only