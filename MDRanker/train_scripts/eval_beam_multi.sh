
# python eval_beam_multi.py \
#   -e -r \
#   --cuda \
#   --config xsum \
#   --gpuid 1 \
#   --model_type facebook/bart-large-cnn \
#   --name cnndm_sifted_bs_ppl_rg.Dyn_Norm_margin_hyper0.01.span_3.weight_rk1_mle0.001.rank_multi_trainable_softmax.fp16

python eval_beam_multi.py \
  -e -r \
  --cuda \
  --config xsum \
  --gpuid 6 \
  --model_type google/pegasus-xsum \
  --name xsum_sifted_bs_ppl_rg.Dyn_Norm_margin_hyper0.15.span_3.weight_rk1_mle0.1.rank_multi_trainable_softmax.grad_ckpt
