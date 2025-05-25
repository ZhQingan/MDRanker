
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# python main_multi.py \
#  --cuda \
#  --name cnndm_AR.sc_mod_base.Dyn_Norm_margin_hyper0.15.span_3.weight_rk1_mle0.rank_by_rg \
#  --gpuid 0 1 \
#  --logits_merge softmax \
#  --score_smooth 10 \
#  --task_smooth 10 \
#  --config cnndm \
#  -p 12333
 
python main.py \
 --cuda \
 --name cnndm_AR.sc_mod_base.Dyn_Norm_margin_hyper0.15.span_3.weight_rk1_mle0.rank_by_rg \
 --gpuid 4 5 6 7 \
 --rank_by 0 \
 --config cnndm \
 -p 12333

# python main.py \
#  --cuda \
#  --name cnndm_AR.sc_mod_base.Dyn_Norm_margin_hyper0.15.span_3.weight_rk1_mle0.rank_by_bs \
#  --gpuid 2 3 \
#  --rank_by 1 \
#  --config cnndm \
#  -p 12331

# python main.py \
#  --cuda \
#  --name cnndm_AR.sc_mod_base.Dyn_Norm_margin_hyper0.15.span_3.weight_rk1_mle0.rank_by_ppl \
#  --gpuid 4 5 \
#  --rank_by 2 \
#  --config cnndm \
#  -p 12332