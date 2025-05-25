


python main_multi.py \
 -e -c -r \
 --cuda \
 --config iwslt14_deen \
 --gpuid 5 \
 --name iwslt14_deen_AR.sc_mod_base.Dyn_Norm_margin_hyper0.015.span_3.weight_rk10_mle0.01.score_smooth_0.9.rank_multi_trainable_softmax

python main_multi.py \
 -e -c -r \
 --cuda \
 --config iwslt14_ende \
 --gpuid 5 \
 --name iwslt14_ende_AR.sc_mod_base.Dyn_Norm_margin_hyper0.015.span_3.weight_rk10_mle0.01.score_smooth_0.9.rank_multi_trainable_softmax
