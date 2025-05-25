
python main.py \
 -e -c -r \
 --test_on_val \
 --cuda \
 --config cnndm \
 --gpuid 4 \
 --name cnndm_sifted.Dyn_Norm_margin_hyper0.1.span_5.weight_1.rank_rg_only

python main.py \
 -e -c -r \
 --test_on_val \
 --cuda \
 --config cnndm \
 --gpuid 4 \
 --name cnndm_sifted.Dyn_Norm_margin_hyper0.15.span_3.weight_1.rank_rg_only