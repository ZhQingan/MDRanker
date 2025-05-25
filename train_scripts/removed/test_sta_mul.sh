
python main.py \
 -e -c -r \
 --test_on_val \
 --cuda \
 --config cnndm \
 --gpuid 6 \
 --name cnndm.multi_rank.span_3.margin_1e-3.weight_1.rank_only

python main.py \
 -e -c -r \
 --test_on_val \
 --cuda \
 --config cnndm \
 --gpuid 6 \
 --name cnndm.multi_rank.span_5.margin_1e-3.weight_1.rank_only