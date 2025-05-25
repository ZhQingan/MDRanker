
python main.py \
 -e -c -r \
 --test_on_val \
 --cuda \
 --config cnndm \
 --gpuid 7 \
 --name cnndm.uni_attn.weight_1.rank_only

python main.py \
 -e -c -r \
 --test_on_val \
 --cuda \
 --config cnndm \
 --gpuid 7 \
 --name cnndm.uni_attn.rank_only