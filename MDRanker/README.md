

# Ranking Augmented Generation: Leveraging Diversity for Better Text Generation

The paper is currently under review.

This project references the codes in repo (https://github.com/yixinL7/BRIO).

## Description of Codes
- `main_multi.py` -> code for train/eval/test
- `models/label_smoothing_loss.py` -> label smoothing loss
- `models/data_utils.py` -> Dataset and Dataloader
- `models/model_multi.py` -> codes for model structure and training loss
- `models/config.py` -> hyperparameters in `main_multi.py`
- `models/modeling_bart_multi.py`, `models/modeling_pegasus_multi.py`, `models/modeling_mbart_multi.py`  -> modefied codes from huggingface.
- `utils/*` -> training tools

# Preprocessing

We use the following datasets for our experiments.

- CNN/DM -> https://github.com/abisee/cnn-dailymail
- XSUM -> https://github.com/EdinburghNLP/XSum
- IWSLT 2014 -> https://wit3.fbk.eu/2014-01

### Preprocessed Data

We use [AR-Diffusion](https://github.com/microsoft/ProphetNet/tree/master/AR-diffusion) to build datasets.

Some data samples are shown in `data/*`.

## How to Run
### Train
```console
python main_multi.py --cuda --gpuid [list of gpuid] --config [name of dataset] --name [name of your model] -p [port for communication]
```

### Eval
```console
python main_multi.py -e --test_on_val --cuda --gpuid [list of gpuid] --config [name of dataset] --name [name of your model]
```

### Test
```console
python main_multi.py -e --cuda --gpuid [list of gpuid] --config [name of dataset] --name [name of your model]
```