import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import os
import random
# from compare_mt.rouge.rouge_scorer import RougeScorer
from transformers import BartTokenizer, PegasusTokenizer, MBart50Tokenizer
import transformers
# from utils import Recorder
from utils.utils import _save
from models.data_utils import to_cuda, to_cpu, collate_mp_brio, BrioDataset, RANKED_BY, METRIC
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from functools import partial
from models.model_multi import RankingLoss, BRIO, RankingLoss_smooth
from models.label_smoothing_loss import label_smoothing_loss
from nltk import sent_tokenize, word_tokenize
from models.config import cnndm_setting, xsum_setting, iwslt14_deen_setting, iwslt14_ende_setting
from tqdm import tqdm
from torch.cuda import amp
import torch.nn.functional as F

from utils.logging import init_logger, logger


# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def _tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    return n_params

def base_setting(args):
    args.total_batch_size = getattr(args, 'total_batch_size', 16) # batch size in one step
    args.batch_size = getattr(args, 'batch_size', 1) # batch size on one gpu, one step
    args.epoch = getattr(args, 'epoch', 100) 
    args.report_freq = getattr(args, "report_freq", 100) # report frequency
    args.accumulate_step = getattr(args, "accumulate_step", -1) # accumulate gradients steps
    args.margin = getattr(args, "margin", 0.001) # margin for ranking loss on candidate summaries
    args.gold_margin = getattr(args, "gold_margin", 0) # margin for ranking loss on gold summaries
    args.gold_weight = getattr(args, "gold_weight", 0) # weight for ranking loss on gold summaries
    args.mle_weight = getattr(args, "mle_weight", 1) # weight for mle loss on gold summaries
    args.rank_weight = getattr(args, "rank_weight", 1) # weight for ranking loss on candidate summaries
    args.model_type = getattr(args, "model_type", "facebook/bart-large-cnn") # model type
    args.warmup_steps = getattr(args, "warmup_steps", 10000) # warmup steps
    args.normalize = getattr(args, "normalize", True) # normalize predicited likelihood
    args.grad_norm = getattr(args, "grad_norm", 0) # gradient norm
    args.seed = getattr(args, "seed", 970903) # random seed
    args.no_gold = getattr(args, "no_gold", False) # whether to use gold summaries
    args.pretrained = getattr(args, "pretrained", None) # pretrained model path
    args.max_lr = getattr(args, "max_lr", 2e-3) # max learning rate (* 1e-2)
    args.scale = getattr(args, "scale", 1) # scale of ranking loss
    args.score_mode = getattr(args, "score_mode", "log") # use log-likelihood for ranking loss
    args.datatype = getattr(args, "datatype", "diverse") # data type
    args.dataset = getattr(args, "dataset", "cnndm") # dataset
    args.max_len = getattr(args, "max_len", 143) # max length of summary
    #args.max_len = getattr(args, "max_len", 120) # max length of summary
    args.max_num = getattr(args, "max_num", 16) # max number of candidate summaries
    args.smooth = getattr(args, "smooth", 0.1) # label smoothing
    args.total_len = getattr(args, "total_len", 1024) # total length of source article
    args.length_penalty = getattr(args, "length_penalty", 2.0) # length penalty
    args.do_sample = getattr(args, "do_sample", True) # whether to generaet summaries during evaluation
    args.gen_max_len = getattr(args, "gen_max_len", 140) # max length of generated summaries
    args.gen_min_len = getattr(args, "gen_min_len", 55) # min length of generated summaries
    args.is_pegasus = getattr(args, "is_pegasus", False) # whether to use Pegasus as the baseline model
    args.adding = getattr(args, "adding", 0) # used for numerical stability
    args.eval_interval = getattr(args, "eval_interval", 1000) # evaluation intervals
    args.num_beams = getattr(args, "num_beams", 4) # number of beams for beam search


def statistic(max_id, scores):
    sc_index = list(enumerate(scores))
    re = [0]*3
    for i in range(3):
        sc_index = sorted(sc_index, key=lambda x:x[1][i], reverse=True)
        idx = [x[0] for x in sc_index].index(max_id)
        while idx:
            if sc_index[idx-1][1][i] - sc_index[idx][1][i] < 1e-6:
                idx -= 1
            else:
                break
        re[i] = idx
    return re

def report_ranking_stat(stats_org):
    ranked_by = RANKED_BY
    total = sum(stats_org[0])
    stats = stats_org / total
    for j in range(3):
        stat = stats[j]
        info_txt = ""
        tot = 0
        for k in range(len(stat)):
            info_txt += "|top@%d: %5.2f|"%(k+1, sum(stat[:k+1])*100)
            tot += (k+1)*stat[k]
        logger.info(ranked_by[j] + info_txt)
        logger.info(ranked_by[j] + "|Mean Rank: %.4f"%tot)

def evaluation(args):
    # load data
    init_logger(args.save_path + f"/{args.name}.log")
    
    logger.info("config is %s" % args.config)
    if args.config == "cnndm":
        cnndm_setting(args)
    elif args.config == "xsum":
        xsum_setting(args)
    elif args.config == "iwslt14_ende":
        iwslt14_ende_setting(args)
    elif args.config == "iwslt14_deen":
        iwslt14_deen_setting(args)
    else:
        base_setting(args)
    
    if args.is_pegasus:
        tok = PegasusTokenizer.from_pretrained(args.model_type)
    elif 'mbart' in args.model_type:
        tok = MBart50Tokenizer.from_pretrained(args.model_type)
    else:
        tok = BartTokenizer.from_pretrained(args.model_type)
    collate_fn = partial(collate_mp_brio, pad_token_id=tok.pad_token_id, is_test=True)
    
    if not args.test_on_val:
        use_set = "Test"
        test_set = BrioDataset(f"./{args.dataset}/{args.datatype}/test", args.model_type, is_test=True, max_len=args.max_len,
                             max_num=args.max_num, is_untok=True, total_len=args.total_len, is_pegasus=args.is_pegasus)
    else:
        use_set = "Validation"
        test_set = BrioDataset(f"./{args.dataset}/{args.datatype}/val", args.model_type, is_test=True, max_len=args.max_len, max_num=args.max_num, total_len=args.total_len, is_pegasus=args.is_pegasus)

    batch_size = 1
    gen_batch_size = 8
    dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    gen_dataloader = DataLoader(test_set, batch_size=gen_batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    # build models
    model_path = args.pretrained if args.pretrained is not None else args.model_type
    model_name = args.name
    reranking_pt = "./checkpoints/%s/model_ranking.bin"%model_name
    accuracy_pt = "./checkpoints/%s/model_accuracy.bin"%model_name
    generation_pt = "./checkpoints/%s/model_generation.bin"%model_name
    cur_pt = "./checkpoints/%s/model_cur.bin"%model_name

    if "softmax" in model_name:
        logits_merge = "softmax"
    elif "trainable" in model_name:
        logits_merge = "trainable"
    else:
        logits_merge = "softmax"
    if not os.path.exists(generation_pt):
        logger.info("No generation model: '%s'"%generation_pt)
        generation_pt = "./checkpoints/%s/model_cur.bin"%model_name
        logger.info("Substitude with '%s'"%generation_pt)
    
    model = BRIO(model_path, tok.pad_token_id, tok.eos_token_id, args.trunc_eos, args.is_pegasus, logits_merge)
    if args.cuda:
        model = model.cuda()

    def mkdir(path):
        if not os.path.exists(path):
            os.mkdir(path)
    
    def log_coef():
        if logits_merge == 'trainable':
            a = model.model.A.data.item()
            b = model.model.B.data.item()
            logger.info("coef: %.4f; %.4f; %.4f"%(a,b,(1-a-b)))
        elif logits_merge == 'softmax':
            if "softmax_10" in model_name:
                c = F.softmax(model.model.C.data*10, dim=0)
            else:
                c = F.softmax(model.model.C.data, dim=0)
            logger.info("coef: %.4f; %.4f; %.4f"%tuple(c))

    logger.info(model_name)
    logger.info(args.dataset+"/"+args.datatype)
    root_dir = "./results/%s/%s"%(args.dataset,model_name)
    mkdir(root_dir)

    ranked_by = RANKED_BY

    def do_reranking(ckpt_pt, outer_file):
        if not os.path.exists(ckpt_pt):
            logger.info("No Such path: %s"%ckpt_pt)
            return
        logger.info("Begin Reranking Test on %s"%ckpt_pt)
        logger.info("Using %s Set do Test"%use_set)
        model.model.load_state_dict(torch.load(ckpt_pt, map_location=f'cuda:{args.gpuid[0]}'), strict=False)
        device = f'cuda:{args.gpuid[0]}'
        model.eval()
        log_coef()
        
        rouge1, rouge2, rougeLsum = 0, 0, 0
        cnt = 0

        stats = torch.zeros(4, 3, args.max_num).to('cuda')

        model.scoring_mode()
        if args.test_on_val:
            outer_rank = open("./results/%s/test_val/%s/%s"%(args.dataset,model_name,outer_file), "w", encoding='utf-8')
        else:
            outer_rank = open("./results/%s/%s/%s"%(args.dataset,model_name,outer_file), "w", encoding='utf-8')
        with amp.autocast(enabled=args.amp, dtype=torch.bfloat16 if args.float_type=='bf16' else torch.float16):
            with torch.no_grad():

                for batch in dataloader:
                    if args.cuda:
                        to_cuda(batch, args.gpuid[0])
                    samples = batch["data"]

                    if args.max_num <=32:
                        output = model(batch["src_input_ids"], batch["candidate_ids"], 1.0, args.normalize, args.score_mode, args.length_penalty, adding=args.adding)
                        similarity = output['score']
                        similarity = similarity.cpu().numpy()
                    else:
                        mini_num = 4
                        le = mini_num+1 #left edge
                        similaritys = []
                        while le-mini_num < len(batch["candidate_ids"][0]):
            
                            output = model(batch["src_input_ids"],
                                        torch.concat((batch["candidate_ids"][:,:1],batch["candidate_ids"][:, le-mini_num: le]),dim=1), 1.0, args.normalize, args.score_mode, args.length_penalty, adding=args.adding)
                            
                            similaritys.append(output['score'])
                            le += mini_num
                        similarity = torch.concat(similaritys,dim=-1)
                        similarity = similarity.cpu().numpy()
                    
                    
                    max_ids = similarity[3].argmax(1)
                    
                    for j in range(similarity[3].shape[0]):
                        sample = samples[j]
                        sents = sample["candidates"][max_ids[j]][0]

                        ranked = statistic(max_ids[j], batch["candidate_scores"][j])
                        for i in range(3):

                            stats[3][i][ranked[i]] += 1
                       
                        rouge1 += batch["candidate_scores"][j][max_ids[j]][0]
                        rouge2 += batch["candidate_scores"][j][max_ids[j]][1]
                        rougeLsum += batch["candidate_scores"][j][max_ids[j]][2]
                        outer_rank.write(" ".join(sample["candidates_untok"][max_ids[j]][0]) + '\n')
                        cnt += 1
                    
                    for k in range(3):
                        max_ids = similarity[k].argmax(1)
                        for j in range(similarity[k].shape[0]):
                            ranked = statistic(max_ids[j], batch["candidate_scores"][j])
                            for i in range(3):
                                stats[k][i][ranked[i]] += 1
                    
        rouge1 = rouge1 / cnt
        rouge2 = rouge2 / cnt
        rougeLsum = rougeLsum / cnt
        logger.info("ranking rouge1: %.6f, rouge2: %.6f, rougeL: %.6f"%(rouge1, rouge2, rougeLsum))
        

        logger.info("Comprehensize Ranking")
        report_ranking_stat(stats[3])
        logger.info("Respective Ranking")
        for i in range(3):
            logger.info("Expert %s"%ranked_by[i])
            report_ranking_stat(stats[i])
        
        total = sum(stats[0][0])
        logger.info("Accuracy:%s/%s/%s/Avg"%(ranked_by[0],ranked_by[1],ranked_by[2]))
        logger.info("Expert %s: %5.2f | %5.2f | %5.2f || %5.2f"%(ranked_by[0],stats[0][0][0]/total*100,stats[0][1][0]/total*100,stats[0][2][0]/total*100,(stats[0][0][0]+stats[0][1][0]+stats[0][2][0])/3/total*100))
        logger.info("Expert %s: %5.2f | %5.2f | %5.2f || %5.2f"%(ranked_by[1],stats[1][0][0]/total*100,stats[1][1][0]/total*100,stats[1][2][0]/total*100,(stats[1][0][0]+stats[1][1][0]+stats[1][2][0])/3/total*100))
        logger.info("Expert %s: %5.2f | %5.2f | %5.2f || %5.2f"%(ranked_by[2],stats[2][0][0]/total*100,stats[2][1][0]/total*100,stats[2][2][0]/total*100,(stats[2][0][0]+stats[2][1][0]+stats[2][2][0])/3/total*100))
        logger.info("Expert ALL: %5.2f | %5.2f | %5.2f || %5.2f"%(stats[3][0][0]/total*100,stats[3][1][0]/total*100,stats[3][2][0]/total*100, (stats[3][0][0]+stats[3][1][0]+stats[3][2][0])/3/total*100))

    do_reranking(accuracy_pt, "%s.accuracy.cand"%(args.dataset+"_"+args.datatype))

    if args.test_cur:
        do_reranking(cur_pt, "%s.reranking_cur.cand"%(args.dataset+"_"+args.datatype))
            

def test(dataloader, gen_dataloader, model, args, tok, gpuid, do_sample=False):
    model.eval()
    if args.cuda:
        device = f"cuda:{gpuid}"
    else:
        device = "cpu"
    if len(args.gpuid) > 1:
        _model = model.module
    else:
        _model = model
    cnt = 0

    rouge1, rouge2, rougeLsum = 0, 0, 0
    mle_loss = 0
    
    mle_fn = label_smoothing_loss(ignore_index=tok.pad_token_id, eos_id=tok.eos_token_id) #Removed log_softmax
    _model.scoring_mode()
    stats = torch.zeros(3, args.max_num).to('cuda')
    stats_res = torch.zeros(3, args.max_num).to('cuda')
    with amp.autocast(enabled=args.amp, dtype=torch.bfloat16 if args.float_type=='bf16' else torch.float16):
        with torch.no_grad():
            # scoring
            for (i, batch) in enumerate(dataloader):
                if args.cuda:
                    to_cuda(batch, device)
                samples = batch["data"]
                
                output = model(batch["src_input_ids"], batch["candidate_ids"], 1.0, args.normalize, args.score_mode, args.length_penalty, adding=args.adding)
                similarity = output['score']
                similarity = similarity.cpu().numpy()
                probs = output["probs"][:, :-1]  # truncate last token
                gold = batch["candidate_ids"][:, 0, 1:]  # shift right
                mle_loss += mle_fn(probs.transpose(1, 2), gold)

                # Test Comprehensive Ranking
                max_ids = similarity[3].argmax(1)
                for j in range(similarity[3].shape[0]):
                    cnt += 1
                    sample = samples[j]
                    
                    ranked = statistic(max_ids[j], batch["candidate_scores"][j])
                    for k in range(3):
                        stats[k][ranked[k]] += 1

                    rouge1 += batch["candidate_scores"][j][max_ids[j]][0]
                    rouge2 += batch["candidate_scores"][j][max_ids[j]][1]
                    rougeLsum += batch["candidate_scores"][j][max_ids[j]][2]

                # Test Respective Ranking
                for k in range(3):
                    max_ids = similarity[k].argmax(1)
                    for j in range(similarity[3].shape[0]):
                        ranked = statistic(max_ids[j], batch["candidate_scores"][j])
                        stats_res[k][ranked[k]] += 1

    rouge1 = rouge1 / cnt
    rouge2 = rouge2 / cnt
    rougeLsum = rougeLsum / cnt
    mle_loss = mle_loss / cnt

    if len(args.gpuid) > 1:
        rouge1 = torch.FloatTensor([rouge1]).to(device)
        dist.all_reduce(rouge1, op=dist.ReduceOp.SUM)
        rouge1 = rouge1.item() / len(args.gpuid)
        rouge2 = torch.FloatTensor([rouge2]).to(device)
        dist.all_reduce(rouge2, op=dist.ReduceOp.SUM)
        rouge2 = rouge2.item() / len(args.gpuid)
        rougeLsum = torch.FloatTensor([rougeLsum]).to(device)
        dist.all_reduce(rougeLsum, op=dist.ReduceOp.SUM)
        rougeLsum = rougeLsum.item() / len(args.gpuid)
        dist.all_reduce(mle_loss, op=dist.ReduceOp.SUM)
        mle_loss = mle_loss.item() / len(args.gpuid)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        dist.all_reduce(stats_res, op=dist.ReduceOp.SUM)
        
    model.train()
    return {
        "rouge1": rouge1,
        "rouge2": rouge2,
        "rougeLsum": rougeLsum,
        "mle_loss": mle_loss,
        "stats": stats,
        "stats_res": stats_res,
        } 


def run(rank, args):
    
    init_logger(args.save_path + f"/{args.name}.log")
    
    if args.config == "cnndm":
        cnndm_setting(args)
    elif args.config == "xsum":
        xsum_setting(args)
    elif args.config == "iwslt14_ende":
        iwslt14_ende_setting(args)
    elif args.config == "iwslt14_deen":
        iwslt14_deen_setting(args)
    else:
        base_setting(args)
    # task initialization
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    gpuid = args.gpuid[rank]
    torch.cuda.set_device(gpuid)
    is_master = rank == 0
    is_mp = len(args.gpuid) > 1
    world_size = len(args.gpuid)
    
    args.accumulate_step = args.total_batch_size / args.batch_size / world_size
    
    if is_master:
        logger.info("config is %s" % args.config)
        logger.info(str(args))
        logger.info("transformer_version = %s"%(transformers.__version__))

    if args.is_pegasus:
        tok = PegasusTokenizer.from_pretrained(args.model_type)
    elif 'mbart' in args.model_type:
        tok = MBart50Tokenizer.from_pretrained(args.model_type)
    else:
        tok = BartTokenizer.from_pretrained(args.model_type)
    collate_fn = partial(collate_mp_brio, pad_token_id=tok.pad_token_id, is_test=False)
    collate_fn_val = partial(collate_mp_brio, pad_token_id=tok.pad_token_id, is_test=True)
    train_set = BrioDataset(f"./{args.dataset}/{args.datatype}/train", args.model_type, max_len=args.max_len, max_num=args.train_max_num, total_len=args.total_len, is_pegasus=args.is_pegasus)

    val_set = BrioDataset(f"./{args.dataset}/{args.datatype}/val", args.model_type, is_test=True, max_len=args.max_len, max_num=args.max_num, total_len=args.total_len, is_pegasus=args.is_pegasus)
    
    if is_mp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_set, num_replicas=world_size, rank=rank, shuffle=True)
        dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn, sampler=train_sampler)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_set, num_replicas=world_size, rank=rank)
        val_dataloader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn_val, sampler=val_sampler)
        val_gen_dataloader = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=4, collate_fn=collate_fn_val, sampler=val_sampler)
    else:
        dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
        val_dataloader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn_val)
        val_gen_dataloader = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=4, collate_fn=collate_fn_val)
    # build models
    model_path = args.pretrained if args.pretrained is not None else args.model_type
    model = BRIO(model_path, tok.pad_token_id, tok.eos_token_id, args.trunc_eos, is_pegasus=args.is_pegasus, logits_merge=args.logits_merge)
    
    n_params = _tally_parameters(model)
    if is_master:
        logger.info('* number of parameters: %d' % n_params)
        
    if len(args.model_pt) > 0:
        model.load_state_dict(torch.load(os.path.join("./cache_%s"%args.dataset, args.model_pt), map_location=f'cuda:{gpuid}'))
    if args.cuda:
        if is_mp:
            # Using DDP
            dist.init_process_group("nccl", rank=rank, world_size=world_size)
            model = nn.parallel.DistributedDataParallel(model.to(gpuid), [gpuid], find_unused_parameters=False)
        else:
            #model = model.cuda()
            model.to(gpuid)
    model.train()
    # set the model to scoring mode
    if is_mp:
        model.module.scoring_mode()
    else:
        model.scoring_mode()

    mle_fn = label_smoothing_loss(ignore_index=tok.pad_token_id, eos_id=tok.eos_token_id, epsilon=args.smooth) #Removed log_softmax
    s_optimizer = optim.AdamW(model.parameters())

    minimum_ranking_loss = 100
    maximum_accuracy_loss = 0
    Max_val_R1 = 0
    Max_val_R2 = 0
    Max_val_RL = 0
    Max_val_R1_sam = 0
    Max_val_R2_sam = 0
    Max_val_RL_sam = 0
    Max_top1 = [0.0]*3
    Max_top5 = [0.0]*3
    # Max_top10 = [0.0]*3
    Min_mean = [1e6]*3
    val_R1 = (0,0,0)
    val_R2 = (0,0,0)
    val_RL = (0,0,0)
    val_R1_sam = (0,0,0)
    val_R2_sam = (0,0,0)
    val_RL_sam = (0,0,0)
    ranked_by = RANKED_BY
    minimum_mle_loss = 1e5
    all_step_cnt = 0

    def eval_fn(rg_mean, bs_mean, ppl_mean):
        return rg_mean
    def eval_fn_acc(rg_top, bs_top, ppl_top):
        return rg_top
    
    # start training
    scaler = amp.GradScaler(enabled=args.amp)
    if is_master:
        logger.info("Ranked by %s"%ranked_by[METRIC])
    for epoch in range(args.epoch):
        if is_mp:
            train_sampler.set_epoch(epoch)
        s_optimizer.zero_grad()
        avg_ranking_loss = 0
        avg_mle_loss = 0
        step_cnt = 0
        epoch_step = 0
        avg_loss = 0
        avg_rg_loss, avg_bs_loss, avg_ppl_loss = 0, 0, 0

        for (i, batch) in enumerate(dataloader):
            if args.cuda:
                to_cuda(batch, gpuid)
            step_cnt += 1
            # forward pass
            with amp.autocast(enabled=args.amp, dtype=torch.bfloat16 if args.float_type=='bf16' else torch.float16):
                output = model(batch["src_input_ids"], batch["candidate_ids"], args.score_smooth, args.normalize, args.score_mode, args.length_penalty, adding=args.adding)
                
                similarity, gold_similarity = output['score'], output['summary_score']
                similarity = similarity * args.scale
                gold_similarity = gold_similarity * args.scale
                if args.task_smooth < 1.0:
                    ranking_loss, loss_multi = RankingLoss_smooth(similarity,  gold_similarity, np.array(batch["candidate_scores"]), args.margin, args.gold_margin, args.gold_weight)
                else:
                    ranking_loss, loss_multi = RankingLoss(similarity, gold_similarity, np.array(batch["candidate_scores"]), args.margin, args.gold_margin, args.gold_weight)

                if not args.trunc_eos:
                    probs = output["probs"][:, :-1]  # truncate last token
                    gold = batch["candidate_ids"][:, 0, 1:]  # shift right
                else:
                    probs = output["probs"][:, :-2]  # truncate last token
                    gold = batch["candidate_ids"][:, 0, 1:-1]  # shift right

                mle_loss = mle_fn(probs.transpose(1, 2), gold)
                loss = args.rank_weight * ranking_loss + args.mle_weight * mle_loss

                loss = loss / args.total_batch_size
                avg_loss += loss.item()
                avg_mle_loss += mle_loss.item() / args.total_batch_size
                avg_ranking_loss += ranking_loss.item() / args.total_batch_size
                avg_rg_loss += loss_multi[0] / args.total_batch_size
                avg_bs_loss += loss_multi[1] / args.total_batch_size
                avg_ppl_loss += loss_multi[2] / args.total_batch_size

            del similarity, gold_similarity, output, probs
            scaler.scale(loss).backward()
            del loss, mle_loss, ranking_loss
            if step_cnt == args.accumulate_step:
                # updating
                if args.grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
                step_cnt = 0
                epoch_step += 1
                all_step_cnt += 1
                # adjust learning rate
                lr = args.max_lr * min(all_step_cnt ** (-0.5), all_step_cnt * (args.warmup_steps ** (-1.5)))
                for param_group in s_optimizer.param_groups:
                    param_group['lr'] = lr

                scaler.step(s_optimizer)
                scaler.update()
                s_optimizer.zero_grad()
            if all_step_cnt % args.report_freq == 0 and step_cnt == 0 and is_master:
                # report stats
                logger.info("epoch: %d; steps: %d; avg loss: %.6f; avg ranking loss: %.6f; avg mle loss: %.6f; avg RG rank: %.6f; avg BS rank: %.6f; avg PPL rank: %.6f;"
                %(epoch+1, all_step_cnt, avg_loss / args.report_freq, avg_ranking_loss / args.report_freq, avg_mle_loss / args.report_freq, avg_rg_loss / args.report_freq, avg_bs_loss / args.report_freq, avg_ppl_loss / args.report_freq))
                
                with torch.no_grad():
                    if is_mp:
                        c = F.softmax(model.module.model.C.data, dim=0)
                    else:
                        c = F.softmax(model.model.C.data, dim=0)

                logger.info("coef: %.4f; %.4f; %.4f"%tuple(c.tolist()))
                logger.info(f"learning rate: {lr:.6f}")
                avg_mle_loss, avg_ranking_loss, avg_loss = 0, 0, 0
                avg_rg_loss, avg_bs_loss, avg_ppl_loss = 0, 0, 0
            
            if all_step_cnt % args.eval_interval == 0 and all_step_cnt != 0 and step_cnt == 0:
                # evaluate the model as a scorer
                result = test(val_dataloader, val_gen_dataloader, model, args, tok, gpuid, args.do_sample)
                stats = result["stats"]
                stats_res = result["stats_res"]
                total = sum(stats[0])
                stats /= total
                stats_res /= total
                means = [.0]*3
                
                if is_master:
                    logger.info("Total %d"%total)
                    logger.info("Comprehensize Ranking")
                for j in range(3):
                    stat = stats[j]
                    Max_top1[j] = max(Max_top1[j], sum(stat[:1]))
                    Max_top5[j] = max(Max_top5[j], sum(stat[:5]))

                    info_txt = ""
                    tot = 0
                    for k in range(len(stat)):
                        info_txt += "|top@%d: %5.2f|"%(k+1, sum(stat[:k+1])*100)
                        tot += (k+1)*stat[k]
                    Min_mean[j] = min(Min_mean[j], tot)
                    means[j] = tot
                    if is_master:
                        logger.info(ranked_by[j] + info_txt)
                        logger.info(ranked_by[j] + "|Mean Rank: %.4f"%tot)
                
                if is_master:
                    logger.info("Respective Ranking")
                for j in range(3):
                    stat = stats_res[j]
                    info_txt = ""
                    tot = 0
                    for k in range(len(stat)):
                        info_txt += "|top@%d: %5.2f|"%(k+1, sum(stat[:k+1])*100)
                        tot += (k+1)*stat[k]
                    if is_master:
                        logger.info(ranked_by[j] + info_txt)
                        logger.info(ranked_by[j] + "|Mean Rank: %.4f"%tot)

                loss_acc = eval_fn_acc(stats[0][0],stats[1][0],stats[2][0])
                loss = eval_fn(means[0],means[1],means[2])

                if result["rouge1"] > Max_val_R1:
                    Max_val_R1 = result["rouge1"]
                    val_R1 = (result["rouge1"], result["rouge2"], result["rougeLsum"])
                if result["rouge2"] > Max_val_R2:
                    Max_val_R2 = result["rouge2"]
                    val_R2 = (result["rouge1"], result["rouge2"], result["rougeLsum"])
                if result["rougeLsum"] > Max_val_RL:
                    Max_val_RL = result["rougeLsum"]
                    val_RL = (result["rouge1"], result["rouge2"], result["rougeLsum"])

                if loss < minimum_ranking_loss and is_master:
                    minimum_ranking_loss = loss
                    
                    logger.info("best ranking loss - epoch: %d, batch: %d"%(epoch, (i+1) / args.accumulate_step))
                if loss_acc > maximum_accuracy_loss and is_master:
                    maximum_accuracy_loss = loss_acc
                    if is_mp:
                        logger.info("Saving Accuracy Model")
                        _save(model.module.model, os.path.join(args.save_path,"model_accuracy.bin"))
                    else:
                        logger.info("Saving Accuracy Model")
                        _save(model.model, os.path.join(args.save_path,"model_accuracy.bin"))
                    logger.info("best accuracy loss - epoch: %d, batch: %d"%(epoch, (i+1) / args.accumulate_step))
                if is_master:
                    logger.info("val ranking loss: %.6f"%(loss))
                    logger.info("val ranking rouge1: %.6f, rouge2: %.6f, rougeLsum: %.6f"
                    %(result["rouge1"], result["rouge2"], result["rougeLsum"]))
                
                # evaluate the model as a generator
                mle_loss = result["mle_loss"]
                if mle_loss < minimum_mle_loss and is_master:
                    minimum_mle_loss = mle_loss
                    logger.info("best generation loss - epoch: %d, batch: %d"%(epoch, (i+1) / args.accumulate_step))
                if is_master:
                    logger.info("val generation loss: %.6f"%(mle_loss))
                    if args.do_sample:
                        logger.info("val generation rouge1: %.6f, rouge2: %.6f, rougeLsum: %.6f"
                        %(result["sample_rouge1"], result["sample_rouge2"], result["sample_rougeLsum"]))

                # save current model
                if is_master:
                    if is_mp:
                        logger.info("Saving Current Model")
                        _save(model.module.model, os.path.join(args.save_path,"model_cur.bin"))
                    else:
                        logger.info("Saving Current Model")
                        _save(model.model, os.path.join(args.save_path,"model_cur.bin"))
                        

    if is_master:
        logger.info("Best ranking val R1: %.6f, rouge2: %.6f, rougeLsum: %.6f"%val_R1)
        logger.info("Best ranking val R2: %.6f, rouge2: %.6f, rougeLsum: %.6f"%val_R2)
        logger.info("Best ranking val RL: %.6f, rouge2: %.6f, rougeLsum: %.6f"%val_RL)
        logger.info("Best generation val R1: %.6f, rouge2: %.6f, rougeLsum: %.6f"%val_R1_sam)
        logger.info("Best generation val R2: %.6f, rouge2: %.6f, rougeLsum: %.6f"%val_R2_sam)
        logger.info("Best generation val RL: %.6f, rouge2: %.6f, rougeLsum: %.6f"%val_RL_sam)
        for j in range(3):
            logger.info("Best rank based on %s Top1: %.6f; Top5: %.6f; Mean: %.4f"%(ranked_by[j], Max_top1[j],Max_top5[j],Min_mean[j]))

def main(args):
    # set env
    if len(args.gpuid) > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = f'{args.port}'
        mp.spawn(run, args=(args,), nprocs=len(args.gpuid), join=True)
    else:
        run(0, args)

if __name__ ==  "__main__":
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument("--cuda", action="store_true", help="use cuda")
    parser.add_argument("--gpuid", nargs='+', type=int, default=0, help="gpu ids")
    parser.add_argument("-e", "--evaluate", action="store_true", help="evaluate model")
    parser.add_argument("-c", "--test_cur", action="store_true", help="test current model")
    parser.add_argument("-l", "--log", action="store_true", help="logging")
    parser.add_argument("-p", "--port", type=int, default=12355, help="port")
    parser.add_argument("--model_pt", default="", type=str, help="model path")
    parser.add_argument("--config", default="", type=str, help="config path")
    parser.add_argument("--name", default="", type=str)
    parser.add_argument("--save_path", default="checkpoints", type=str)
    parser.add_argument("--logits_merge", default="softmax", type=str)
    parser.add_argument("--trunc_eos", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("--score_smooth", type=float, default=10)
    parser.add_argument("--task_smooth", type=float, default=0.9)
    parser.add_argument("--float_type", type=str,default="bf16")
    parser.add_argument("--test_on_val", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("--amp", type=str2bool, nargs='?',const=True,default=True)

    args = parser.parse_args()
    
    if "trunc_eos" in args.name:
        args.trunc_eos = True
    args.save_path = os.path.join(args.save_path,args.name)
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    if args.cuda is False:
        if args.evaluate:
            evaluation(args)
        else:
            main(args)
    else:
        if args.evaluate:
            with torch.cuda.device(args.gpuid[0]):
                evaluation(args)
        elif len(args.gpuid) == 1:
            with torch.cuda.device(args.gpuid[0]):
                main(args)
        else:
            main(args)
