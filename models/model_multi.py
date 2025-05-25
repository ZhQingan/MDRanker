import torch
from torch import nn
import torch.nn.functional as F
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
# from modeling_bart import BartScorer, BartScorerClassification
from models.modeling_bart_multi import BartScorer, BartScorerClassification
from models.modeling_pegasus_multi import PegasusScorer
from models.modeling_mbart_multi import MBartScorer
from models.data_utils import METRIC
import numpy as np

def build_point(scs):
    scs_index = list(enumerate(scs))
    return sorted(scs_index, key=lambda x:x[1], reverse=True)

def single_loss(score, cand_scores, bt, pos, neg, hyper):
    ones = torch.ones(1).cuda()
    loss_func = torch.nn.MarginRankingLoss((cand_scores[bt][pos]-cand_scores[bt][neg]) * hyper)
    return loss_func(score[bt:bt+1, pos], score[bt:bt+1, neg], ones)

def simple_rank(score, cand_scores,hyper):
    TotalLoss = torch.tensor(0.0).cuda()
    bsz = score.size(0)
    n = score.size(1)
    for b in range(bsz):
        # TotalLoss += single_loss(score, cand_scores, b, 0, -1)
        for i in range(1,n-1):
            TotalLoss += single_loss(score, cand_scores, b, 0, i,hyper)
            TotalLoss += single_loss(score, cand_scores, b, i, -1,hyper)

    return TotalLoss
    
def full_rank(score, cand_scores,hyper, end_point=True):
    TotalLoss = torch.tensor(0.0).cuda()
    bsz = score.size(0)
    n = score.size(1)
    for b in range(bsz):
        for i in range(n):
            for j in range(i+1,n):
                if i==0 and j==n-1 and end_point==False:
                    continue
                TotalLoss += single_loss(score, cand_scores, b, i, j,hyper)
    return TotalLoss


def RankingLoss_smooth(scores,  summary_score=None, cand_socres=None, margin=0, gold_margin=0, gold_weight=1, no_gold=False, no_cand=False):
    
    n = scores.size(2)
    TotalLoss = torch.tensor(0.0).cuda()
    loss_multi = ()
    span = int(n**0.5)-1
    edge = list(range(0,n,span))
    if not no_cand:
        for i in range(3):
            TmpLoss = torch.tensor(0.0).cuda()
            point_candsc = [build_point(bch) for bch in cand_socres[:,:,i]]
            point = [[_[0] for _ in bch] for bch in point_candsc]
            candsc = np.array([[_[1] for _ in bch] for bch in point_candsc])
            
            for j in range(3):
                task_smooth = 0.9 if i==j else 0.05
                score = torch.gather(scores[j],1,torch.tensor(point).cuda())

                eachLoss = full_rank(score[:, edge], candsc[:, edge], margin)
                if span > 1:
                    for t in range(int((n-1)/span)):
                        eachLoss += simple_rank(score[:, edge[t]:edge[t+1]+1], candsc[:, edge[t]:edge[t+1]+1], margin)
                TmpLoss += eachLoss*task_smooth
            
            TotalLoss += TmpLoss / 3
            loss_multi += (TmpLoss.item(),)
        
    return TotalLoss, loss_multi

def RankingLoss(scores, summary_score=None, cand_socres=None, margin=0, gold_margin=0, gold_weight=1, no_gold=False, no_cand=False):
    
    n = scores.size(2)
    TotalLoss = torch.tensor(0.0).cuda()
    loss_multi = ()
    span = int(n**0.5)-1
    if not no_cand:
        for i in range(3):
            TmpLoss = torch.tensor(0.0).cuda()
            point_candsc = [build_point(bch) for bch in cand_socres[:,:,i]]
            point = [[_[0] for _ in bch] for bch in point_candsc]
            candsc = np.array([[_[1] for _ in bch] for bch in point_candsc])
            score = torch.gather(scores[i],1,torch.tensor(point).cuda())

            if margin < 0:
                margin = score.max()-score.min()
                margin = margin.item()

            edge = list(range(0,n,span))
            TmpLoss = full_rank(score[:, edge], candsc[:, edge], margin)
            if span > 1:
                for t in range(int((n-1)/span)):
                    TmpLoss += simple_rank(score[:, edge[t]:edge[t+1]+1], candsc[:, edge[t]:edge[t+1]+1], margin)
                
            TotalLoss += TmpLoss / 3
            loss_multi += (TmpLoss.item(),)

    return TotalLoss, loss_multi

class BRIO(nn.Module):
    
    def __init__(self, mname, pad_token_id, eos_token_id, trunc_eos, is_pegasus=False, logits_merge=None):
        super(BRIO, self).__init__()
        
        if is_pegasus:
            self.model = PegasusScorer.from_pretrained(mname,local_files_only=True,cache_dir="../double_enc/pretrain_model_cache")
        elif 'mbart' in mname:
            self.model = MBartScorer.from_pretrained(mname,local_files_only=True,)
        else:
            self.model = BartScorer.from_pretrained(mname,local_files_only=True,cache_dir="../double_enc/pretrain_model_cache")
            
        self.model.model.decoder.layers[-1].load_state_dict(
                self.model.model.decoder.layers[-3].state_dict(),
                strict=True
            )
        self.model.model.decoder.layers[-2].load_state_dict(
                self.model.model.decoder.layers[-3].state_dict(),
                strict=True
            )
        if is_pegasus or 'mbart' in mname:
            self.model.model.decoder.layer_norm1.load_state_dict(
                self.model.model.decoder.layer_norm.state_dict(),
                strict=True
            )
            self.model.model.decoder.layer_norm2.load_state_dict(
                self.model.model.decoder.layer_norm.state_dict(),
                strict=True
            )
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.trunc_eos = trunc_eos

        assert logits_merge in ["trainable", "softmax"]
        self.model.logits_merge = logits_merge

        self.model.config.gradient_checkpointing = True

    def forward(self, text_id, candidate_id, score_smooth=1.0, normalize=True, score_mode="log", length_penalty=1, require_gold=True, adding=0):
        
        batch_size = text_id.size(0)
        
        input_mask = text_id != self.pad_token_id
        cand_mask = candidate_id != self.pad_token_id
        cand_mask[:, :, 0] = 1
        outputs = self.model(
            input_ids=text_id, 
            attention_mask=input_mask,
            decoder_input_ids=candidate_id, 
            decoder_attention_mask=cand_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=False,
            )
        
        # Probs
        outputs = outputs[0] # [4, bz x cand_num, seq_len, word_dim]

        if score_smooth < 1.0:
            outputs_tmp = []
            sm = score_smooth
            ism = (1.0-sm)/2
            for i in range(3):
                coef = [ism]*3
                coef[i]=sm

                outputs_tmp.append(outputs[0]*coef[0] + outputs[1]*coef[1] + outputs[2]*coef[2])
            outputs_tmp.append(outputs[-1])
            outputs = torch.stack([output.view(batch_size, -1, output.size(1), output.size(2)) for output in outputs_tmp]) # [4, bz, cand_num, seq_len, word_dim]
        else:
            outputs = torch.stack([output.view(batch_size, -1, output.size(1), output.size(2)) for output in outputs]) # [4, bz, cand_num, seq_len, word_dim]
        
        probs = outputs[3, :, 0]
        probs = torch.log(probs)

        if self.trunc_eos == False:
            outputs = outputs[:, :, :, :-1]  # truncate last token
            candidate_id = candidate_id[:, :, 1:]  # shift right
            cand_mask = candidate_id != self.pad_token_id
        else:
            outputs = outputs[:, :, :, :-2]  # truncate eos and last token
            candidate_id = candidate_id[:, :, 1:-1]  # shift right and truncate eos
            cand_mask = (candidate_id != self.pad_token_id) & (candidate_id != self.eos_token_id)
        
        candidate_id = candidate_id.unsqueeze(-1).repeat(4,1,1,1,1)

        if normalize:
            if score_mode == "log":
                outputs = torch.log(outputs + 1e-6)
        scores = torch.gather(outputs, 4, candidate_id).squeeze(-1)  # [4, bz, cand_num, seq_len]
        cand_mask = cand_mask.float().repeat(4,1,1,1)
        
        scores = torch.mul(scores, cand_mask).sum(-1) / ((cand_mask.sum(-1) + adding) ** length_penalty) # [4, bz, cand_num]
        if require_gold:
            output = {'score': scores[:, :, 1:], "summary_score": scores[:, :, 0], "probs": probs}
        else:
            output = {'score': scores, "probs": probs}
        return output
    
    def scoring_mode(self):
        self.model.model.scoring_mode()

    def generation_mode(self):
        self.model.model.generation_mode()
