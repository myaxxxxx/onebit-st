# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import random
from dataclasses import dataclass, field

import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterion,
    LabelSmoothedCrossEntropyCriterionConfig, 
)
from fairseq.dataclass import FairseqDataclass
from omegaconf import II
from fairseq import checkpoint_utils, tasks, utils

import torch.nn.functional as F

@dataclass
class SpeechAndTextTranslationCriterionConfig(LabelSmoothedCrossEntropyCriterionConfig):
    mt_finetune: bool = field(
        default=False,
        metadata={"help": "st + mt multi-task finetune"},
    )

@register_criterion(
    "speech_and_text_translation", dataclass=SpeechAndTextTranslationCriterionConfig
)
class SpeechAndTextTranslationCriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
        mt_finetune=False,
    ):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)
        self.mt_finetune = mt_finetune
        
# load ft_model
# add hubert

        FT_MODEL="/workspace/s2t/deltalm_data/save_dir/de/pruning_layer_rate_090_finetuned_layer6_dim1/ft_model.pt"
        self.hubert_model_path = FT_MODEL

        ckpt = checkpoint_utils.load_checkpoint_to_cpu(self.hubert_model_path)
        
        # state = checkpoint_utils.load_checkpoint_to_cpu(self.hubert_model_path)
        # model = state['model']

        # self.ft_model = task.build_model(hubert_args.model)
        # self.ft_model.load_state_dict(ckpt["model"])   
        # model.load_state_dict(state['model'])

        hubert_args = ckpt["cfg"]
        # print(hubert_args.task)
        # print(type(hubert_args.model))
        # print(hubert_args.model)
        # exit()
        task = tasks.setup_task(hubert_args.task)
        if "task_state" in ckpt:
            task.load_state_dict(ckpt["task_state"])

        self.ft_model = task.build_model(hubert_args.model)

        self.ft_model.load_state_dict(ckpt["model"])    
        
        self.ft_model.eval()    


    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            # lprobs: B x T x C
            lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
            target = target[:, self.ignore_prefix_size :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)


    def forward_st(self, model, sample, reduce):
        audio_input = {
            "src_tokens": sample["net_input"]["audio"],
            "src_lengths": sample["net_input"]["audio_lengths"],
            "mode": "st",
            "prev_output_tokens": sample["net_input"]["prev_output_tokens"],
        }
        audio_output = model(**audio_input)
        loss, _ = self.compute_loss(model, audio_output, sample, reduce=reduce)
        return loss ,audio_output
    
    def forward_mt(self, model, sample, reduce):
        text_input = {
            "src_tokens": sample["net_input"]["source"],
            "src_lengths": sample["net_input"]["source_lengths"],
            "mode": "mt",
            "prev_output_tokens": sample["net_input"]["prev_output_tokens"],
        }
        text_output = model(**text_input)
        loss, _ = self.compute_loss(model, text_output, sample, reduce=reduce)
        return loss
    
    def forward_ext_mt(self, model, sample, reduce):
        text_output = model(**sample["net_input"])
        loss, _ = self.compute_loss(model, text_output, sample, reduce=reduce)
        return loss
    
    def foward_teacher_student_st(self, model, sample, audio_output, teacher_audio_output):
        audio_input = {
            "src_tokens": sample["net_input"]["audio"],
            "src_lengths": sample["net_input"]["audio_lengths"],
            "mode": "st",
            "prev_output_tokens": sample["net_input"]["prev_output_tokens"],
        }     
        # self.ft_model.to(model.device)
        # audio_output = model(**audio_input)
        # teacher_audio_output = self.ft_model(**audio_input)
        
        
        # student_lprobs, _ = self.get_lprobs_and_target(model, audio_output, sample)
        # teacher_lprobs, _ = self.get_lprobs_and_target(self.ft_model, teacher_audio_output, sample)
        
        loss = F.mse_loss(audio_output.float(), teacher_audio_output.float(), reduction="mean")
        

        
        return loss.item()

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        st_loss, mt_loss, ext_mt_loss = torch.Tensor([0]).cuda(), torch.Tensor([0]).cuda(), torch.Tensor([0]).cuda()
        st_size, mt_size, ext_mt_size = 0, 0, 0

        mode = sample["net_input"]["mode"]
        if mode == "st":
            if self.mt_finetune and self.training:
                st_loss, audio_output = self.forward_st(model, sample, reduce)
                mt_loss = self.forward_mt(model, sample, reduce)
                
                # add mes loss
                audio_input = {
                    "src_tokens": sample["net_input"]["audio"],
                    "src_lengths": sample["net_input"]["audio_lengths"],
                    "mode": "st",
                    "prev_output_tokens": sample["net_input"]["prev_output_tokens"],
                }     
                # self.ft_model.to(model.device)
                # audio_output = model(**audio_input)
                # self.ft_model.to(st_loss.device)
                teacher_audio_output = self.ft_model(**audio_input)[0]       
            
                mse_loss = self.foward_teacher_student_st(model, sample, audio_output[0], teacher_audio_output)
                
  
                
                loss = st_loss + mt_loss + mse_loss
                st_size = mt_size = sample_size = sample["ntokens"]
            else:
                loss = st_loss = self.forward_st(model, sample, reduce)
                st_size = sample_size = sample["ntokens"]
        elif mode == "ext_mt":
            loss = ext_mt_loss = self.forward_ext_mt(model, sample, reduce)
            ext_mt_size = sample_size = sample["ntokens"]

        logging_output = {
            "loss": loss.data,
            "st_loss": st_loss.data,
            "st_sample_size": st_size,
            "mt_loss": mt_loss.data,
            "mt_sample_size": mt_size,
            "ext_mt_loss": ext_mt_loss.data,
            "ext_mt_sample_size": ext_mt_size,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "mse_loss": mse_loss
        }
        
        return loss, sample_size, logging_output

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        st_loss_sum = sum(log.get("st_loss", 0) for log in logging_outputs)
        mt_loss_sum = sum(log.get("mt_loss", 0) for log in logging_outputs)
        mse_loss_sum = sum(log.get("mse_loss", 0) for log in logging_outputs)
        ext_mt_loss_sum = sum(log.get("ext_mt_loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        st_sample_size = sum(log.get("st_sample_size", 0) for log in logging_outputs)
        mt_sample_size = sum(log.get("mt_sample_size", 0) for log in logging_outputs)
        ext_mt_sample_size = sum(log.get("ext_mt_sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "st_loss", st_loss_sum / st_sample_size / math.log(2) if st_sample_size != 0 else 0, st_sample_size, round=3
        )
        metrics.log_scalar(
            "mt_loss", mt_loss_sum / mt_sample_size / math.log(2) if mt_sample_size != 0 else 0, mt_sample_size, round=3
        )
        metrics.log_scalar(
            "mse_loss", mse_loss_sum / st_sample_size / math.log(2) if st_sample_size != 0 else 0, st_sample_size, round=3
        )
        metrics.log_scalar(
            "ext_mt_loss", ext_mt_loss_sum / ext_mt_sample_size / math.log(2) if ext_mt_sample_size != 0 else 0, ext_mt_sample_size, round=3
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True