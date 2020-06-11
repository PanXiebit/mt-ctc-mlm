# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

from fairseq import utils
import torch.nn as nn
from . import FairseqCriterion, register_criterion


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        non_pad_mask = target.ne(ignore_index)
        nll_loss = nll_loss[non_pad_mask]
        smooth_loss = smooth_loss[non_pad_mask]
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion('label_smoothed_cross_entropy_ctc')
class LabelSmoothedCrossEntropyCTCCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.mlm_weights = args.mlm_weights
        self.ctc_weights = args.ctc_weights
        self.ctc_loss = nn.CTCLoss(blank=0, reduction='sum', zero_infinity=True)

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--mlm-weights', default=1., type=float, metavar='D',
                            help='weights of cmlm loss')
        parser.add_argument('--ctc-weights', default=5., type=float, metavar='D',
                            help='weights of ctc loss')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        ctc_decoder_out, cmlm_decoder_out = model(**sample['net_input'])
        loss, mlm_loss, nll_loss, ctc_loss = self.compute_loss(model, ctc_decoder_out, cmlm_decoder_out, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ctc_loss': utils.item(ctc_loss.data) if reduce else ctc_loss.data,
            'ntokens': sample['ntokens'],
            'total_ntokens': sample['total_ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, ctc_decoder_out, cmlm_decoder_out, sample, reduce=True):
        lprobs = cmlm_decoder_out.log_softmax(-1)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = sample["target"].view(-1, 1)
        mlm_loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        
        real_target = sample["real_target"]
        src_lengths = sample["net_input"]["src_lengths"]
        no_real_mask = real_target.ne(self.padding_idx)
        tgt_lengths = no_real_mask.sum(-1)
        ctc_logits = ctc_decoder_out.log_softmax(-1)
        ctc_loss = self.ctc_loss(ctc_logits, real_target, src_lengths, tgt_lengths)
        
        loss = self.ctc_weights * ctc_loss + self.mlm_weights * mlm_loss
        return loss, mlm_loss, nll_loss, ctc_loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        total_ntokens = sum(log.get('total_ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        
        nll_loss = sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2) if ntokens > 0 else 0.
        ctc_loss = sum(log.get('ctc_loss', 0) for log in logging_outputs) / total_ntokens / math.log(2) if ntokens > 0 else 0.
        loss = nll_loss + ctc_loss
        return {
            'loss': loss,
            'ctc_loss' : ctc_loss, 
            'nll_loss': nll_loss,
            'total_ntokens': total_ntokens,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
