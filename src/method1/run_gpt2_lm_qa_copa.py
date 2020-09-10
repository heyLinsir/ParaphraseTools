# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""


import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
import csv
import json
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    BertPreTrainedModel,
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    CamembertConfig,
    CamembertForMaskedLM,
    CamembertTokenizer,
    DistilBertConfig,
    DistilBertForMaskedLM,
    DistilBertTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTConfig,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
)


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


import corrupt

end_token, sep_token = '<|endoftext|>', '<|sepoftext|>'
tokenizer = None

logger = logging.getLogger(__name__)

class NewRobertaForMaskedLM(RobertaForMaskedLM):
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        masked_lm_labels=None,
        num_samples=None,
        log_factors=None
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0] # [batch, seq_length, hidden_size]
        
        mean_state = []
        for _sequence_output, _attention_mask in zip(sequence_output, attention_mask):
            seq_length = int(_attention_mask.sum().item()) - 2
            mean_state.append(_sequence_output[1:seq_length + 1].mean(dim=0))
        mean_state = torch.stack(mean_state, dim=0) # [batch, hidden_size]

        return mean_state

class NewGPT2LMHeadModel(GPT2LMHeadModel):
    def prob_forward(
        self,
        input_ids=None,
        past=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for language modeling.
            Note that the labels **are shifted** inside the model, i.e. you can set ``lm_labels = input_ids``
            Indices are selected in ``[-100, 0, ..., config.vocab_size]``
            All labels set to ``-100`` are ignored (masked), the loss is only
            computed for labels in ``[0, ..., config.vocab_size]``
    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.GPT2Config`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape `(1,)`, `optional`, returned when ``labels`` is provided)
            Language modeling loss.
        prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers` with each tensor of shape :obj:`(2, batch_size, num_heads, sequence_length, embed_size_per_head)`):
            Contains pre-computed hidden-states (key and values in the attention blocks).
            Can be used (see `past` input) to speed up sequential decoding. The token ids which have their past given to this model
            should not be passed as input ids as they have already been computed.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    Examples::
        import torch
        from transformers import GPT2Tokenizer, GPT2LMHeadModel
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=input_ids)
        loss, logits = outputs[:2]
        """
        transformer_outputs = self.transformer(
            input_ids,
            past=past,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        outputs = (lm_logits,) + transformer_outputs[1:]
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')
            loss = []
            for lm_logit, label in zip(lm_logits, labels):
                # Shift so that tokens < n predict n
                shift_logit = lm_logit[:-1, :]
                shift_label = label[1:]
                # Flatten the tokens
                loss.append(loss_fct(shift_logit, shift_label))
            outputs = (torch.stack(loss, dim=0),) + outputs

        return outputs  # (loss), lm_logits, presents, (all hidden_states), (attentions)

    @torch.no_grad()
    def generate(
        self,
        input_ids=None,
        max_length=None,
        min_length=None,
        do_sample=None,
        early_stopping=None,
        num_beams=None,
        temperature=None,
        top_k=None,
        top_p=None,
        repetition_penalty=None,
        bad_words_ids=None,
        bos_token_id=None,
        pad_token_id=None,
        eos_token_id=None,
        length_penalty=None,
        no_repeat_ngram_size=None,
        num_return_sequences=None,
        attention_mask=None,
        decoder_start_token_id=None,
    ):

        # We cannot generate if the model does not have a LM head
        if self.get_output_embeddings() is None:
            raise AttributeError(
                "You tried to generate sequences with a model that does not have a LM Head."
                "Please use another model class (e.g. `OpenAIGPTLMHeadModel`, `XLNetLMHeadModel`, `GPT2LMHeadModel`, `CTRLLMHeadModel`, `T5WithLMHeadModel`, `TransfoXLLMHeadModel`, `XLMWithLMHeadModel`, `BartForConditionalGeneration` )"
            )

        max_length = max_length if max_length is not None else self.config.max_length
        min_length = min_length if min_length is not None else self.config.min_length
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        temperature = temperature if temperature is not None else self.config.temperature
        top_k = top_k if top_k is not None else self.config.top_k
        top_p = top_p if top_p is not None else self.config.top_p
        repetition_penalty = repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
        no_repeat_ngram_size = (
            no_repeat_ngram_size if no_repeat_ngram_size is not None else self.config.no_repeat_ngram_size
        )
        bad_words_ids = bad_words_ids if bad_words_ids is not None else self.config.bad_words_ids
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )
        decoder_start_token_id = (
            decoder_start_token_id if decoder_start_token_id is not None else self.config.decoder_start_token_id
        )

        if input_ids is not None:
            batch_size = input_ids.shape[0]  # overriden by the input batch_size
        else:
            batch_size = 1

        assert isinstance(max_length, int) and max_length > 0, "`max_length` should be a strictly positive integer."
        assert isinstance(min_length, int) and min_length >= 0, "`min_length` should be a positive integer."
        assert isinstance(do_sample, bool), "`do_sample` should be a boolean."
        assert isinstance(early_stopping, bool), "`early_stopping` should be a boolean."
        assert isinstance(num_beams, int) and num_beams > 0, "`num_beams` should be a strictly positive integer."
        assert temperature > 0, "`temperature` should be strictly positive."
        assert isinstance(top_k, int) and top_k >= 0, "`top_k` should be a positive integer."
        assert 0 <= top_p <= 1, "`top_p` should be between 0 and 1."
        assert repetition_penalty >= 0., "`repetition_penalty` should be >= 0."
        assert input_ids is not None or (
            isinstance(bos_token_id, int) and bos_token_id >= 0
        ), "If input_ids is not defined, `bos_token_id` should be a positive integer."
        assert pad_token_id is None or (
            isinstance(pad_token_id, int) and (pad_token_id >= 0)
        ), "`pad_token_id` should be a positive integer."
        assert (eos_token_id is None) or (
            isinstance(eos_token_id, int) and (eos_token_id >= 0)
        ), "`eos_token_id` should be a positive integer."
        assert length_penalty > 0, "`length_penalty` should be strictly positive."
        assert (
            isinstance(no_repeat_ngram_size, int) and no_repeat_ngram_size >= 0
        ), "`no_repeat_ngram_size` should be a positive integer."
        assert (
            isinstance(num_return_sequences, int) and num_return_sequences > 0
        ), "`num_return_sequences` should be a strictly positive integer."
        assert (
            bad_words_ids is None or isinstance(bad_words_ids, list) and isinstance(bad_words_ids[0], list)
        ), "`bad_words_ids` is either `None` or a list of lists of tokens that should not be generated"

        if input_ids is None:
            assert isinstance(bos_token_id, int) and bos_token_id >= 0, (
                "you should either supply a context to complete as `input_ids` input "
                "or a `bos_token_id` (integer >= 0) as a first token to start the generation."
            )
            input_ids = torch.full(
                (batch_size, 1), bos_token_id, dtype=torch.long, device=next(self.parameters()).device,
            )
        else:
            assert input_ids.dim() == 2, "Input prompt should be of shape (batch_size, sequence length)."

        # not allow to duplicate outputs when greedy decoding
        if do_sample is False:
            if num_beams == 1:
                # no_beam_search greedy generation conditions
                assert (
                    num_return_sequences == 1
                ), "Greedy decoding will always produce the same output for num_beams == 1 and num_return_sequences > 1. Please set num_return_sequences = 1"

            else:
                # beam_search greedy generation conditions
                assert (
                    num_beams >= num_return_sequences
                ), "Greedy beam search decoding cannot return more sequences than it has beams. Please set num_beams >= num_return_sequences"

        # create attention mask if necessary
        # TODO (PVP): this should later be handled by the forward fn() in each model in the future see PR 3140
        if (attention_mask is None) and (pad_token_id is not None) and (pad_token_id in input_ids):
            attention_mask = input_ids.ne(pad_token_id).long()
        elif attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)

        # set pad_token_id to eos_token_id if not set. Important that this is done after
        # attention_mask is created
        if pad_token_id is None and eos_token_id is not None:
            logger.warning(
                "Setting `pad_token_id` to {} (first `eos_token_id`) to generate sequence".format(eos_token_id)
            )
            pad_token_id = eos_token_id

        # current position and vocab size
        vocab_size = self.config.vocab_size

        # set effective batch size and effective batch multiplier according to do_sample
        if do_sample:
            effective_batch_size = batch_size * num_return_sequences
            effective_batch_mult = num_return_sequences
        else:
            effective_batch_size = batch_size
            effective_batch_mult = 1

        if self.config.is_encoder_decoder:
            if decoder_start_token_id is None:
                decoder_start_token_id = bos_token_id

            assert (
                decoder_start_token_id is not None
            ), "decoder_start_token_id or bos_token_id has to be defined for encoder-decoder generation"
            assert hasattr(self, "get_encoder"), "{} should have a 'get_encoder' function defined".format(self)
            assert callable(self.get_encoder), "{} should be a method".format(self.get_encoder)

            # get encoder and store encoder outputs
            encoder = self.get_encoder()

            encoder_outputs = encoder(input_ids, attention_mask=attention_mask)

        # Expand input ids if num_beams > 1 or num_return_sequences > 1
        if num_return_sequences > 1 or num_beams > 1:
            input_ids_len = input_ids.shape[-1]
            input_ids = input_ids.unsqueeze(1).expand(batch_size, effective_batch_mult * num_beams, input_ids_len)
            attention_mask = attention_mask.unsqueeze(1).expand(
                batch_size, effective_batch_mult * num_beams, input_ids_len
            )

            input_ids = input_ids.contiguous().view(
                effective_batch_size * num_beams, input_ids_len
            )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)
            attention_mask = attention_mask.contiguous().view(
                effective_batch_size * num_beams, input_ids_len
            )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)

        if self.config.is_encoder_decoder:
            # create empty decoder_input_ids
            input_ids = torch.full(
                (effective_batch_size * num_beams, 1),
                decoder_start_token_id,
                dtype=torch.long,
                device=next(self.parameters()).device,
            )
            cur_len = 1

            assert (
                batch_size == encoder_outputs[0].shape[0]
            ), f"expected encoder_outputs[0] to have 1st dimension bs={batch_size}, got {encoder_outputs[0].shape[0]} "

            # expand batch_idx to assign correct encoder output for expanded input_ids (due to num_beams > 1 and num_return_sequences > 1)
            expanded_batch_idxs = (
                torch.arange(batch_size)
                .view(-1, 1)
                .repeat(1, num_beams * effective_batch_mult)
                .view(-1)
                .to(input_ids.device)
            )
            # expand encoder_outputs
            encoder_outputs = (encoder_outputs[0].index_select(0, expanded_batch_idxs), *encoder_outputs[1:])

        else:
            encoder_outputs = None
            cur_len = input_ids.shape[-1]

        self.prompt_inputs = input_ids.clone().detach()
        if num_beams > 1:
            output = self._generate_beam_search(
                input_ids,
                cur_len=cur_len,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                early_stopping=early_stopping,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                bos_token_id=bos_token_id,
                pad_token_id=pad_token_id,
                decoder_start_token_id=decoder_start_token_id,
                eos_token_id=eos_token_id,
                batch_size=effective_batch_size,
                num_return_sequences=num_return_sequences,
                length_penalty=length_penalty,
                num_beams=num_beams,
                vocab_size=vocab_size,
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
            )
        else:
            output = self._generate_no_beam_search(
                input_ids,
                cur_len=cur_len,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                bos_token_id=bos_token_id,
                pad_token_id=pad_token_id,
                decoder_start_token_id=decoder_start_token_id,
                eos_token_id=eos_token_id,
                batch_size=effective_batch_size,
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
            )

        return output

    def enforce_repetition_penalty_(self, lprobs, batch_size, num_beams, prev_output_tokens, repetition_penalty):
        """repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858). """
        for i in range(batch_size * num_beams):
            for previous_token in set(self.prompt_inputs[i].tolist()):
                # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                if lprobs[i, previous_token] < 0:
                    lprobs[i, previous_token] *= repetition_penalty
                else:
                    lprobs[i, previous_token] /= repetition_penalty

            if repetition_penalty < 1:
                for previous_token in prev_output_tokens[i][len(self.prompt_inputs[i]):].tolist():
                    # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                    if lprobs[i, previous_token] < 0:
                        lprobs[i, previous_token] /= repetition_penalty
                    else:
                        lprobs[i, previous_token] *= repetition_penalty


MODEL_CLASSES = {
    "gpt2": (GPT2Config, NewGPT2LMHeadModel, GPT2Tokenizer),
    "openai-gpt": (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "bert": (BertConfig, BertForMaskedLM, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
    "camembert": (CamembertConfig, CamembertForMaskedLM, CamembertTokenizer),
}


CLS_ID, END_ID, PAD_ID = 50256, 50256, 50256
SENTENCE_BOUNDARIES = ['.', '?', '!']

def split_document_to_sentences(document):
    sentences = []
    sentence = []
    document = [word for word in document.split(' ') if word != '']
    for word in document:
        sentence.append(word)
        if word in SENTENCE_BOUNDARIES:
            sentences.append(' '.join(sentence))
            sentence = []
    if len(sentence) > 0:
        sentences.append(' '.join(sentence))
    return sentences

class TextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, file_path: str, block_size=512):
        assert os.path.isfile(file_path)

        block_size = block_size - (tokenizer.max_len - tokenizer.max_len_single_sentence)

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, args.model_type + "_cached_lm_" + str(block_size) + "_" + filename
        )
        self.cached_features_file = cached_features_file

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.corpus = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)

            self.corpus = []
            with open(file_path, encoding="utf-8") as f:
                sentence = None
                query_type = None
                for text in f:
                    text = text.strip()
                    if text[:3] == '<p>':
                        sentence = text[3:-4].strip()[:-1] + ' ' + query_type
                    elif 'most-plausible-alternative' in text:
                        if 'effect' in text:
                            query_type = 'so'
                        elif 'cause' in text:
                            query_type = 'because'
                        continue
                    else:
                        continue

                    self.corpus.append(sentence)

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.corpus, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, item):
        return self.corpus[item]

    def save_corpus(self):
        logger.info("Saving features into cached file %s", self.cached_features_file)
        with open(self.cached_features_file, "wb") as handle:
            pickle.dump(self.corpus, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_and_cache_examples(args, tokenizer, evaluate=False):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    dataset = TextDataset(tokenizer, args, file_path=file_path, block_size=args.block_size)
    return dataset


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def _rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = _sorted_checkpoints(args, checkpoint_prefix, use_mtime)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)

def collate(examples):
    corrupted_sentences = [corrupt.corrupt(sentence, shuffle_prob=0.2, replace_prob=0.2) for sentence in examples]
    input_ids = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize('%s %s %s %s' % (corrupted_sentence, sep_token, sentence, end_token))) for sentence, corrupted_sentence in zip(examples, corrupted_sentences)]
    len_corrupted_sentences = [len(tokenizer.convert_tokens_to_ids(tokenizer.tokenize('%s %s' % (corrupted_sentence, sep_token)))) for corrupted_sentence in corrupted_sentences]
    labels = [torch.tensor([-100] * len_corrupted_sentence + _input_ids[len_corrupted_sentence:]) for _input_ids, len_corrupted_sentence in zip(input_ids, len_corrupted_sentences)]
    input_ids = [torch.tensor(_input_ids) for _input_ids in input_ids]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=PAD_ID)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    examples = {'input_ids': input_ids.long(), 
            'labels': labels.long()}
    return examples


def train(args, train_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Tuple[int, float]:
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(log_dir=args.output_dir + '-log')

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=collate
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if (
        args.model_name_or_path
        and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))
        and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if args.model_name_or_path and os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0
    tr_pos_loss, logging_pos_loss = 0.0, 0.0
    tr_neg_loss, logging_neg_loss = 0.0, 0.0
    tr_acc, logging_acc = 0.0, 0.0

    model_to_resize = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
    model_to_resize.resize_token_embeddings(len(tokenizer))

    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproducibility
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            batch['input_ids'] = batch['input_ids'].to(args.device)
            batch['labels'] = batch['labels'].to(args.device)
            model.train()
            outputs = model(**batch)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    checkpoint_prefix = "checkpoint"
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, global_step))
                    os.makedirs(output_dir, exist_ok=True)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    _rotate_checkpoints(args, checkpoint_prefix)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prefix="") -> Dict:
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)

    if args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir, exist_ok=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate
    )

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    acc = []
    loss = []
    pos_loss_list = []
    neg_loss_list = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        # for batch in batchs:
        batch['input_ids'] = batch['input_ids'].to(args.device)
        batch['labels'] = batch['labels'].to(args.device)

        with torch.no_grad():
            outputs = model(**batch)
            lm_loss = outputs[0]
            loss.append(lm_loss.item())
        nb_eval_steps += 1

    result = {"loss": np.mean(loss)}


    logger.info("***** Eval results {} *****".format(prefix))
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))
    try:
        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("%s = %s\n" % (key, str(result[key])))
    except:
        pass

    return result


def generate(model, tokenizer, encoded_prompts, num_return_sequences=30, repetition_penalty=1., top_k=40, top_p=1):
    '''
    encoded_prompt: [1, docuemnt_length]
    '''

    new_texts, new_output_sequences = [], []
    for generation_round in range(100):
        encoded_prompt = random.choice(encoded_prompts)
        DQ_text = tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)
        output_sequences = model.generate(
            input_ids=encoded_prompt,
            max_length=15 + len(encoded_prompt[0]),
            temperature=1.,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            num_return_sequences=num_return_sequences,
        )

        # Remove the batch dimension when returning multiple sequences
        if len(output_sequences.shape) > 2:
            output_sequences.squeeze_()

        texts = [tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)[len(DQ_text) : ] for generated_sequence in output_sequences]
        for text, generated_sequence in zip(texts, output_sequences):
            if len(text.strip().split(' ')) < 2:
                continue
            if '\n' in text:
                continue
            elif '<|endoftext|>' in text and '.' in text[: text.find('<|endoftext|>')].strip():
                new_texts.append(text)
                new_output_sequences.append((generated_sequence, DQ_text))
            elif '<|endoftext|>' not in text and '.' in text:
                new_texts.append(text)
                new_output_sequences.append((generated_sequence, DQ_text))
            else:
                continue

        print(len(new_output_sequences))
        if len(new_output_sequences) > 2000:
            break

    generated_sequences = []
    entire_texts = []

    for generated_sequence_idx, (generated_sequence, DQ_text) in enumerate(new_output_sequences):
        generated_sequence = generated_sequence.tolist()

        # Decode text
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

        # Remove all text after the stop token
        text = text[len(DQ_text) : ]

        if '<|endoftext|>' in text:
            explanation_text = text[: text.find('<|endoftext|>')].strip()
        else:
            explanation_text = text.strip()
        # explanation_text = text[: text.find('<|endoftext|>')].strip()
        explanation_text = text[: text.find('.') + 1].strip()
        
        entire_texts.append('%s %s' % (DQ_text, explanation_text))

        generated_sequences.append(explanation_text)

    print('Non-repetitive paraphrase number:', len(set(generated_sequences)))

    return entire_texts, generated_sequences

        

def generate_file(args, model, tokenizer):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)

    # Eval!
    model.eval()

    generated_sequences_list = []

    for sentence in eval_dataset:

        tokenized_prompts = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence)) for i in range(1)]
        with torch.no_grad():
            input_ids = [torch.tensor([tokenized_prompt]).long().to(args.device) for tokenized_prompt in tokenized_prompts]
            entire_texts, generated_sequences = generate(model, tokenizer, input_ids, num_return_sequences=300, repetition_penalty=args.repetition_penalty, top_k=0, top_p=1)

        print(entire_texts[0][:-len(generated_sequences[0])])
        print(generated_sequences[0])
        print('Generated number: ', len(generated_sequences))
        generated_sequences_list.append(generated_sequences)
        print('%d%s' % (len(generated_sequences_list), '=' * 20))

        if random.random() > 0.95:
            pickle.dump(generated_sequences_list, open(args.eval_data_file + '.gpt2large.qa.%.2fpenalty.sample.pkl' % (args.repetition_penalty), 'wb'))

    pickle.dump(generated_sequences_list, open(args.eval_data_file + '.gpt2large.qa.%.2fpenalty.sample.pkl' % (args.repetition_penalty), 'wb'))

def upper_first_word(sentence):
    sentence = [word for word in sentence.split(' ') if word != '']
    sentence = ' '.join(sentence)
    sentence = sentence[0].upper() + sentence[1:]
    return sentence

def lower_first_word(sentence):
    sentence = [word for word in sentence.split(' ') if word != '']
    if sentence[0] != 'I':
        sentence[0] = sentence[0].lower()
    sentence = ' '.join(sentence)
    return sentence

def calc_probs(args, model, tokenizer, prompt, paraphrases, do_upper=True):
    batch = {'input_ids': [],
            'labels': []}
    tokenized_prompt = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(prompt))
    for paraphrase in paraphrases:
        if do_upper:
            paraphrase = upper_first_word(paraphrase)
        tokenized_paraphrase = tokenizer.convert_tokens_to_ids(tokenizer.tokenize('%s %s' % (prompt, paraphrase)))
        batch['input_ids'].append(torch.tensor(tokenized_paraphrase))
        batch['labels'].append(torch.tensor([-100] * len(tokenized_prompt) + tokenized_paraphrase[len(tokenized_prompt):]))
    batch['input_ids'] = pad_sequence(batch['input_ids'], batch_first=True, padding_value=PAD_ID).long().to(args.device)
    batch['labels'] = pad_sequence(batch['labels'], batch_first=True, padding_value=-100).long().to(args.device)

    model.eval()
    with torch.no_grad():
        loss = model.prob_forward(**batch)[0]
        probs = torch.exp(-loss).tolist()

    return probs

def calc_distance(x, y):
    distance = x - y
    return -torch.sqrt((distance * distance).mean(dim=-1))


def calc_cosine(x, y):
    x = x / torch.sqrt((x * x).sum(dim=-1, keepdim=True))
    y = y / torch.sqrt((y * y).sum(dim=-1, keepdim=True))
    return (x * y).sum(dim=-1)

def calc_simlarities(args, model, tokenizer, query, sentences, do_upper=True):
    model.eval()
    with torch.no_grad():
        batch = {'input_ids': [],
                'attention_mask': []}
        for sentence in [query] + sentences:
            if do_upper:
                sentence = upper_first_word(sentence)
            input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize('<s> %s </s>' % sentence))
            batch['input_ids'].append(torch.tensor(input_ids))
            batch['attention_mask'].append(torch.tensor([1] * len(input_ids)))
        batch['input_ids'] = pad_sequence(batch['input_ids'], batch_first=True, padding_value=1).long().to(args.device)
        batch['attention_mask'] = pad_sequence(batch['attention_mask'], batch_first=True, padding_value=0).float().to(args.device)
        states = model(**batch)
        simlarities = calc_cosine(states[0].unsqueeze(dim=0), states[1:])
    return simlarities


def evaluate_console(args, model, lm_model, nli_model, tokenizer, nli_tokenizer):
    model.eval()
    lm_model.eval()
    while True:
        context = input('Please input context:').strip()
        while True:
            option = input('Please input option(EXIT for stop):').strip()
            accumulate_probs = []
            for i in range(10000):
                tokenized_prompts = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(context)) for i in range(1)]
                with torch.no_grad():
                    input_ids = [torch.tensor([tokenized_prompt]).long().to(args.device) for tokenized_prompt in tokenized_prompts]
                    entire_texts, generated_sequences = generate(lm_model, tokenizer, input_ids, num_return_sequences=300, repetition_penalty=1, top_k=0, top_p=1)

                    # probs = calc_probs(args, lm_model, tokenizer, context, ['%s <|endoftext|>' % (generated_sequence) for generated_sequence in generated_sequences], do_upper=False)
                    # probs = torch.tensor(probs).float().to(args.device)

                    # tokenized_context = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(context))
                    # tokenized_options = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize('%s %s <|endoftext|>' % (context, generated_sequence))) for generated_sequence in generated_sequences]

                    # batch = {'input_ids':[],
                    #         'labels': []}
                    # len_context = len(tokenized_context)
                    # for tokenized_option in tokenized_options:
                    #     batch['input_ids'].append(torch.tensor(tokenized_option))
                    #     batch['labels'].append(torch.tensor([-100] * len_context + tokenized_option[len_context:]))
                    # batch['input_ids'] = pad_sequence(batch['input_ids'], batch_first=True, padding_value=PAD_ID).long().to(args.device)
                    # batch['labels'] = pad_sequence(batch['labels'], batch_first=True, padding_value=-100).long().to(args.device)

                    # condition_probs = torch.exp(-lm_model.prob_forward(**batch)[0])
                    # condition_probs = (condition_probs / probs).mean().item()

                    simlarities = calc_simlarities(args, nli_model, nli_tokenizer, option, generated_sequences, do_upper=True)
                    condition_probs = simlarities.tolist()

                    accumulate_probs.extend(condition_probs)

                    print('=========Round %d==============' % (i))
                    print('Current prob:', np.mean(condition_probs))
                    print('Accumulated prob(mean):', np.mean(accumulate_probs))
                    print('Accumulated prob(std):', np.std(accumulate_probs))
                    if (i + 1) % 10 == 0:
                        do_stop = input('Input STOP to stop.')
                        if do_stop == 'STOP':
                            break


# def evaluate_console0(args, model, lm_model, tokenizer):
#     model.eval()
#     lm_model.eval()
#     while True:
#         context = input('Please input context:').strip()
#         while True:
#             option = input('Please input option(EXIT for stop):').strip()
#             accumulate_probs = []
#             for i in range(10000):
#                 tokenized_prompts = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize('%s %s' % (upper_first_word(corrupt.corrupt(option, save_prob=1, shuffle_prob=0, replace_prob=0)), sep_token))) for i in range(1)]
#                 with torch.no_grad():
#                     input_ids = [torch.tensor([tokenized_prompt]).long().to(args.device) for tokenized_prompt in tokenized_prompts]
#                     entire_texts, generated_sequences = generate(model, tokenizer, input_ids, num_return_sequences=300, repetition_penalty=1, top_k=0, top_p=1)

#                     probs = calc_probs(args, model, tokenizer, entire_texts[0][:-len(generated_sequences[0])].strip(), ['%s <|endoftext|>' % (lower_first_word(generated_sequence)) for generated_sequence in generated_sequences], do_upper=False)
#                     probs = torch.tensor(probs).float().to(args.device)

#                     tokenized_context = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(context))
#                     tokenized_options = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize('%s %s <|endoftext|>' % (context, lower_first_word(generated_sequence)))) for generated_sequence in generated_sequences]

#                     batch = {'input_ids':[],
#                             'labels': []}
#                     len_context = len(tokenized_context)
#                     for tokenized_option in tokenized_options:
#                         batch['input_ids'].append(torch.tensor(tokenized_option))
#                         batch['labels'].append(torch.tensor([-100] * len_context + tokenized_option[len_context:]))
#                     batch['input_ids'] = pad_sequence(batch['input_ids'], batch_first=True, padding_value=PAD_ID).long().to(args.device)
#                     batch['labels'] = pad_sequence(batch['labels'], batch_first=True, padding_value=-100).long().to(args.device)

#                     condition_probs = torch.exp(-lm_model.prob_forward(**batch)[0])
#                     condition_probs = (condition_probs / probs).mean().item()

#                     accumulate_probs.append(condition_probs)

#                     print('=========Round %d==============' % (i))
#                     print('Current prob:', condition_probs)
#                     print('Accumulated prob(mean):', np.mean(accumulate_probs))
#                     print('Accumulated prob(std):', np.std(accumulate_probs))


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--train_data_file", default=None, type=str, required=True, help="The input training data file (a text file)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--model_type", type=str, required=True, help="The model architecture to be trained or fine-tuned.",
    )

    # Other parameters
    parser.add_argument(
        "--eval_data_file",
        default=None,
        type=str,
        help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
    )
    parser.add_argument(
        "--line_by_line",
        action="store_true",
        help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
    )
    parser.add_argument(
        "--should_continue", action="store_true", help="Whether to continue from latest checkpoint in output_dir"
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
    )

    parser.add_argument(
        "--mlm", action="store_true", help="Train with masked-language modeling loss instead of language modeling."
    )
    parser.add_argument(
        "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss"
    )

    parser.add_argument(
        "--config_name",
        default=None,
        type=str,
        help="Optional pretrained config name or path if not the same as model_name_or_path. If both are None, initialize a new config.",
    )
    parser.add_argument(
        "--tokenizer_name",
        default=None,
        type=str,
        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path. If both are None, initialize a new tokenizer.",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)",
    )
    parser.add_argument(
        "--block_size",
        default=-1,
        type=int,
        help="Optional input sequence length after tokenization."
        "The training dataset will be truncated in block of this size for training."
        "Default to the model max input length for single sentence inputs (take into account special tokens).",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_eval_console", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_generate", action="store_true", help="Whether to run generation on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=4, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--repetition_penalty", default=1.2, type=float, help="Repetition penalty for generation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=1.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    args = parser.parse_args()

    if args.model_type in ["bert", "roberta", "distilbert", "camembert"] and not args.mlm:
        raise ValueError(
            "BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the --mlm "
            "flag (masked language modeling)."
        )
    if args.eval_data_file is None and args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )
    if args.should_continue:
        sorted_checkpoints = _sorted_checkpoints(args)
        if len(sorted_checkpoints) == 0:
            raise ValueError("Used --should_continue but no checkpoint was found in --output_dir.")
        else:
            args.model_name_or_path = sorted_checkpoints[-1]

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    global tokenizer

    if args.config_name:
        config = config_class.from_pretrained(args.config_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        config = config_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        config = config_class()

    if args.tokenizer_name:
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new {} tokenizer. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name".format(tokenizer_class.__name__)
        )
    if len(tokenizer) == 50257:
        tokenizer.add_special_tokens({'additional_special_tokens': [sep_token]})

    if args.block_size <= 0:
        args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        args.block_size = min(args.block_size, tokenizer.max_len)

    if args.model_name_or_path:
        model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        model = model_class(config=config)

    model.to(args.device)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)

        if args.local_rank == 0:
            torch.distributed.barrier()

        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir, exist_ok=True)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    # Evaluation on console
    results = {}
    if args.do_eval_console and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            
            lm_model = model_class.from_pretrained(
                                        '/home/niuyilin/pre-trained-models/gpt2-large',
                                        from_tf=False,
                                        config=config,
                                    )
            lm_model.to(args.device)
            
            nli_config = RobertaConfig.from_pretrained('/home/niuyilin/pre-trained-models/sentence-robert-large-nli-mean-tokens/0_RoBERTa')
            nli_tokenizer = RobertaTokenizer.from_pretrained('/home/niuyilin/pre-trained-models/sentence-robert-large-nli-mean-tokens/0_RoBERTa')
            nli_model = NewRobertaForMaskedLM.from_pretrained(
                                        '/home/niuyilin/pre-trained-models/sentence-robert-large-nli-mean-tokens/0_RoBERTa',
                                        from_tf=False,
                                        config=nli_config,
                                    )
            nli_model.to(args.device)

            result = evaluate_console(args, model, lm_model, nli_model, tokenizer, nli_tokenizer)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    # Generation
    if args.do_generate and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            generate_file(args, model, tokenizer)

    return results


if __name__ == "__main__":
    main()
