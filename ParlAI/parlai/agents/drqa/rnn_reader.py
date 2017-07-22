# Modified by CNSL
# 1) including TDNN based char embedding
# 06/02/17

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
import torch
import sys
from os import path

import torch.nn as nn
from . import layers
from .tdnn import TDNN
from .highway import Highway
import torch.nn.functional as F


sys.path.append(path.abspath('../../../ptrnet/src'))
from pointer import PointerNetwork
from pointer import LinearTanhSeqAttn

import pdb

class RnnDocReader(nn.Module):
    """Network for the Document Reader module of DrQA."""
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}

    def __init__(self, opt, padding_idx=0, padding_idx_char=0):
        super(RnnDocReader, self).__init__()
        # Store config
        self.opt = opt

        #Cudnn
        #if not opt['use_cudnn']:
        #    torch.backends.cudnn.enabled=False

        # Word embeddings (+1 for padding), usually initialized by GloVE
        self.embedding = nn.Embedding(opt['vocab_size'],
                                      opt['embedding_dim'],
                                      padding_idx=padding_idx)

        # Char embeddings (+1 for padding)
        #pdb.set_trace()
        if opt['add_char2word']:
            self.char_embedding = nn.Embedding(opt['vocab_size_char'],
                                               opt['embedding_dim_char'],
                                               padding_idx=padding_idx_char)

            self.char_embedding.weight = nn.Parameter(torch.Tensor(opt['vocab_size_char'],opt['embedding_dim_char']).uniform_(-1,1))

            self.TDNN = TDNN(opt)

            if opt['nLayer_Highway'] > 0 :
                self.Highway = Highway(opt['embedding_dim'] + opt['embedding_dim_TDNN'], opt['nLayer_Highway'], F.relu)

        # ...(maybe) keep them fixed  (word only)
        if opt['fix_embeddings']:
            for p in self.embedding.parameters():
                p.requires_grad = False

        # Register a buffer to (maybe) fill later for keeping *some* fixed
        if opt['tune_partial'] > 0:
            buffer_size = torch.Size((
                opt['vocab_size'] - opt['tune_partial'] - 2,
                opt['embedding_dim']
            ))
            self.register_buffer('fixed_embedding', torch.Tensor(buffer_size))

        # Projection for attention weighted question
        if opt['use_qemb']:
            if opt['add_char2word'] and (not opt['qemb_with_wordonly']):
                self.qemb_match = layers.SeqAttnMatch(opt['embedding_dim'] + opt['embedding_dim_TDNN'])
            else:
                self.qemb_match = layers.SeqAttnMatch(opt['embedding_dim'])

        # Input size to RNN: word emb + question emb + manual features
        if opt['add_char2word']:
            doc_input_size = opt['embedding_dim'] + opt['num_features'] + opt['embedding_dim_TDNN']
        else:
            doc_input_size = opt['embedding_dim'] + opt['num_features']

        if opt['use_qemb']:
            if opt['add_char2word'] and (not opt['qemb_with_wordonly']):
                doc_input_size += opt['embedding_dim'] + opt['embedding_dim_TDNN']
            else:
                doc_input_size += opt['embedding_dim']
        #pdb.set_trace()

        # RNN document encoder
        self.doc_rnn = layers.StackedBRNN(
            input_size=doc_input_size,
            hidden_size=opt['hidden_size'],
            num_layers=opt['doc_layers'],
            dropout_rate=opt['dropout_rnn'],
            dropout_output=opt['dropout_rnn_output'],
            concat_layers=opt['concat_rnn_layers'],
            rnn_type=self.RNN_TYPES[opt['rnn_type']],
            padding=opt['rnn_padding'],
            )

        # RNN question encoder
        q_input_size = opt['embedding_dim']
        if opt['add_char2word']:
            q_input_size += opt['embedding_dim_TDNN']

        self.question_rnn = layers.StackedBRNN(
            input_size=q_input_size,
            hidden_size=opt['hidden_size'],
            num_layers=opt['question_layers'],
            dropout_rate=opt['dropout_rnn'],
            dropout_output=opt['dropout_rnn_output'],
            concat_layers=opt['concat_rnn_layers'],
            rnn_type=self.RNN_TYPES[opt['rnn_type']],
            padding=opt['rnn_padding'],
            )

        # Output sizes of rnn encoders
        doc_hidden_size = 2 * opt['hidden_size']
        question_hidden_size = 2 * opt['hidden_size']
        if opt['concat_rnn_layers']:
            doc_hidden_size *= opt['doc_layers']
            question_hidden_size *= opt['question_layers']

        # Question merging
        if opt['question_merge'] not in ['avg', 'self_attn']:
            raise NotImplementedError('merge_mode = %s' % opt['merge_mode'])
        if opt['question_merge'] == 'self_attn':
            self.self_attn = LinearTanhSeqAttn(question_hidden_size, question_hidden_size)

        # Bilinear attention for span start/end
        if opt['task_QA']:
            self.start_attn = layers.BilinearSeqAttn(
                doc_hidden_size,
                question_hidden_size,
                )
            self.end_attn = layers.BilinearSeqAttn(
                doc_hidden_size,
                question_hidden_size,
                )

        # Paragraph Hierarchical Encoder
        if opt['ans_sent_predict'] :
            self.meanpoolLayer = layers.Selective_Meanpool(doc_hidden_size)
            self.sentBRNN = layers.StackedBRNN(
                input_size=doc_hidden_size,
                hidden_size=opt['hidden_size_sent'],
                num_layers=opt['nLayer_Sent'],
                concat_layers=False,
                rnn_type=self.RNN_TYPES[opt['rnn_type']],
                padding=opt['rnn_padding_sent'],
            )
            self.sentseqAttn = layers.BilinearSeqAttn(
                opt['hidden_size_sent']*2,
                question_hidden_size,
                )
            #print('DEBUG (no hRNN)')

    #def forward(self, x1, x1_f, x1_mask, x2, x2_mask, x1_c, x1_c_mask, x2_c, x2_c_mask):
    #def forward(self, x1, x1_f, x1_mask, x2, x2_mask, x1_c=None, x2_c=None):  # for this version, we do not utilize mask for char
    def forward(self, x1, x1_f, x1_mask, x2, x2_mask, x1_c=None, x2_c=None, x1_sent_mask=None, word_boundary=None):  # for this version, we do not utilize mask for char

""" PTRNET
#        self.ptrnet = PointerNetwork(doc_hidden_size, doc_hidden_size, 1, decoder_length=2)
"""

    def forward(self, x1, x1_f, x1_mask, x2, x2_mask, decoder_inputs=None):
        """Inputs:
        x1 = document word indices             [batch * len_d]
        x1_f = document word features indices  [batch * len_d * nfeat]
        x1_mask = document padding mask        [batch * len_d] ==>
        x2 = question word indices             [batch * len_q]
        x2_mask = question padding mask        [batch * len_q] ==>
        x1_c = document char indices           [batch * len_d * max_char_per_word]
        x1_c_mask = document char padding mask [batch * len_d * max_char_per_word] --> not implemented in this version
        x2_c = question char indices           [batch * len_q * max_char_per_word]
        x2_c_mask = question char padding mask [batch * len_q * max_char_per_word] --> not implemented in this version
        """
        # Embed both document and question
        batch_size = x1.size()[0]
        doc_len = x1.size()[1]
        ques_len = x2.size()[1]
        x1_emb = self.embedding(x1) # N x Td x D
        x2_emb = self.embedding(x2) # N x Tq x D

        if self.opt['add_char2word']:
            max_wordL_d = x1_c.size()[2]
            max_wordL_q = x2_c.size()[2]
            x1_c = x1_c.view(-1, max_wordL_d)
            x2_c = x2_c.view(-1, max_wordL_q)
            x1_c_emb = self.char_embedding(x1_c)
            x2_c_emb = self.char_embedding(x2_c)
            x1_c_emb = x1_c_emb.view(batch_size,
                                     doc_len,
                                     max_wordL_d,
                                     -1)
            x2_c_emb = x2_c_emb.view(batch_size,
                                     ques_len,
                                     max_wordL_q,
                                     -1)
            # Produce char-aware word embed
            x1_cw_emb = self.TDNN(x1_c_emb)  # N x Td x sum(H)
            x2_cw_emb = self.TDNN(x2_c_emb)  # N x Tq x sum(H)

            # Merge word + char
            x1_emb_combine = torch.cat((x1_emb, x1_cw_emb), 2)
            x2_emb_combine = torch.cat((x2_emb, x2_cw_emb), 2)
            ###x1_mask = torch.cat([x1_mask, x1_c_mask], 2)  # For this version, we do not utilize char mask
            ###x2_mask = torch.cat([x2_mask, x2_c_mask], 2)  # For this version, we do not utilize char mask

            # Highway network
            if self.opt['nLayer_Highway'] > 0:
                #pdb.set_trace()
                [batch_size, seq_len, embed_size] = x1_emb_combine.size()
                x1_emb_combine = self.Highway(x1_emb_combine.view(-1, embed_size))
                x1_emb_combine = x1_emb_combine.view(batch_size, -1, embed_size)

                [batch_size, seq_len, embed_size] = x2_emb_combine.size()
                x2_emb_combine = self.Highway(x2_emb_combine.view(-1, embed_size))
                x2_emb_combine = x2_emb_combine.view(batch_size, -1, embed_size)
        else:
            if (x1_c and x2_c):
                x1_sent_mask = x1_c
                word_boundary = x2_c

        # Dropout on embeddings
        if self.opt['dropout_emb'] > 0:
            x1_emb_combine = nn.functional.dropout(x1_emb_combine, p=self.opt['dropout_emb'], training=self.training)
            x2_emb_combine = nn.functional.dropout(x2_emb_combine, p=self.opt['dropout_emb'], training=self.training)

        # Add attention-weighted question representation
        if self.opt['use_qemb']:
            #pdb.set_trace()
            if self.opt['qemb_with_wordonly']:
                x2_weighted_emb = self.qemb_match(x1_emb, x2_emb, x2_mask)
            else:
                x2_weighted_emb = self.qemb_match(x1_emb_combine, x2_emb_combine, x2_mask)
            drnn_input = torch.cat([x1_emb_combine, x2_weighted_emb, x1_f], 2)
        else:
            drnn_input = torch.cat([x1_emb_combine, x1_f], 2)

        # Encode document with RNN
        doc_hiddens = self.doc_rnn(drnn_input, x1_mask)

        #pdb.set_trace()

        # Encode question with RNN + merge hiddens
        question_hiddens = self.question_rnn(x2_emb_combine, x2_mask)
        if self.opt['question_merge'] == 'avg':
            q_merge_weights = layers.uniform_weights(question_hiddens, x2_mask)
        elif self.opt['question_merge'] == 'self_attn':
            q_merge_weights = self.self_attn(question_hiddens, x2_mask)
        question_hidden = layers.weighted_avg(question_hiddens, q_merge_weights)

        return_list = []
        # Predict start and end positions
        if self.opt['task_QA']:
            start_scores = self.start_attn(doc_hiddens, question_hidden, x1_mask)
            end_scores = self.end_attn(doc_hiddens, question_hidden, x1_mask)
            return_list = return_list + [start_scores, end_scores]

        # Pooling , currently no multi-task learning
        if self.opt['ans_sent_predict']:
            sent_hiddens = self.meanpoolLayer(doc_hiddens, word_boundary)
            if self.opt['nLayer_Sent'] > 0:
                sent_hiddens = self.sentBRNN(sent_hiddens, x1_sent_mask)

            sent_scores = self.sentseqAttn(sent_hiddens, question_hidden, x1_sent_mask)
            return_list = return_list + [sent_scores]

        return return_list
""" PTRNET
        # Create decoder_inputs based on doc_hiddens
        if decoder_inputs is not None:
            decoder_inputs = [doc_hiddens[:, i, :] for i in decoder_inputs]

        # Predict start and end positions
        scores = self.ptrnet(doc_hiddens.transpose(0, 1), decoder_inputs, question_hidden, x1_mask)

        if self.training:
            # In training we output log-softmax for NLL
            start_scores = F.log_softmax(scores[0])
            end_scores = F.log_softmax(scores[1])
        else:
            # ...Otherwise 0-1 probabilities
            start_scores = F.softmax(scores[0])
            end_scores = F.softmax(scores[1])

        # start_scores = self.start_attn(doc_hiddens, question_hidden, x1_mask)
        # end_scores = self.end_attn(doc_hiddens, question_hidden, x1_mask)
        return start_scores, end_scores
"""
