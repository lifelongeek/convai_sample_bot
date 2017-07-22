# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging

from torch.autograd import Variable
from .utils import load_embeddings, AverageMeter
#from .rnn_reader import RnnDocReader


import pdb

logger = logging.getLogger('DrQA')


class DocReaderModel(object):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    def __init__(self, opt, word_dict, char_dict, feature_dict, state_dict=None):

        #Cudnn
        #if not opt['use_cudnn']:
        #    torch.backends.cudnn.enabled=False

        # Book-keeping.
        self.opt = opt
        self.word_dict = word_dict
        self.char_dict = char_dict
        self.feature_dict = feature_dict
        self.updates = 0
        self.train_loss = AverageMeter()
        self.train_loss_QA = AverageMeter()
        self.train_loss_sentpredict = AverageMeter()

       #pdb.set_trace()

        self.input_idx_bdy=5
        self.target_idx_start=5
        if opt['add_char2word']:
            self.input_idx_bdy += 2  # x1_c, x2_c
            self.target_idx_start += 2

        if opt['ans_sent_predict']:
            self.input_idx_bdy += 1  # x1_sent_mask
            self.target_idx_start += 3

        # Building network.
        if opt['net'] == 'rnn_reader':
            from .rnn_reader import RnnDocReader
        elif opt['net'] == 'rnet_qp':
            from .rnet_qp import RnnDocReader
        elif opt['net'] == 'rnet':
            from .rnet import RnnDocReader
        self.network = RnnDocReader(opt)
        
        if state_dict:
            new_state = set(self.network.state_dict().keys())
            for k in list(state_dict['network'].keys()):
                if not k in new_state:
                    del state_dict['network'][k]
            self.network.load_state_dict(state_dict['network'])

        # Building optimizer.
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if opt['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(parameters, opt['learning_rate'],
                                       momentum=opt['momentum'],
                                       weight_decay=opt['weight_decay'])
        elif opt['optimizer'] == 'adamax':
            self.optimizer = optim.Adamax(parameters,
                                          weight_decay=opt['weight_decay'])
        else:
            raise RuntimeError('Unsupported optimizer: %s' % opt['optimizer'])
        
    def set_lrate(self, lrate):
        self.optimizer.param_groups[0]['lr']=lrate      
        
    def set_embeddings(self):
        # Read word embeddings.
        if 'embedding_file' not in self.opt:
            logger.warning('[ WARNING: No embeddings provided. '
                           'Keeping random initialization. ]')
            return
        logger.info('[ Loading pre-trained embeddings ]')
        embeddings = load_embeddings(self.opt, self.word_dict)
        logger.info('[ Num embeddings = %d ]' % embeddings.size(0))

        # Sanity check dimensions
        new_size = embeddings.size()
        old_size = self.network.embedding.weight.size()
        if new_size[1] != old_size[1]:
            raise RuntimeError('Embedding dimensions do not match.')
        if new_size[0] != old_size[0]:
            logger.warning(
                '[ WARNING: Number of embeddings changed (%d->%d) ]' %
                (old_size[0], new_size[0])
            )

        # Swap weights
        self.network.embedding.weight.data = embeddings

        # If partially tuning the embeddings, keep the old values
        if self.opt['tune_partial'] > 0:
            if self.opt['tune_partial'] + 2 < embeddings.size(0):
                fixed_embedding = embeddings[self.opt['tune_partial'] + 2:]
                self.network.fixed_embedding = fixed_embedding

    def update(self, ex):
        # Train mode
        self.network.train()

        #pdb.set_trace()
        # Transfer to GPU
        if self.opt['cuda']:
            #inputs = [Variable(e.cuda(async=True)) for e in ex[:5]]
            #target_s = Variable(ex[5].cuda(async=True))
            #target_e = Variable(ex[6].cuda(async=True))

            inputs = [Variable(e.cuda(async=True)) for e in ex[:self.input_idx_bdy]]
            target_s = Variable(ex[self.target_idx_start].cuda(async=True))
            target_e = Variable(ex[self.target_idx_start+1].cuda(async=True))
        else:
            #inputs = [Variable(e) for e in ex[:5]]
            #target_s = Variable(ex[5])
            #target_e = Variable(ex[6])

            inputs = [Variable(e) for e in ex[:self.input_idx_bdy]]
            target_s = Variable(ex[self.target_idx_start])
            target_e = Variable(ex[self.target_idx_start+1])

        #pdb.set_trace()

        if self.opt['ans_sent_predict']:
            inputs = inputs + [ex[self.input_idx_bdy]]
            target_sent = Variable(torch.from_numpy(np.asarray(ex[self.input_idx_bdy+1])).cuda(async=True))

        # Run forward
        #pdb.set_trace()
        score_list = self.network(*inputs)

        if len(score_list) == 3:
            score_s = score_list[0] 
            score_e = score_list[1]
            score_sent = score_list[2]
        elif len(score_list) == 2:
            score_s = score_list[0]
            score_e = score_list[1]
        elif len(score_list) == 1:
            score_sent = score_list[0]

        # Define computation graph for multi-task learning
        if self.opt['task_QA'] and not self.opt['ans_sent_predict']:
            loss_QA = F.nll_loss(score_s, target_s) + F.nll_loss(score_e, target_e)
            loss = loss_QA
        elif self.opt['ans_sent_predict'] and self.opt['task_QA']:
            loss_QA = F.nll_loss(score_s, target_s) + F.nll_loss(score_e, target_e)
            loss_sent = F.nll_loss(score_sent, target_sent)
            loss = loss_QA + self.opt['coeff_ans_predict']*loss_sent
            self.train_loss_QA.update(loss_QA.data[0], ex[0].size(0))
            self.train_loss_sentpredict.update(loss_sent.data[0], ex[0].size(0))

        # Update
        #pdb.set_trace()
        self.train_loss.update(loss.data[0], ex[0].size(0))

        # Clear gradients and run backward
        self.optimizer.zero_grad()

        #if self.opt['ans_sent_predict']:
            #loss.backward(retain_variables=True)  # since we define multi-task learning graph, we don't have to do retain_variables
        #else:

        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm(self.network.parameters(), self.opt['grad_clipping'])

        # Update parameters
        self.optimizer.step()

        self.updates += 1

        # Reset any partially fixed parameters (e.g. rare words)
        self.reset_parameters()

    def predict(self, ex):
        # Eval mode
        self.network.eval()

        #pdb.set_trace()
        # Transfer to GPU
        if self.opt['cuda']:
            inputs = [Variable(e.cuda(async=True)) for e in ex[:self.input_idx_bdy]]
        else:
            inputs = [Variable(e) for e in ex[:self.input_idx_bdy]]

        # Run forward
        #pdb.set_trace()
        score_list = self.network(*inputs)

        score_s = score_list[0]
        score_e = score_list[1]


        # Transfer to CPU/normal tensors for numpy ops
        score_s = score_s.data.cpu()
        score_e = score_e.data.cpu()

        # Get argmax text spans
        text = ex[-2]
        spans = ex[-1]
        predictions = []
        max_len = self.opt['max_len'] or score_s.size(1)
        for i in range(score_s.size(0)):
            scores = torch.ger(score_s[i], score_e[i])
            scores.triu_().tril_(max_len - 1)
            scores = scores.numpy()
            s_idx, e_idx = np.unravel_index(np.argmax(scores), scores.shape)
            s_offset, e_offset = spans[i][s_idx][0], spans[i][e_idx][1]
            predictions.append(text[i][s_offset:e_offset])

        return predictions

    def reset_parameters(self):
        # Reset fixed embeddings to original value
        if self.opt['tune_partial'] > 0:
            offset = self.opt['tune_partial'] + 2
            if offset < self.network.embedding.weight.data.size(0):
                self.network.embedding.weight.data[offset:] \
                    = self.network.fixed_embedding

    def save(self, filename):
        params = {
            'state_dict': {
                'network': self.network.state_dict(),
            },
            'word_dict': self.word_dict,
            'char_dict': self.char_dict,
            'feature_dict': self.feature_dict,
            'config': self.opt,
        }
        try:
            torch.save(params, filename)
        except BaseException:
            logger.warn('[ WARN: Saving failed... continuing anyway. ]')

    def cuda(self):
        self.network.cuda()
