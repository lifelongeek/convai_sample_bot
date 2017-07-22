# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Trains (a partial) implementation of the DrQa Document Reader from:

Danqi Chen, Adam Fisch, Jason Weston, Antoine Bordes. 2017.
Reading Wikipedia to Answer Open-Domain Questions.
In Association for Computational Linguistics (ACL).

Link: https://arxiv.org/abs/1704.00051

Note:
To use pretrained word embeddings, set the --embeddings_file path argument.
GloVe is recommended, see http://nlp.stanford.edu/data/glove.840B.300d.zip.
"""
try:
    import torch
except ModuleNotFoundError:
    raise ModuleNotFoundError('Need to install pytorch: go to pytorch.org')
import numpy as np
import logging
import copy
import sys
import random

import os.path
import pickle

from parlai.agents.drqa.agents import SimpleDictionaryAgent
from parlai.agents.drqa.agents import SimpleCharDictionaryAgent
from parlai.agents.drqa.agents import DocReaderAgent
from parlai.agents.drqa.utils import Timer
from parlai.core.worlds import DialogPartnerWorld
from parlai.core.params import ParlaiParser
from parlai.core.worlds import create_task

import pdb
import gc # garbage collector

def build_dict(opt):
    opt = copy.deepcopy(opt)
    opt['batchsize'] = 1
    dictionary = SimpleDictionaryAgent(opt)

    # We use the train set to build the dictionary.
    logger.info('[ Building word dictionary... ]')
    opt['datatype'] = 'train:ordered'
    world = create_task(opt, dictionary)
    for _ in world:
        world.parley()

    #dictionary.sort()
    nKeep=opt['vocab_size']
    dictionary.sort_and_keep(nKeep)
    logger.info('[ Dictionary built. ]')
    logger.info('[ Num words = %d ]' % len(dictionary))
    return dictionary

def build_dict_char(opt):
    opt = copy.deepcopy(opt)
    opt['batchsize'] = 1
    dictionary = SimpleCharDictionaryAgent(opt)

    # We use the train set to build the dictionary.
    logger.info('[ Building character dictionary... ]')
    opt['datatype'] = 'train:ordered'
    world = create_task(opt, dictionary)
    for _ in world:
        world.parley()

    nKeep=opt['vocab_size_char']
    dictionary.sort_and_keep(nKeep, True)
    logger.info('[ Dictionary built. ]')
    logger.info('[ Num chars = %d ]' % len(dictionary))
    return dictionary



def validate(opt, agent, n_iter):
    opt = copy.deepcopy(opt)
    opt['datatype'] = 'valid'
    valid_world = create_task(opt, agent)

    logger.info('[ Running validation... ]')

    # Sent prediction
    valid_world.world.agents[1].opt['ans_sent_predict'] = False
    valid_world.world.agents[1].model.network.opt['ans_sent_predict'] = False  # disable sentence predicction by default
    if opt['ans_sent_predict']:
        valid_world.world.agents[1].model.input_idx_bdy -= 1


    valid_time = Timer()

    # pdb.set_trace()
    for _ in valid_world:
        #pdb.set_trace()
        valid_world.parley()


    metrics = valid_world.report()
    if 'tasks' in metrics:
        for task, t_metrics in metrics['tasks'].items():
            logger.info('[valid] task = %s | iter = %d | exs = %d | ' %
                        (task, n_iter, t_metrics['total']) +
                        'EM = %.4f | F1 = %.4f' %
                        (t_metrics['accuracy'], t_metrics['f1']))
        logger.info('[valid] iter = %d | overall EM = %.4f | exs = %d' %
                    (n_iter, metrics['accuracy'], metrics['total']))
    else:
        logger.info(
            '[valid] iter = %d | EM = %.4f | F1 = %.4f | exs = %d' %
            (n_iter, metrics['accuracy'], metrics['f1'], metrics['total'])
        )
    logger.info('[ Done. Time = %.2f (s) ]' % valid_time.time())

    valid_world.world.agents[1].opt['ans_sent_predict'] = opt['ans_sent_predict'] # recover
    valid_world.world.agents[1].model.network.opt['ans_sent_predict'] = opt['ans_sent_predict']  # recover
    if opt['ans_sent_predict']:
        valid_world.world.agents[1].model.input_idx_bdy += 1

    return metrics[opt['valid_metric']]


def main(opt):
    #iter_global = 0
    #pdb.set_trace()
    # Cudnn

    # Build word dictionary from task data
    if os.path.isfile(("data/SQuAD/dict.word." + str(opt['vocab_size']) + ".pkl")):
        dictionary = pickle.load( open( ("data/SQuAD/dict.word." + str(opt['vocab_size']) + ".pkl"), "rb") )       # word dictionary
        logger.info('successfully load word dictionary')
    else:
        if 'pretrained_model' in opt:
            dictionary = None
        else:
            dictionary = build_dict(opt)
        pickle.dump( dictionary , open( ("data/SQuAD/dict.word." + str(opt['vocab_size']) + ".pkl"), "wb") )

    dictionary_char=None
    if opt['add_char2word']:
        opt['NULLWORD_Idx_in_char'] = opt['vocab_size_char']-1
        if os.path.isfile(("data/SQuAD/dict.char." + str(opt['vocab_size_char']) + ".pkl")):
            dictionary_char = pickle.load( open( ("data/SQuAD/dict.char." + str(opt['vocab_size_char']) + ".pkl"), "rb") )  # char dictionary
            logger.info('successfully load char dictionary')
        else:
            # Build char dictionary from task data
            dictionary_char = build_dict_char(opt)
            pickle.dump( dictionary_char , open( ("data/SQuAD/dict.char." + str(opt['vocab_size_char']) + ".pkl"), "wb") )

        # Figure out max word len

        # figuring out max_word_len from word dictionary is not valid choice ==> which is 25
        #opt['max_word_len'] = -100 # initialize
        #for i in range(len(dictionary)):
    #        cur_word_len = len(dictionary[i])
    #        if opt['max_word_len'] < cur_word_len:
    #            opt['max_word_len'] = cur_word_len

        # just set as hyperparameter in config.py
        logger.info('maximum word len = %d' % (opt['max_word_len']))

        # Calculate TDNN embedding dim (after applying TDNN to char tensor)
        opt['kernels'] = ''.join(opt['kernels'])
        if isinstance(opt['kernels'], str):
               opt['kernels'] = eval(opt['kernels']) # convert string list of tuple --> list of tuple
        opt['embedding_dim_TDNN']=0
        for i, n in enumerate(opt['kernels']):
            opt['embedding_dim_TDNN'] += n[1]

        logger.info('TDNN embedding dim = %d' % (opt['embedding_dim_TDNN']))

    #pdb.set_trace()

    # Build document reader
    doc_reader = DocReaderAgent(opt, word_dict=dictionary, char_dict=dictionary_char)

    # Log params
    logger.info('[ Created with options: ] %s' %
                ''.join(['\n{}\t{}'.format(k, v)
                         for k, v in doc_reader.opt.items()]))

    # Build training world once
    opt['datatype'] = 'train'
    train_world = create_task(opt, doc_reader)
    train_time = Timer()

    # Keep track of best model + how long since the last improvement
    best_valid = 0
    impatience = 0
    lrate_decay = 0

    logger.info("[ Ok, let's go... ]")
    iteration = 0
    while impatience < opt['patience']:
        
        # Train...
        logger.info('[ Training for %d iters... ]' % opt['train_interval'])
        train_time.reset()
        iter = 0
        for _ in range(opt['train_interval']):
            iter+=1
            train_world.parley()
            #if iter % opt['collect_garbage_every'] == 0:
            if iter % 100 == 0:
                gc.collect()

        # ...validate!
        print('start validation')
        valid_metric = validate(opt, doc_reader, iteration)
        if valid_metric > best_valid:
        #if False:
            logger.info(
                '[ Best eval %d: %s = %.4f (old = %.4f) ]' %
                (iteration, opt['valid_metric'], valid_metric, best_valid)
            )
            best_valid = valid_metric
            impatience = 0
            if 'model_file' in opt:
                doc_reader.save(opt['model_file'])

            if valid_metric == 1:
                logger.info('[ Task solved! Stopping. ]')
                break
        else:
            if opt['lrate_decay']:
                # doc_reader.model.opt['learning_rate'] *= 0.5
                opt['learning_rate'] *= opt['lrate_decay_factor']
                doc_reader.model.set_lrate(opt['learning_rate'])
                logger.info('[ Decrease learning_rate %.2e]' % opt['learning_rate'] )
                lrate_decay +=1
                if lrate_decay > 10:
                    break
            else:                
                impatience += 1
                logger.info('[ Increase impatience %d ]' % impatience)

        iteration += 1
    logger.info('[ >> Best eval : %s = %.4f ]' % (opt['valid_metric'], best_valid))


if __name__ == '__main__':
    # Get command line arguments
    argparser = ParlaiParser()
    argparser.add_arg(
        '--train_interval', type=int, default=1000,
        help='Validate after every N train updates',
    )
    argparser.add_arg(
        '--patience', type=int, default=16,
        help='Number of intervals to continue without improvement'
    )
    SimpleDictionaryAgent.add_cmdline_args(argparser)
    DocReaderAgent.add_cmdline_args(argparser)
    opt = argparser.parse_args()

    # Set logging
    logger = logging.getLogger('DrQA')
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: %(message)s', '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    if 'log_file' in opt:
        logfile = logging.FileHandler(opt['log_file'], 'w')
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)
    logger.info('[ COMMAND: %s ]' % ' '.join(sys.argv))

    # Set cuda
    opt['cuda'] = not opt['no_cuda'] and torch.cuda.is_available()
    if opt['cuda']:
        logger.info('[ Using CUDA (GPU %d) ]' % opt['gpu'])
        torch.cuda.set_device(opt['gpu'])

    # Set random state
    np.random.seed(opt['random_seed'])
    random.seed(opt['random_seed'])
    torch.manual_seed(opt['random_seed'])
    if opt['cuda']:
        torch.cuda.manual_seed(opt['random_seed'])

    # Run!
    main(opt)
