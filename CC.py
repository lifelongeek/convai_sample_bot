import re
import chainer
import pickle
import numpy as np

import sys
#sys.path.append('../DSTC/ChatbotBaseline/egs/opensubs/tools') #bot.sh
sys.path.append('DSTC/ChatbotBaseline/egs/opensubs/tools') #main

#import seq2seq_model

import pdb
from chainer import cuda
from nltk.tokenize import casual_tokenize

class CC:
    def __init__(self, use_gpu=False, gpu=0, model_path='model/cc.opensub.bst', maxlen=20, beam=5, penalty=1, nbest=1):
        print('initialize Chitchat module')
        self.use_gpu = use_gpu
        self.num_turn_history = 2  # 2 consecutive turns as history baseline
        #pdb.set_trace()
        if self.use_gpu and gpu >= 0:
            cuda.check_cuda_available()
            cuda.get_device_from_id(gpu).use()
            self.xp = cuda.cupy
        else:
            #chainer.cuda.available = False
            xp = np

        # use chainer in testing mode
        chainer.config.train = False
        #pdb.set_trace()
        # Prepare RNN model and load data
        print('Loading model params from ' + model_path)
        with open(model_path, 'rb') as f:
            #pdb.set_trace()
            self.vocab, self.model, train_args = pickle.load(f)
            #pdb.set_trace()
        if self.use_gpu and gpu >= 0:
            self.model.to_gpu()
        # report data summary
        print('vocabulary size = %d' % len(self.vocab))
        self.vocablist = sorted(self.vocab.keys(), key=lambda s: self.vocab[s])
        # generate sentences
        self.unk = self.vocab['<unk>']
        self.eos = self.vocab['<eos>']
        self.state = None
        self.maxlen = maxlen
        self.beam = beam
        self.penalty = penalty
        self.nbest = nbest


    def get_reply(self, history_context, history_reply, message=""):
        # Note : history_context & history_reply : collections.deque
        #print('cc0')
        history_context_text = ""
        if (len(history_context) >= self.num_turn_history):
            #print(history_context[0])
            #print(history_context[1])
            for i in range(self.num_turn_history):
                #print('history_context')
                #print(len(history_context) + i - self.num_turn_history)
                history_context_text += history_context[len(history_context) + i - self.num_turn_history] + " "

        #print('cc01')
        history_reply_text = ""
        if (len(history_reply) >= self.num_turn_history):
            for i in range(self.num_turn_history):
                history_reply_text += history_reply[len(history_reply) + i - self.num_turn_history] + " "

        # print('(CC) history_context = ' + history_context_text) # Test PASS
        # print('(CC) history_reply = ' + history_reply_text)     # Test PASS
        # print('(CC) message = ' + message)                      # Test PASS

        #print('cc02')
        sentence = []
        for token in casual_tokenize(message, preserve_case=False, reduce_len=True):
            # make a space before apostrophe
            token = re.sub(r'^([a-z]+)\'([a-z]+)$', '\\1 \'\\2', token)
            for w in token.split():
                sentence.append(self.vocab[w] if w in self.vocab else self.unk)

        #print('cc1')
        x_data = np.array(sentence, dtype=np.int32)
        #print('cc2')
        x = chainer.Variable(self.xp.asarray(x_data))
        #print('cc3')
        #self.state exists
        """
        besthyps, self.state = self.model.generate(self.state, x, self.eos, self.eos, unk=self.unk,
                                                   maxlen=self.maxlen,
                                                   beam=self.beam,
                                                   penalty=self.penalty,
                                                   nbest=self.nbest)
        """

        #self.state = None
        besthyps, self.state = self.model.generate(None, x, self.eos, self.eos, unk=self.unk,
                                                   maxlen=self.maxlen,
                                                   beam=self.beam,
                                                   penalty=self.penalty,
                                                   nbest=self.nbest)

        #print('cc4')
        reply = ""

        #print('cc5')
        for w in besthyps[0][0]:
            if w != self.eos:
                reply += self.vocablist[w] + " "

        #print('cc6')
        reply = reply[:-1]

        return reply


if __name__ == "__main__":
    cc = CC()
    print(cc.get_reply([], [], "Hi."))
