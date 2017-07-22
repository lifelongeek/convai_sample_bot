# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.agents import Teacher
from parlai.core.dialog_teacher import DialogTeacher
from .build import build

import json
import random
import os

import pdb

class HandwrittenTeacher(Teacher):
    """Hand-written SQuAD teacher, which loads the json squad data and
    implements its own `act()` method for interacting with student agent, rather
    than inheriting from the core Dialog Teacher. This code is here as an
    example of rolling your own without inheritance.
    """

    def __init__(self, opt, shared=None):
        self.datatype = opt['datatype']
        build(opt)
        if opt['datatype'].startswith('train'):
            suffix = 'train'
        else:
            suffix = 'dev'
        datapath = os.path.join(opt['datapath'], 'SQuAD', suffix + '-v1.1.json')
        self.data = self._setup_data(datapath)
        self.episode_idx = -1
        super().__init__(opt, shared)

    def __len__(self):
        return self.len

    # return state/action dict based upon passed state
    def act(self):
        if self.datatype == 'train':
            self.episode_idx = random.randrange(len(self.examples))
        else:
            self.episode_idx = (self.episode_idx + 1) % len(self.examples)
        article_idx, paragraph_idx, qa_idx = self.examples[self.episode_idx]
        article = self.squad[article_idx]
        paragraph = article['paragraphs'][paragraph_idx]
        qa = paragraph['qas'][qa_idx]
        question = qa['question']
        answers = [a['text'] for a in qa['answers']]
        context = paragraph['context']

        if (self.episode_idx == (len(self.examples) - 1) and
            self.datatype != 'train'):
            self.epochDone = True

        return {
            'text': context + '\n' + question,
            'labels': answers,
            'episode_done': True
        }

    def _setup_data(self, path):
        print('loading: ' + path)
        #pdb.set_trace()
        with open(path) as data_file:
            self.squad = json.load(data_file)['data']
        self.len = 0
        self.examples = []
        for article_idx in range(len(self.squad)):
            article = self.squad[article_idx]
            for paragraph_idx in range(len(article['paragraphs'])):
                paragraph = article['paragraphs'][paragraph_idx]
                num_questions = len(paragraph['qas'])
                self.len += num_questions
                for qa_idx in range(num_questions):
                    #pdb.set_trace()
                    self.examples.append((article_idx, paragraph_idx, qa_idx))


class DefaultTeacher(DialogTeacher):
    """This version of SQuAD inherits from the core Dialog Teacher, which just
    requires it to define an iterator over its data `setup_data` in order to
    inherit basic metrics, a default `act` function, and enables
    Hogwild training with shared memory with no extra work.
    For SQuAD, this does not efficiently store the paragraphs in memory.
    """

    def __init__(self, opt, shared=None):
        self.datatype = opt['datatype']
        self.opt = opt
        build(opt)
        if opt['datatype'].startswith('train'):
            suffix = 'train'
        else:
            suffix = 'dev'
        # SQuAD
        #opt['datafile'] = os.path.join(opt['datapath'], 'SQuAD', suffix + '-v1.1.json')

        # MS marco
        opt['datafile'] = os.path.join(opt['datapath'], opt['DB'], suffix + '-v1.1.json')
        self.id = 'squad'
        super().__init__(opt, shared)

    # SQuAD
    """
    def setup_data(self, path):
        print('loading: ' + path)
        #pdb.set_trace()
        with open(path) as data_file:
            self.squad = json.load(data_file)['data']
        for article in self.squad:
            # each paragraph is a context for the attached questions
            for paragraph in article['paragraphs']:
                # each question is an example
                for qa in paragraph['qas']:
                    question = qa['question']
                    answers = (a['text'] for a in qa['answers'])
                    answer_start = [a['answer_start'] for a in qa['answers']]  # for extracting answer_start
                    context = paragraph['context']
                    #pdb.set_trace()
                    if self.opt['datatype']=='train':
                        if self.opt['ans_sent_predict']:
                            yield (context + '\n' + question, answers, answer_start), True  # include answer_start
                        else:
                            yield (context + '\n' + question, answers), True

                    else:  # for making valid predictions.json
                        uid = qa['id']
                        if self.opt['ans_sent_predict']:
                            yield (context + '\n' + question, answers, answer_start, uid), True # include answer_start
                        else:
                            yield (context + '\n' + question, answers, uid), True
    """

    # MS marco
    def setup_data(self, path):
        print('loading: ' + path)
        #pdb.set_trace()
        with open(path) as data_file:
            self.msmarco = json.load(data_file)['data']
        for article in self.msmarco:
            # each paragraph is a context for the attached questions
            question = article['query']
            answers = article['answers']   # What if there is no answer?
            if(len(answers) >= 1):
                answers = answers[0]
            else:   # no answers
                #answers = ''
                continue  # Skip these examples for now


            nParagraph = len(article['passages'])
            query_id = article['query_id']
            if self.opt['msmarco_paragraph_concat']:
                context = ''
                for i in range(nParagraph):
                    context += article['passages'][i]['passage_text'] + ' '

            else:
                selected_passage = -100 # default value
                for i in range(nParagraph):
                    if(article['passages'][i]['is_selected']):
                        selected_passage = i
                        break

                if(selected_passage == -100):  # no passage is selected --> Skip these examples for now
                    continue

                context = article['passages'][selected_passage]['passage_text']

            #pdb.set_trace()

            if self.opt['datatype']=='train':
                yield (context + '\n' + question, answers), True

            else:  # for making valid predictions.json
                yield (context + '\n' + question, answers, query_id), True