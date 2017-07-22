try:
    import torch
except ModuleNotFoundError:
    raise ModuleNotFoundError('Need to install pytorch: go to pytorch.org')

import sys
sys.path.append("ParlAI")

import pdb

from parlai.agents.drqa_msmarco.agents import DocReaderAgent

class QA:
    def __init__(self, opt, use_gpu=False):
        print('initialize QA module')
        self.use_gpu = use_gpu
        # Check options
        assert('pretrained_model' in opt)
        assert(opt['datatype'] == 'valid') # SQuAD only have valid data

        #pdb.set_trace()
        if self.use_gpu:
            opt['cuda']=True
        else:
            opt['cuda']=False
        # Load document reader
        self.doc_reader = DocReaderAgent(opt)
        self.doc_reader.model.network.eval()

    def get_reply(self, passage="", question=""):
        #print('(QA) passage = ' + passage)   # Test PASS
        #print('(QA) question = ' + question) # Test PASS

        reply = self.doc_reader.QA_single(passage, question)

        return reply[0]

# Config
def get_opt(pretrained_model_path, use_gpu=False):
    if use_gpu:
        mdl = torch.load(pretrained_model_path)
    else:
        #pdb.set_trace()
        mdl = torch.load(pretrained_model_path, map_location=lambda storage, loc: storage)
    opt = mdl['config']
    del mdl

    return opt


if __name__ == "__main__":
    # Ver1
    #pretrained_mdl_path = '../../ParlAI/exp-squad/qp-pp-basic'  # qp-pp-basic

    # Ver2
    pretrained_mdl_path = '/data3/kenkim/ParlAI_DEBUG/exp_release-squad/exph14-bt-gt-rt-if'  # release ver

    opt =  get_opt(pretrained_mdl_path)
    opt['pretrained_model'] = pretrained_mdl_path
    opt['datatype'] = 'valid'
    opt['embedding_file'] = '../../ParlAI/data/glove.840B.300d.txt'

    # Temporary options?
    #opt['pp_gate'] = False  # for ver mismatch? temporary?
    #opt['pp_rnn'] = True    # for ver mismatch? temporary?

    #pdb.set_trace()
    qa = QA(opt)


    # Example1 (in train)
    passage_sample = "The College of Engineering was established in 1920, however, early courses in civil and mechanical engineering were a part of the College of Science since the 1870s. Today the college, housed in the Fitzpatrick, Cushing, and Stinson-Remick Halls of Engineering, includes five departments of study \u2013 aerospace and mechanical engineering, chemical and biomolecular engineering, civil engineering and geological sciences, computer science and engineering, and electrical engineering \u2013 with eight B.S. degrees offered. Additionally, the college offers five-year dual degree programs with the Colleges of Arts and Letters and of Business awarding additional B.A. and Master of Business Administration (MBA) degrees, respectively."
    question_sample = "How many BS level degrees are offered in the College of Engineering at Notre Dame?"

    # Example2 (in train)
    #passage_sample = "The College of Engineering was established in 1920, however, early courses in civil and mechanical engineering were a part of the College of Science since the 1870s. Today the college, housed in the Fitzpatrick, Cushing, and Stinson-Remick Halls of Engineering, includes five departments of study \u2013 aerospace and mechanical engineering, chemical and biomolecular engineering, civil engineering and geological sciences, computer science and engineering, and electrical engineering \u2013 with eight B.S. degrees offered. Additionally, the college offers five-year dual degree programs with the Colleges of Arts and Letters and of Business awarding additional B.A. and Master of Business Administration (MBA) degrees, respectively."
    #question_sample = "What is the oldest structure at Notre Dame?"

    #print('length of passage = ')
    #print(len(passage_sample))

    #pdb.set_trace()
    print(qa.get_reply(passage_sample, question_sample))