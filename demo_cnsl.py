from CC import CC
from DA_CNN import DA_CNN
from QA import QA, get_opt


class Demo(object):
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu
        print('use_gpu = ' + str(self.use_gpu))
        # QA config

        QA_mdl_path = 'model/qa.bst'  # in bot_code
        qa_opt =  get_opt(QA_mdl_path) # Simplify this
        qa_opt['pretrained_model'] = QA_mdl_path
        qa_opt['datatype'] = 'valid'
        qa_opt['embedding_file'] = 'data/glove.840B.300d.txt'

        # DA opt
        da_checkpoint_dir = "model/da_cnn/checkpoints/" # in bot_code

        # Initialize all models
        self.QA = QA(qa_opt, self.use_gpu)
        self.DA = DA_CNN(da_checkpoint_dir, self.use_gpu)
        self.CC = CC(self.use_gpu)


    def run(self, passage, question):
        qa_mode = self.DA.classify_user_query(question, passage)

        if qa_mode:
            #print('qa mode')
            response = self.QA.get_reply(passage, question)
        else:
            #print('cc mode')
            response = self.CC.get_reply('sample_context','sample_reply', question)

        return response
if __name__ == "__main__":
    demo = Demo()

    my_passage = "sample passage"
    my_question = "who are you ?"
    my_answer = demo.run(my_passage, my_question)

    print('passage = ')
    print(my_passage)
    print('question = ')
    print(my_question)
    print('answer = ')
    print(my_answer)
