import sys
import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append("DA_cnn") # for main
#sys.path.append("../DA_cnn") # for bot.sh

import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn

import pdb

class DA_CNN:
    def __init__(self, checkpoint_dir, use_gpu=False):
        print('initialize DA (CNN) module')
        self.use_gpu = use_gpu
        vocab_path = os.path.join(checkpoint_dir, "..", "vocab")
        self.vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
        # pdb.set_trace()
        self.checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
        print('(DA) latest checkpoint_file = ')
        print(self.checkpoint_file)
        self.graph = tf.Graph()

    def classify_user_query(self, query="", passage=""):
        #print('(DA) query = ' + query) # Test PASS
        #print('(DA) passage = ' + passage) # Test PASS
        QA_mode = self.classify_query(query, passage)

        return QA_mode

    def classify_query(self, query, passage):
        with self.graph.as_default():
            if self.use_gpu:
                gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
                self.session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options)
            else:
                self.session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, device_count = {'GPU': 0})

            self.sess = tf.Session(config=self.session_conf)
            #print('4')
            with self.sess.as_default():
                # Transform data
                x_test = np.array(list(self.vocab_processor.transform([query])))

                # Load the saved meta graph and restore variables
                if self.use_gpu:
                    saver = tf.train.import_meta_graph("{}.meta".format(self.checkpoint_file))
                else:
                    saver = tf.train.import_meta_graph("{}.meta".format(self.checkpoint_file) , clear_devices=True)
                saver.restore(self.sess, self.checkpoint_file)


                # Get the placeholders from the graph by name
                input_x = self.graph.get_operation_by_name("input_x").outputs[0]
                # input_y = graph.get_operation_by_name("input_y").outputs[0]
                dropout_keep_prob = self.graph.get_operation_by_name("dropout_keep_prob").outputs[0]

                # Tensors we want to evaluate
                predictions = self.graph.get_operation_by_name("output/predictions").outputs[0]

                # Generate batches for one epoch
                #batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)
                batches = data_helpers.batch_iter(list(x_test), 1, 1, shuffle=False)

                # Collect the predictions here
                all_predictions = []

                for x_test_batch in batches:
                    batch_predictions = self.sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                    all_predictions = np.concatenate([all_predictions, batch_predictions])

        return int(all_predictions[0])

if __name__ == "__main__":
    checkpoint_dir = "../model/da_cnn/checkpoints/" # main()
    da = DA_CNN(checkpoint_dir)

    passage = 'dummy passage'
    # DA case1 : QA
    query = 'what label were the burrell\'s briefly with before their success?'
    print('label for query = ' + query + ' is ' + str(da.classify_query(query, passage)))

    # DA case2 : CC
    query = 'how are you feeling today ?'
    print('label for query = ' + query + ' is ' + str(da.classify_query(query, passage)))

    # DA case3 : CC
    query = 'hello !'
    print('label for query = ' + query + ' is ' + str(da.classify_query(query, passage)))




