
import string
from data_helpers import clean_str
import json

import pdb
from nltk import word_tokenize 


# Read a file and split into lines
def readLines(filename):
    with open(filename) as data_file:
        read_dat = json.load(data_file, encoding = 'utf-8')['data']
        passages = []
        questions = []
        match = []
        article_idx = 0
        pssg_idx = 0
        
        for article in read_dat:
            # each paragraph is a context for the attached questions
            for paragraph in article['paragraphs']:
                passages.append(clean_str(paragraph['context']))
                    
                # each question is an example
                for qa in paragraph['qas']:
                    questions.append(clean_str(qa['question']))
                    match.append([article_idx, pssg_idx])
                    
                pssg_idx = pssg_idx + 1
            article_idx = article_idx + 1

    return passages, questions, match
import re

def write_text(filename, dict):
    print(filename)
    writer = open(filename, 'w')
    for sent in dict:
        writer.write(sent + "\n")    
    writer.close()

def write_text_noqmark(filename, dict):
    print(filename)
    writer = open(filename, 'w')
    for sent in dict:
        writer.write(re.sub(r"\\\?", "", sent).strip() + "\n")
    writer.close()

def squad_preprocessing():
    dataset = ['train', 'dev']
    for data in dataset:
        raw_file = '/data2/hwaranlee/convai/ParlAI-v2/data/SQuAD/' + data + '-v1.1.json'
        processed_q = '/data2/hwaranlee/convai/DA_cnn/data/squad_' + data + '_q.txt'
        processed_p = '/data2/hwaranlee/convai/DA_cnn/data/squad_' + data + '_p.txt'
        processed_no_qmark = '/data2/hwaranlee/convai/DA_cnn/data/squad_' + data + '_no_qmark.txt'
        processed_idx = '/data2/hwaranlee/convai/DA_cnn/data/squad_' + data + '_idx.json'
        
        passages, questions, match = readLines(raw_file)
        
        write_text(processed_p, passages)
        write_text(processed_q, questions)
        write_text_noqmark(processed_no_qmark, questions)
        
        with open(processed_idx, 'w') as outfile:
            json.dump(match, outfile)
    
def logs_preprocessing():
    dataset = ['cc', 'qa']
    for data in dataset:
        raw_data = '/data2/hwaranlee/convai/DA_cnn/data/log_' + data + '.txt'
        processed = '/data2/hwaranlee/convai/DA_cnn/data/log_processed_' + data + '.txt'
        processed_no_qmark = '/data2/hwaranlee/convai/DA_cnn/data/log_processed_' + data + '_no_qmark.txt'
        
        print(raw_data)
        questions = []
        f = open(raw_data, 'r')        
        lines = f.readlines()
        for line in lines:
            questions.append(clean_str(line))
        f.close()

        write_text(processed, questions)
        write_text_noqmark(processed_no_qmark, questions)        
        
def opensubs_preprocessing():
    dataset = ['dev', 'eval', 'train']
    for data in dataset:
        raw_data = '/data2/hwaranlee/convai/DA_cnn/data/opensubs_trial_data_' + data + '.txt'
        processed = '/data2/hwaranlee/convai/DA_cnn/data/opensubs_trial_data_processed_' + data + '.txt'
        processed_no_qmark = '/data2/hwaranlee/convai/DA_cnn/data/opensubs_trial_data_processed_' + data + '_no_qmark.txt'
        
        print(raw_data)
        questions = []
        f = open(raw_data, 'r')        
        lines = f.readlines()
        for line in lines:
            if line.find("U:") == 0:
                questions.append(clean_str(line[3:len(line)]))
        f.close()

        write_text(processed, questions)
        write_text_noqmark(processed_no_qmark, questions)
        print('# sentences : ' + str(len(questions)))
        
def opensubs_prep_qdsep(): 
    # Q/D seperation
    # Both U/S are used
    dataset = ['train'] # [train', 'dev', 'eval']
    for data in dataset:
        raw_data = '/data2/hwaranlee/convai/DA_cnn/data/opensubs_trial_data_' + data + '.txt'
        processed_p = '/data2/hwaranlee/convai/DA_cnn/data/opensubs_trial_data_processed_' + data + '_q.txt' # question
        processed_d = '/data2/hwaranlee/convai/DA_cnn/data/opensubs_trial_data_processed_' + data + '_d.txt' # declarative
        
        print(raw_data)
        questions = []
        declarative = []

        f = open(raw_data, 'r')        
        lines = f.readlines()
        for line in lines:
            if len(line) > 1 : # except "\n" 
                if line.find("?") == -1:
                    declarative.append(clean_str(line[3:len(line)]))
                else:
                    questions.append(clean_str(line[3:len(line)]))
        f.close()

        write_text(processed_p, questions)
        write_text(processed_d, declarative)
        print('# sentences : ' + str(len(questions)))                
        print('# sentences : ' + str(len(declarative)))                


if __name__ == '__main__':
    
    #squad_preprocessing()
    #logs_preprocessing()
    #opensubs_preprocessing()
    opensubs_prep_qdsep()










