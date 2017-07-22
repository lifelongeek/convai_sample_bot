from flask import Flask, render_template, redirect, request, jsonify
#from squad.demo_prepro import prepro
from demo_cnsl import Demo
import json
import requests

import argparse

parser = argparse.ArgumentParser(description='launch https server for chatbot, team kAIb')
parser.add_argument('--use_gpu', type=bool, help='Use gpu or not', default=False)
args = parser.parse_args()

app = Flask(__name__)
demo = Demo(args.use_gpu)

@app.route('/')
def main():
    return render_template('index.html')


def getAnswer(paragraph, question):
    return demo.run(paragraph, question)


@app.route('/submit', methods=['GET', 'POST'])
def submit():
    paragraph = request.args.get('paragraph')
    question = request.args.get('question')
    answer = getAnswer(paragraph, question)
    return answer

if __name__ == "__main__":
    app.run(host="0.0.0.0", port="1990", threaded=True)
