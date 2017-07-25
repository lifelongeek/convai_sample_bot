"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import requests
import os
import json
import time
import random
import requests
import collections

CONTEXT_SIZE=5  # currently we don't require much history

class ConvAISampleBot:

    def __init__(self):
        self.history = {}  # chatid, paragraph
        self.ids = []
        self.observation = None


    def observe(self, m):
        print("Observe:")
        cur_chat_id = m['message']['chat']['id']
        cur_chat_text = m['message']['text']
        if cur_chat_id not in self.ids:
            self.ids.append(cur_chat_id)
        #if self.chat_id is None:
            if m['message']['text'].startswith('/start '):
                self.history[cur_chat_id] = {'id' : cur_chat_id, 'paragraph' : m['message']['text'][7:], 'message' : collections.deque(maxlen=CONTEXT_SIZE)}
                #self.history[cur_chat_id]['message'].append(cur_chat_text) # at first turn, message is paragraph. Do not append
                print("\tStart new chat #%s" % cur_chat_id)
                print("with paragraph %s" % self.history[cur_chat_id]['paragraph'])
            else:
                #self.observation = None  # Do we need this ?
                print("\tChat not started yet. Ignore message")
        else:
            #if self.chat_id == m['message']['chat']['id']:
            if m['message']['text'] == '/end':
                self.observation = None
                print("\tEnd chat #%s" % self.chat_id)
                self.chat_id = None
            else:
                self.history[cur_chat_id]['message'].append(cur_chat_text)
                print("\tAccept message as part of chat #%s: %s" % (cur_chat_id, cur_chat_text))
            return cur_chat_text

    #def act(self):
    def act(self, m):
        print("Act:")

        cur_chat_id = m['message']['chat']['id']
        cur_chat_text = m['message']['text']

        if not cur_chat_id in self.ids:
            print("\tChat not started yet. Do not act.")
            return

        if cur_chat_text is None:
            print("\tNo new messages for chat #%s. Do not act." % cur_chat_id)
            return

        query = cur_chat_text
        paragraph  = self.history[cur_chat_id]['paragraph']

        data = {}
        if query == '':
            print("\tDecide to do not respond and wait for new message")
            return
        elif query == '/end':
            print("\tDecide to finish chat %s" % cur_chat_id)
            #self.chat_id = None
            self.ids.remove(cur_chat_id)
            data['text'] = '/end'
            data['evaluation'] = {
                'quality': 0,
                'breadth': 0,
                'engagement': 0
            }
        elif query.startswith('/start '):
            print("\tDecide to skip actting to /start with ID %s" % cur_chat_id)
            return
        else:
            # print("\tDecide to respond with text: %s" % query)
            print("\tGet response for user query : %s" % query)
            response = requests.get(
                'http://0.0.0.0:1990/submit',
                params={'question': query, 'paragraph': paragraph})
            #print(type(response))
            message = {
                'chat_id': cur_chat_id
            }

            data = {
                'text': response.text,
                'evaluation': 0
            }

        message['text'] = json.dumps(data)
        return message


def main():

    """
    !!!!!!! Put your bot id here !!!!!!!
    """
    BOT_ID = 'DA008C35-73CD-4A64-8D67-5C922808D6B4'

    if BOT_ID is None:
        raise Exception('You should enter your bot token/id!')

    # BOT_URL = os.path.join('https://ipavlov.mipt.ru/nipsrouter/', BOT_ID) # NIPS (@ConvaiBot)
    BOT_URL = os.path.join('https://ipavlov.mipt.ru/nipsrouter-alt/', BOT_ID) # alternative (@AltConvaiBot)


    bot = ConvAISampleBot()

    while True:
        try:
            time.sleep(1)
            print("Get updates from server")
            res = requests.get(os.path.join(BOT_URL, 'getUpdates'))

            if res.status_code != 200:
                print(res.text)
                res.raise_for_status()

            print("Got %s new messages" % len(res.json()))
            for m in res.json():
                print("Process message %s" % m)
                bot.observe(m)
                #new_message = bot.act()  # Ver original
                new_message = bot.act(m) # Ver kAb
                if new_message is not None:
                    print("Send response to server.")
                    res = requests.post(os.path.join(BOT_URL, 'sendMessage'),
                                        json=new_message,
                                        headers={'Content-Type': 'application/json'})
                    if res.status_code != 200:
                        print(res.text)
                        res.raise_for_status()
            print("Sleep for 1 sec. before new try")
        except Exception as e:
            print("Exception: {}".format(e))


if __name__ == '__main__':
    main()
