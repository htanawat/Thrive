from flask import Flask, Response, render_template, make_response, jsonify, request as req
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import json
import boto3
from datetime import datetime
import requests
import speech_recognition as sr
from scipy.io import wavfile
import os

from google.auth import compute_engine
from google.oauth2 import service_account
from google.auth.transport.requests import AuthorizedSession

import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from random import choice

def get_pred(text, model, tok, p=0.7):
    input_ids = torch.tensor(tok.encode(text)).unsqueeze(0)
    logits = model(input_ids)[0][:, -1]
    probs = F.softmax(logits, dim=-1).squeeze()
    idxs = torch.argsort(probs, descending=True)
    res, cumsum = [], 0.
    for idx in idxs:
        res.append(idx)
        cumsum += probs[idx]
        if cumsum > p:
            pred_idx = idxs.new_tensor([choice(res)])
            break
    pred = tok.convert_ids_to_tokens(int(pred_idx))
    return tok.convert_tokens_to_string(pred)

tok = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

SCOPES = ['https://www.googleapis.com/auth/application-default']
data = json.load(open("whdcred.json"))
credentials = service_account.Credentials.from_service_account_info(data, scopes=["https://www.googleapis.com/auth/cloud-platform", "https://www.googleapis.com/auth/cloud-language"])
authed_session = AuthorizedSession(credentials)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "./upload"

@app.route('/')
def main():
    return render_template('main.html', name="main")

GOOGLE_SPEECH_API_KEY = "ya29.c.Ko8BzQdRYK12keOUD-reXlUKS4BS9aIgPonRmB0C6M-8qz5Frhrb-pVZftmPZlgPtyh9uvzVIPQjU0qHbLGJtRK6C4D3ymD1CKqW0d0V5gNpY5dNs00MU-RQFTAMZz0nRlf8MM3ve2fvbuCblgOLYmtRaRKJXUdmXiIV6_8NcuYGJsCIYRIkYKJZgnzxfCOIb8"

def generate_text(n, start):
    generated_text = start
    for i in range(n):
        pred = get_pred(generated_text, model, tok)
        generated_text += pred
    return generated_text

# @app.route('/preview', methods=["POST"])
# def preview():
#     global authed_session
#     vfile = req.files["voice"]
#     voice = vfile.read()
#     vcodes = base64.b64encode(voice)

@app.route('/generates', methods=["POST"])
def generates():
    global authed_session
    vfile = req.files["voice"]
    voice = vfile.read()
    vcodes = base64.b64encode(voice)

    f = open("test.wav", "wb")
    f.write(voice)
    f.close()

    status_code = 0

    while status_code != 200:

        resp = authed_session.post("https://speech.googleapis.com/v1/speech:recognize", 
                    headers = {
                        "Content-Type": "application/json; charset=utf-8",
                    },
                    data = json.dumps({
                        'config': {
                            'encoding': 'LINEAR16',
                            'sampleRateHertz': 48000,
                            'languageCode': 'en-US',
                            'enableWordTimeOffsets': True
                        },
                        'audio': {
                            'content': vcodes.decode("utf-8")
                        }
                    })
                )
        status_code = resp.status_code

        if status_code != 200:
         
            SCOPES = ['https://www.googleapis.com/auth/application-default']
            data = json.load(open("whdcred.json"))
            credentials = service_account.Credentials.from_service_account_info(data, scopes=["https://www.googleapis.com/auth/cloud-platform", "https://www.googleapis.com/auth/cloud-language"])
            authed_session = AuthorizedSession(credentials)

    data = json.loads(resp.text)
    try:
        transcript = data["results"][0]["alternatives"][0]["transcript"]
        num_words = 100
        # output_text = "lk/afjknafclk/afjknafc lk/afjknafc lk/afjknafc lk/afjknafc lk/afjknafc lk/afjknafc lk/afjknafc"
        output_text = generate_text(num_words, transcript + ".")
        # output_text = generate_text(num_words, "there are many delicious salad in my shop. you must try it")
        # print(transcript)
        
        return json.dumps({"data": "Success", "text": transcript, "gen":  output_text})
    except:
        return '{"data": "Fail"}'

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=8000)