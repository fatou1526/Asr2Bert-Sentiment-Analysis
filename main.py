from fastapi import FastAPI 
from pydantic import BaseModel
from utils import *
import torch
import librosa
import torch.nn as nn
from fastapi import UploadFile, File
import soundfile as sf
from transformers import AutoTokenizer
import huggingface_hub

app = FastAPI()

asr = ASRInference()

model_loaded = SentimentAnalysisBertModel.from_pretrained("Fatou/asr2bert-sentimentanalysis")
tokenizer_loaded = AutoTokenizer.from_pretrained("Fatou/asr2bert-sentimentanalysis") 

# Definition des classes
classes = ["positive", "negative"]

def predict(text):
    with torch.no_grad():
        inputs = tokenizer_loaded(text, return_tensors='pt')

        outputs = model_loaded(inputs['input_ids'], inputs['attention_mask'])
        pred = torch.max(torch.softmax(outputs, dim=1), dim=1)
        classe = classes[pred.indices.item()]
    return classe

class InputText(BaseModel):
    text: str

@app.post('/asr')
def inference(file: UploadFile = File(...)):
    audio, _ = librosa.load(file.file, sr=16_000)
    text = asr.inference(audio)
    sentiment = predict(text)
    return {"Transcription": text,
            "Sentiment": sentiment
            }
    
    