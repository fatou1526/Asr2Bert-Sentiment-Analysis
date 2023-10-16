import os
import torch
import torch.nn as nn
import librosa
from transformers import AutoTokenizer, BertModel
from huggingface_hub import PyTorchModelHubMixin
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

class ASRInference:
    def __init__(self, model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-french"):
        
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)

    def inference(self, audio):
        inputs = self.processor(audio, sampling_rate=16_000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = self.model(inputs.input_values, attention_mask=inputs.attention_mask).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        text = self.processor.decode(predicted_ids[0]).lower()
        return text

config = {
    "model_name": "nlptown/bert-base-multilingual-uncased-sentiment",
    "max_length": 80,
    "trainfile": "/kaggle/input/allocine-movies-review/train.csv",
    "testfile": "/kaggle/input/allocine-movies-review/test.csv",
    "valfile": "/kaggle/input/allocine-movies-review/valid.csv",
    "batch_size": 10,
    "learning_rate": 2e-5,
    "n_epochs": 4,
    "n_classes": 1,
    "device": torch.device("cuda" if torch.cuda.is_available else "cpu")

}

class SentimentAnalysisBertModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self):
        super(SentimentAnalysisBertModel, self).__init__()
        self.pretrained_model = BertModel.from_pretrained(config['model_name'])   # bert base 768 hidden state
        self.classifier = nn.Linear(768, config['n_classes'])  # MLP

    def forward(self, input_ids, attention_mask):

        output = self.pretrained_model(input_ids = input_ids, attention_mask = attention_mask)    # batch de 768
        output = self.classifier(output.last_hidden_state)

        return output
    