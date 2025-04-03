# # Use a pipeline as a high-level helper
# from transformers import pipeline
#
# pipe = pipeline("text-classification", model="Captain-1337/CrudeBERT")

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
import pandas as pd

headlines = [
    "Major Explosion, Fire at Oil Refinery in Southeast Philadelphia",
    "PETROLEOS confirms Gulf of Mexico oil platform accident",
    "CASUALTIES FEARED AT OIL ACCIDENT NEAR IRANS BORDER",
    "EIA Chief expects Global Oil Demand Growth 1 M B/D to 2011",
    "Turkey Jan-Oct Crude Imports +98.5% To 57.9M MT",
    "China’s crude oil imports up 78.30% in February 2019",
    "Russia Energy Agency: Sees Oil Output put Flat In 2005",
    "Malaysia Oil Production Steady This Year At 700,000 B/D",
    "ExxonMobil:Nigerian Oil Output Unaffected By Union Threat",
    "Yukos July Oil Output Flat On Mo, 1.73M B/D - Prime-Tass",
    "2nd UPDATE: Mexico’s Oil Output Unaffected By Hurricane",
    "UPDATE: Ecuador July Oil Exports Flat On Mo At 337,000 B/D",
    "China February Crude Imports -16.0% On Year",
    "Turkey May Crude Imports down 11.0% On Year",
    "Japan June Crude Oil Imports decrease 10.9% On Yr",
    "Iran’s Feb Oil Exports +20.9% On Mo at 1.56M B/D - Official",
    "Apache announces large petroleum discovery in Philadelphia",
    "Turkey finds oil near Syria, Iraq border"
]
example_headlines = pd.DataFrame(headlines, columns=["Headline"])

config_path = './crude_bert_config.json'
model_path = '../Local_Data/crude_bert_model.bin'

# Load the configuration
config = AutoConfig.from_pretrained(config_path)

# Create the model from the configuration
model = AutoModelForSequenceClassification.from_config(config)

# Load the model's state dictionary
state_dict = torch.load(model_path)

# Inspect keys, if "bert.embeddings.position_ids" is unexpected, remove or adjust it
state_dict.pop("bert.embeddings.position_ids", None)

# Load the adjusted state dictionary into the model
model.load_state_dict(state_dict, strict=False)  # Using strict=False to ignore non-critical mismatches

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Define the prediction function
def predict_to_df(texts, model, tokenizer):
    model.eval()
    data = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        softmax_scores = torch.nn.functional.softmax(logits, dim=-1)
        pred_label_id = torch.argmax(softmax_scores, dim=-1).item()
        class_names = ['positive', 'negative', 'neutral']
        predicted_label = class_names[pred_label_id]
        data.append([text, predicted_label])
    df = pd.DataFrame(data, columns=["Headline", "Classification"])
    return df

# # Create DataFrame
# example_headlines = pd.DataFrame(headlines, columns=["Headline"])
#
# # Apply classification
# result_df = predict_to_df(example_headlines['Headline'].tolist(), model, tokenizer)
#
# print(result_df)

# Define a function to predict score integers from a list of strings

def predict_scores(texts):
    model.eval()
    scores = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        softmax_scores = torch.nn.functional.softmax(logits, dim=-1)
        # Get the index of the highest score (integer score corresponding to the predicted class)
        pred_label_id = torch.argmax(softmax_scores, dim=-1).item()
        scores.append(pred_label_id)
    return scores

if __name__ == "__main__":
    print(predict_scores(headlines))
