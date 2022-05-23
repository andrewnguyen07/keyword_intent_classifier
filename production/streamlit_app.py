import sys
import streamlit as st 
from ludwig.api import LudwigModel
import pandas as pd
import tensorflow as tf
from pathlib import Path
import numpy as np
from jsonschema import validate

model_path = Path(__file__).parents[1]
input = st.text_area('Keyword separated by new line', '')

def prob_handle(x):
    if x < 0.1:
        x = 0
    else:
        x = np.round(x*100,2)
    return x 

def prediction(input):
    input = input.split("\n")
    text = pd.Series(input)
    df = pd.DataFrame(text)
    df.columns = ["keywords"]
    model = LudwigModel.load(model_path)
    preds = model.predict(df)
    preds = preds[0]
    cols = ["intents_predictions","intents_probabilities_<UNK>",
    	"intents_probabilities_navigational","intents_probabilities_commercial",
        "intents_probabilities_transactional","intents_probabilities_informational"]
    final = preds[cols]
    final.columns = ["predicted_intents", "unknown", "navigational", "commercial", "transactional", "informational"]
    final = final.reset_index(drop=True)
    final = pd.concat([df,final], axis=1)
    final.unknown = final.unknown.apply(prob_handle)
    final.navigational = final.navigational.apply(prob_handle)
    final.commercial = final.commercial.apply(prob_handle)
    final.transactional = final.transactional.apply(prob_handle)
    final.informational = final.informational.apply(prob_handle)
    return final

def main():
    preds = prediction(input)
    st.write('Keyword intents are: ', preds)
    st.download_button("Press to download", preds.to_csv().encode('utf-8'), 'preds.csv', 'text/csv', key='download-csv'
)

if __name__ == '__main__':
    main()

# preds = prediction(input)

# st.write('Keyword intents are: ', preds)
# st.download_button(
#   "Press to download", preds.to_csv().encode('utf-8'), 'preds.csv', 'text/csv', key='download-csv'
# )

