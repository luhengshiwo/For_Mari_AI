"""Reload and serve a saved model"""

__author__ = "luheng"

from pathlib import Path
from tensorflow.contrib import predictor
texts = ['I hate Harry Potter.',
         'The Da Vinci Code is actually a good movie...']

def test(path):
    texts = [text.split(' ') for text in path]
    lengths = [len(text) for text in texts]
    max_len = max(lengths)
    new_texts = [text + ['<pad>'] * (max_len - l)
                 for text, l in zip(texts, lengths)]
    labels_len = len(path)
    return {'text': new_texts, 'text_len': lengths, 'label': ['0'] * labels_len}

if __name__ == '__main__':
    export_dir = 'saved_model'
    subdirs = [x for x in Path(export_dir).iterdir()
               if x.is_dir() and 'temp' not in str(x)]
    latest = str(sorted(subdirs)[-1])
    predict_fn = predictor.from_saved_model(latest)
    predictions = predict_fn(test(texts))
    print(predictions)
