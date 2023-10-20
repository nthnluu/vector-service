import numpy as np

import torch
from transformers import BertTokenizer, BertModel

import tqdm
import regex as re
import string


# 'justin871030/bert-base-uncased-goemotions-original-finetuned'
def clean_text(text):
    """Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers."""
    text = str(text).lower()  # Converting all text to lowercase
    text = re.sub("@[A-Za-z0-9_]+", "", text)  # Removing Mentions (Eg: @tserre)
    text = re.sub('\[.*?\]', '', text)  # Removing Text in brackets
    text = re.sub('https?://\S+|www\.\S+', '', text)  # Removing URLs
    text = re.sub('<.*?>+', ' ', text)  # Removing Text within '< >'
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)  # Removing Punctuations
    text = re.sub('\n', '', text)  # Removing new-line characters
    text = re.sub('\r', '', text)  # Removing Carriage Return characters
    text = re.sub(' +', ' ', text)  # Removing more than 1 space
    text = re.sub('\w*\d\w*', '', text)  # Removing text with digits
    text = text.strip()  # Removing any remaining redundant spaces
    return text




class TextTransformer:
    def __init__(self, version):
        self.model = BertModel.from_pretrained(version)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.tokenizer = BertTokenizer.from_pretrained(version, do_lower_case=True)
        self.embeddings = None

    def embed_text(self, comments):
        results = []
        for text in tqdm.notebook.tqdm(comments):
            input_ids = torch.tensor(self.tokenizer.encode(text)).unsqueeze(0)
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()

            if input_ids.shape[1] > 512:
                input_ids = input_ids[:, :512]

            outputs = self.model(input_ids)[0].mean(1).detach()
            if torch.cuda.is_available():
                outputs = outputs.cpu()
            results.append(outputs.squeeze().numpy())

        return torch.Tensor(np.array(results))

    def compare_similarity(self, input_embedding):
        return np.array([torch.nn.functional.cosine_similarity(input_embedding, embedding, dim=0)
                         for embedding in self.embeddings])

    def load_queries(self, queries):
        self.embeddings = self.embeddings(queries)
