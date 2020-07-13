import numpy as np
import stanfordnlp
import spacy
import nltk
import os

#stanfordnlp.download('fr', "stanfordnlp_resources")
nlpspacy = spacy.load('fr_core_news_md')
stop_words = set(nltk.corpus.stopwords.words('french'))

config = {
    'processors': 'tokenize,pos,lemma', # Comma-separated list of processors to use
    'lang': 'fr', # Language code for the language to build the Pipeline in
    'tokenize_model_path': os.path.join('stanfordnlp_resources/fr_gsd_models/fr_gsd_tokenizer.pt'), # Processor-specific arguments are set with keys "{processor_name}_{argument_name}"
    'mwt_model_path': './stanfordnlp_resources/fr_gsd_models/fr_gsd_mwt_expander.pt',
    'pos_model_path': './stanfordnlp_resources/fr_gsd_models/fr_gsd_tagger.pt',
    'pos_pretrain_path': './stanfordnlp_resources/fr_gsd_models/fr_gsd.pretrain.pt',
    'lemma_model_path': './stanfordnlp_resources/fr_gsd_models/fr_gsd_lemmatizer.pt',
    'depparse_model_path': './stanfordnlp_resources/fr_gsd_models/fr_gsd_parser.pt',
    'depparse_pretrain_path': './stanfordnlp_resources/fr_gsd_models/fr_gsd.pretrain.pt'
}
nlp = stanfordnlp.Pipeline(**config) # Initialize the pipeline using a configuration dict

keep_pos = ['NOUN', 'VERB', 'PROPN']

def extract_words(s):
    return [
        wrd.lemma.lower()
        for sent in nlp(s).sentences
        for wrd in sent.words
        if (
            wrd.upos in keep_pos
            and wrd.lemma not in stop_words
            and wrd.lemma not in ['.', ',', ':', ';']
        )
    ]

def compute_embeddings(words, weights = None, normalize = True):
    embs = []
    
    if weights == None:
        weights = np.full(len(words), 1)

    assert len(words) == len(weights), "word array's length does not match weights"

    for i in range(len(words)):
        word = words[i]
        weight = weights[i]

        nlpwrd = nlpspacy(word)
        embs.append(np.multiply(weight, np.array(nlpwrd.vector)))

    if len(embs) == 0:
        return np.zeros(300)

    vector = np.sum(embs, axis = 0)

    if normalize:
        return vector/np.linalg.norm(vector)
    else:
        return np.divide(vector, np.sum(weights))

from io import StringIO
from html.parser import HTMLParser

class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.text = StringIO()
    def handle_data(self, d):
        self.text.write(d)
    def get_data(self):
        return self.text.getvalue()

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()