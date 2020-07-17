import numpy as np
import stanfordnlp
import spacy
import nltk
import os

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

class NLPProcessor:
    def __init__(self):
        self.keep_pos = ['NOUN', 'VERB', 'PROPN']
        self.stop_words = set(nltk.corpus.stopwords.words('french'))

        #stanfordnlp.download('fr', "stanfordnlp_resources")
        self.nlpspacy = spacy.load('fr_core_news_md')

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
        self.nlp = stanfordnlp.Pipeline(**config) # Initialize the pipeline using a configuration dict

    def extract_words(self, s):
        if not s:
            return []

        return [
            wrd.lemma.lower()
            for sent in self.nlp(s).sentences
            for wrd in sent.words
            if (
                wrd.upos in self.keep_pos
                and wrd.lemma not in self.stop_words
                and wrd.lemma not in ['.', ',', ':', ';']
            )
        ]

    def compute_embeddings(self, words, weights = None, normalize = True):
        embs = []
        
        if weights == None:
            weights = np.full(len(words), 1)

        assert len(words) == len(weights), "word array's length does not match weights"

        for i in range(len(words)):
            word = words[i]
            weight = weights[i]

            nlpwrd = self.nlpspacy(word)
            embs.append(np.multiply(weight, np.array(nlpwrd.vector)))

        if len(embs) == 0:
            return np.zeros(300)

        vector = np.sum(embs, axis = 0)

        if normalize:
            return vector/np.linalg.norm(vector)
        else:
            return np.divide(vector, np.sum(weights))

