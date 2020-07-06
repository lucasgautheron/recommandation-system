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
    return [wrd.lemma.lower() for sent in nlp(s).sentences for wrd in sent.words if (wrd.upos in keep_pos and wrd.lemma not in stop_words and wrd.lemma not in ['.', ',', ':', ';'])]

# extract lemma
def get_embedding(s):
    embs = []
    for sent in nlp(s).sentences:
        for wrd in sent.words:
            if wrd.upos in keep_pos:
                nlpwrd = nlpspacy(wrd.lemma)
                embs.append(np.array(nlpwrd.vector))
    if len(embs) == 0:
        embs.append(np.zeros(300))

    embs = np.array(embs)
    return np.mean(embs, axis=0)

def compute_embeddings(words, weights):
    embs = []

    weight_sum = 0
    for word in words:
        idf = weights[word]
        weight_sum += idf
        nlpwrd = nlpspacy(word)
        embs.append(np.multiply(idf, np.array(nlpwrd.vector)))

    if len(embs) == 0:
        return np.zeros(300)

    return np.divide(np.mean(embs, axis=0), weight_sum)

    
