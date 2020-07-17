import numpy as np 
import requests
import datetime
import math
import timeit
import text
import sklearn.decomposition
import scipy.spatial
import json
import os

import multiprocessing as mp

class SimilarArticles:
    TAG_WEIGHTS = {
        'STORY_TAG': 0.5,
        'PROGRAM': 0,
        'WORD': 0.5
    }

    WORD_WEIGHTS = {
        'TITLE': 0.5,
        'CONTENT': 0.5
    }

    def to_json(self):
        return json.dumps({
            'articles': self.articles,
            'title_words': self.title_words,
            'content_words': self.content_words,
            'tags': self.tags
        })

    def from_json(self, json_str):
        data = json.loads(json_str)

        self.articles = data['articles']
        self.title_words = data['title_words']
        self.content_words = data['content_words']
        self.tags = data['tags']

        self.article_list = sorted(self.articles.keys())
        self.title_word_list = sorted(self.title_words.keys())
        self.content_word_list = sorted(self.content_words.keys())
        self.tag_list = sorted(self.tags.keys())
 
    def __init__(self):
        self.articles = {}
        self.title_words = {}
        self.title_word_list = []
        self.content_words = {}
        self.content_word_list = []
        self.tags = {}
        self.tag_list = []
        self.method = "embeddings"

        self.lifetimes = json.load(open('lifetimes.json', 'r+'))

        self.nlp_processor = None
    
    def load(self):
        if self.nlp_processor is None:
            self.nlp_processor = text.NLPProcessor()

        res = requests.get(
            "https://api.lemediatv.fr/api/1/public/stories/?page=1&per_page=1000",
            headers = {
                'X-Fields': 'title,primary_category,published_at,slug,story_tags,headline_or_extract_medium,content'
            }
        )
        entries = res.json()['results']

        for entry in entries:
            title_words = self.nlp_processor.extract_words(entry['title'])
            content_words = self.nlp_processor.extract_words(text.strip_tags(entry['content']))

            article = {
                'title': entry['title'],
                'headline_or_extract_medium': entry['headline_or_extract_medium'],
                'category': entry['primary_category']['slug'],
                'published_at': entry['published_at'],#datetime.datetime.strptime(entry['published_at'][:19], '%Y-%m-%dT%H:%M:%S'),
                'slug': entry['slug'],
                'tags': [tag['label'] for tag in entry['story_tags']],
                'title_words': title_words,
                'content_words': content_words
            }
            
            self.articles[article['slug']] = article
            for tag in article['tags']:
                if tag not in self.tags:
                    self.tags[tag] = 1
                else:
                    self.tags[tag] += 1

            for word in title_words:
                if word not in self.title_words:
                    self.title_words[word] = 1
                else:
                    self.title_words[word] += 1

            for word in content_words:
                if word not in self.content_words:
                    self.content_words[word] = 1
                else:
                    self.content_words[word] += 1

            print(len(self.content_words), len(self.title_words))
        
        self.article_list = sorted(self.articles.keys())
        self.title_word_list = sorted(self.title_words.keys())
        self.content_word_list = sorted(self.content_words.keys())
        self.tag_list = sorted(self.tags.keys())

    def prepare(self):
        if self.nlp_processor is None:
            self.nlp_processor = text.NLPProcessor()

        self.title_word_idf = np.array([math.log(len(self.article_list)/self.title_words[word]) for word in self.title_word_list])
        self.content_word_idf = np.array([math.log(len(self.article_list)/self.content_words[word]) for word in self.content_word_list])
        self.tag_idf = np.array([math.log(len(self.article_list)/self.tags[tag]) for tag in self.tag_list])
        #idf_norm = np.linalg.norm(self.tag_idf)
        #self.tag_idf = np.divide(self.tag_idf, idf_norm)

        self.tag_matrix = np.array([
            np.multiply([1 if tag in self.articles[article]['tags'] else 0 for tag in self.tag_list], self.tag_idf)
            for article in self.article_list
        ])

        pool = mp.Pool(mp.cpu_count())

        title_embeddings = np.array(pool.starmap(self.nlp_processor.compute_embeddings, [
            (
                self.articles[article]['title_words'],
                [self.title_words[word] for word in self.articles[article]['title_words']],
                False
            )
            for article in self.article_list
        ]))

        content_embeddings = np.array(pool.starmap(self.nlp_processor.compute_embeddings, [
            (
                self.articles[article]['content_words'],
                [self.content_words[word] for word in self.articles[article]['content_words']],
                False
            )
            for article in self.article_list
        ]))

        tag_embeddings = np.array(pool.starmap(self.nlp_processor.compute_embeddings, [
            (
                self.articles[article]['tags'],
                [self.tags[tag] for tag in self.articles[article]['tags']],
                False
            )
            for article in self.article_list
        ]))

        self.word_embeddings = np.add(
            np.multiply(self.WORD_WEIGHTS['TITLE'], title_embeddings),
            np.multiply(self.WORD_WEIGHTS['CONTENT'], content_embeddings)
        )

        self.embeddings = np.add(
            np.multiply(0.125, title_embeddings),
            np.multiply(0.125, content_embeddings),
            np.multiply(0.75, tag_embeddings)
        )

        r = 0.75
        self.matrix = np.concatenate(
            (
                np.multiply((1-r)/self.word_embeddings.shape[1], self.word_embeddings),
                np.multiply(r/self.tag_matrix.shape[1]/np.mean(self.tag_idf), self.tag_matrix)
            ),
            axis = 1
        )

        open('matrix.json', 'w+').write(json.dumps(self.matrix.tolist()))
        open('embeddings.json', 'w+').write(json.dumps(self.matrix.tolist()))

    def reduce(self, target_explained_variance = 0.95):
        for n_dims in range(self.matrix.shape[1]):
            pca = sklearn.decomposition.PCA(n_components = n_dims)
            pca.fit(self.matrix)
            explained_variance = np.sum(pca.explained_variance_ratio_)

            print(n_dims, explained_variance)

            if explained_variance >= target_explained_variance:
                self.matrix = pca.transform(self.matrix)
                return True

        raise ValueError('Cannot reach target explained variance')            

    def distance(self, a, b):
        a_pos = self.article_list.index(a)
        b_pos = self.article_list.index(b)

        if self.method == "mixed":
            return scipy.spatial.distance.cosine(self.matrix[a_pos,:], self.matrix[b_pos,:])
        elif self.method == "embeddings":
            return scipy.spatial.distance.cosine(self.embeddings[a_pos,:], self.embeddings[b_pos,:])
        else:
            raise ValueError("Invalid method '%s'" % (self.method))


    def closest(self, article, n):
        article_pos = self.article_list.index(article)

        articles = []
        for compare_slug in self.articles:
            if compare_slug == article:
                continue

            published_at = datetime.datetime.strptime(self.articles[compare_slug]['published_at'][:19], '%Y-%m-%dT%H:%M:%S')
            now = datetime.datetime.now()
            weeks = (now-published_at).total_seconds()/(86400*7)
            tau = self.lifetimes[self.articles[compare_slug]['category']]

            articles.append({
                'slug': compare_slug,
                'distance': self.distance(article, compare_slug),
                'time_factor': math.sqrt(1+(weeks/tau)**2)
            })

        articles.sort(key = lambda x: x['distance']*x['timefactor'])
        return articles[:n]

    def show_article(self, slug):
        return self.articles[slug]['title']
