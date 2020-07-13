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

    
    def load(self):
        res = requests.get(
            "https://api.lemediatv.fr/api/1/public/stories/?page=1&per_page=1000",
            headers = {
                'X-Fields': 'title,primary_category,published_at,slug,story_tags,headline_or_extract_medium,content'
            }
        )
        entries = res.json()['results']

        for entry in entries:
            title_words = text.extract_words(entry['title'])
            content_words = text.extract_words(text.strip_tags(entry['content']))

            article = {
                'title': entry['title'],
                'headline_or_extract_medium': entry['headline_or_extract_medium'],
                'category': entry['primary_category']['slug'],
                'published_at': entry['published_at'],#datetime.datetime.strptime(entry['published_at'][:19], '%Y-%m-%dT%H:%M:%S'),
                'slug': entry['slug'],
                'tags': [tag['slug'] for tag in entry['story_tags']],
                'title_words': title_words,
                'content_words': content_words
            }
            
            self.articles[article['slug']] = article
            for tag in entry['story_tags']:
                if tag['slug'] not in self.tags:
                    self.tags[tag['slug']] = 1
                else:
                    self.tags[tag['slug']] += 1

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
        self.title_word_idf = np.array([math.log(len(self.article_list)/self.title_words[word]) for word in self.title_word_list])
        self.content_word_idf = np.array([math.log(len(self.article_list)/self.content_words[word]) for word in self.content_word_list])
        self.tag_idf = np.array([math.log(len(self.article_list)/self.tags[tag]) for tag in self.tag_list])
        #idf_norm = np.linalg.norm(self.tag_idf)
        #self.tag_idf = np.divide(self.tag_idf, idf_norm)

        self.tag_matrix = np.array([
            np.multiply([1 if tag in self.articles[article]['tags'] else 0 for tag in self.tag_list], self.tag_idf)
            for article in self.article_list
        ])

        title_embeddings = np.array([
            text.compute_embeddings(
                self.articles[article]['title_words'],
                [self.title_words[word] for word in self.articles[article]['title_words']],
                False
            )
            for article in self.article_list
        ])

        content_embeddings = np.array([
            text.compute_embeddings(
                self.articles[article]['content_words'],
                [self.content_words[word] for word in self.articles[article]['content_words']],
                False
            )
            for article in self.article_list
        ])

        self.word_embeddings = np.add(
            np.multiply(self.WORD_WEIGHTS['TITLE'], title_embeddings),
            np.multiply(self.WORD_WEIGHTS['CONTENT'], content_embeddings)
        )

        r = 0.5
        self.matrix = np.concatenate(
            (
                np.multiply((1-r)/self.word_embeddings.shape[1], self.word_embeddings),
                np.multiply(r/self.tag_matrix.shape[1], self.tag_matrix)
            ),
            axis = 1
        )

        open('matrix.json', 'w+').write(json.dumps()).close()

    def reduce(self, n_dims):
        print(self.matrix.shape)
        pca = sklearn.decomposition.PCA(n_components = n_dims)
        pca.fit(self.matrix)
        print(pca.explained_variance_ratio_)
        self.matrix = pca.transform(self.matrix)
        print(self.matrix.shape)


    def distance(self, a, b):
        a_pos = self.article_list.index(a)
        b_pos = self.article_list.index(b)

        return scipy.spatial.distance.cosine(self.matrix[a_pos,:], self.matrix[b_pos,:])    

    def closest(self, article, n):
        article_pos = self.article_list.index(article)

        articles = []
        for compare_slug in self.articles:
            if compare_slug == article:
                continue

            articles.append({
                'slug': compare_slug,
                'distance': self.distance(article, compare_slug)
            })

        articles.sort(key = lambda x: x['distance'])
        return articles[:n]

    def show_article(self, slug):
        return self.articles[slug]['title']

similar = SimilarArticles()
if os.path.exists('matrix.json'):
    similar.matrix = json.load(open('matrix.json', 'r'))
else:
    if os.path.exists('cache.json'):
        similar.from_json(open('cache.json', 'r').read())
    else:
        similar.load()
        with open('cache.json', 'w+') as f:
            f.write(similar.to_json())
            f.close()
    similar.prepare()

similar.reduce(12)

print(similar.distance("convention-pour-le-climat-macron-arnaque-les-citoyens-Dk9Yx_51TruQT2kMmp8qaw", "rojava-lavenir-suspendu-6J-ixMmYTZWjKgbndIqRxA"))
print(similar.distance("convention-pour-le-climat-macron-arnaque-les-citoyens-Dk9Yx_51TruQT2kMmp8qaw", "convention-citoyenne-pour-le-climat-macron-face-a-ses-contradictions-7GJB3OutTdaUHksYArtz8Q"))
print(similar.distance("convention-pour-le-climat-macron-arnaque-les-citoyens-Dk9Yx_51TruQT2kMmp8qaw", "bolivie-retour-sur-un-putsch-uKOZhoppQ7ydHATA7Xo7yA"))

slug = "convention-pour-le-climat-macron-arnaque-les-citoyens-Dk9Yx_51TruQT2kMmp8qaw"
closest = similar.closest(slug, 10)
print(similar.show_article(slug), " : ")
for item in closest:
    print("%s (%.3f)" % (similar.show_article(item['slug']), item['distance']))

slug = "lex-agent-secret-qui-en-savait-beaucoup-trop-4-contre-la-corruption-dans-lirak-sous-tutelle-des-etats-unis-7H5duI4KRPq3Cq2r9wqLwA"
closest = similar.closest(slug, 6)
print(similar.show_article(slug), " : ")
for item in closest:
    print("%s (%.3f)" % (similar.show_article(item['slug']), item['distance']))

slug = "startup-nation-larnaque-du-siecle-NnM1i8etQ4i-h07dMwikgQ"
closest = similar.closest(slug, 6)
print(similar.show_article(slug), " : ")
for item in closest:
    print("%s (%.3f)" % (similar.show_article(item['slug']), item['distance']))
#print(timeit.timeit('similar.closest("convention-pour-le-climat-macron-arnaque-les-citoyens-Dk9Yx_51TruQT2kMmp8qaw", 6)', number = 100, globals=globals()))