import numpy as np 
import requests
import datetime
import math
import timeit
import text
import sklearn.decomposition

def inverse_document_frequency(x, n):
    return math.log(n/x)

class SimilarArticles:
    TAG_WEIGHTS = {
        'STORY_TAG': 1,
        'PROGRAM': 0,
        'WORD': 0.1
    }

    def __init__(self):
        self.articles = []
        self.tags = {}
        self.tag_list = [0]

        res = requests.get(
            "https://api.lemediatv.fr/api/1/public/stories/?page=1&per_page=1000",
            headers = {
                'X-Fields': 'title,primary_category,published_at,slug,story_tags,headline_or_extract_medium'
            }
        )
        entries = res.json()['results']

        articles = {}
        tags = {}
        for entry in entries:
            words = text.extract_words(entry['title'])

            article = {
                'title': entry['title'],
                'headline_or_extract_medium': entry['headline_or_extract_medium'],
                'category': entry['primary_category']['slug'],
                'published_at': datetime.datetime.strptime(entry['published_at'][:19], '%Y-%m-%dT%H:%M:%S'),
                'slug': entry['slug'],
                'tags': [tag['slug'] for tag in entry['story_tags']] + [entry['primary_category']['slug']] + words
            }
            
            articles[article['slug']] = article
            for tag in entry['story_tags']:
                if tag['slug'] not in tags:
                    tags[tag['slug']] = {'scheme': tag['scheme']}

            if entry['primary_category']['slug'] not in tags:
                tags[entry['primary_category']['slug']] = {'scheme': 'PROGRAM'}

            for word in words:
                if word not in tags:
                    tags[word] = {'scheme': 'WORD'}

            print(words)

        self.articles = articles
        self.article_list = sorted(articles.keys())
        self.tags = tags
        self.tag_list = sorted(tags.keys())

    def prepare(self):
        self.article_tag_matrix = np.array([
            [
                1 if tag in self.articles[article]['tags'] else 0 for article in self.article_list
            ]
            for tag in self.tag_list
        ])

        pca = sklearn.decomposition.PCA(n_components=100)
        pca.fit(np.transpose(self.article_tag_matrix))
        
        self.tag_idf = np.array([inverse_document_frequency(
            tag_frequency,
            len(self.article_list)
        ) for tag_frequency in np.sum(self.article_tag_matrix, axis = 1)])

        self.tag_weights = np.array([
            self.TAG_WEIGHTS[self.tags[tag]['scheme']] for tag in self.tag_list
        ])

        self.tag_weights = np.multiply(self.tag_weights, self.tag_idf)


    def distance(self, a, b):
        a_pos = self.article_list.index(a)
        b_pos = self.article_list.index(b)

        return np.linalg.norm(
            np.multiply(self.article_tag_matrix[:,a_pos]-self.article_tag_matrix[:,b_pos], self.tag_weights)
        )

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


similar = SimilarArticles()
similar.prepare()
print(similar.article_tag_matrix.shape)
print(similar.tag_idf)
print(similar.tag_weights)

print(similar.distance("convention-pour-le-climat-macron-arnaque-les-citoyens-Dk9Yx_51TruQT2kMmp8qaw", "rojava-lavenir-suspendu-6J-ixMmYTZWjKgbndIqRxA"))
print(similar.distance("convention-pour-le-climat-macron-arnaque-les-citoyens-Dk9Yx_51TruQT2kMmp8qaw", "convention-citoyenne-pour-le-climat-macron-face-a-ses-contradictions-7GJB3OutTdaUHksYArtz8Q"))
print(similar.distance("convention-pour-le-climat-macron-arnaque-les-citoyens-Dk9Yx_51TruQT2kMmp8qaw", "bolivie-retour-sur-un-putsch-uKOZhoppQ7ydHATA7Xo7yA"))

print(similar.closest("convention-pour-le-climat-macron-arnaque-les-citoyens-Dk9Yx_51TruQT2kMmp8qaw", 10))
print(similar.closest("lex-agent-secret-qui-en-savait-beaucoup-trop-4-contre-la-corruption-dans-lirak-sous-tutelle-des-etats-unis-7H5duI4KRPq3Cq2r9wqLwA", 6))
print(similar.closest("startup-nation-larnaque-du-siecle-NnM1i8etQ4i-h07dMwikgQ", 6))
#print(timeit.timeit('similar.closest("convention-pour-le-climat-macron-arnaque-les-citoyens-Dk9Yx_51TruQT2kMmp8qaw", 6)', number = 100, globals=globals()))