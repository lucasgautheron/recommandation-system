import numpy as np 
import requests
import datetime
import math
import timeit
import text
import sklearn.decomposition
import scipy.spatial

class SimilarArticles:
    TAG_WEIGHTS = {
        'STORY_TAG': 0.5,
        'PROGRAM': 0,
        'WORD': 0.5
    }

    def __init__(self):
        self.articles = {}
        self.words = {}
        self.word_list = []
        self.tags = {}
        self.tag_list = []

        res = requests.get(
            "https://api.lemediatv.fr/api/1/public/stories/?page=1&per_page=1000",
            headers = {
                'X-Fields': 'title,primary_category,published_at,slug,story_tags,headline_or_extract_medium'
            }
        )
        entries = res.json()['results']

        for entry in entries:
            words = text.extract_words(entry['title'])

            article = {
                'title': entry['title'],
                'headline_or_extract_medium': entry['headline_or_extract_medium'],
                'category': entry['primary_category']['slug'],
                'published_at': datetime.datetime.strptime(entry['published_at'][:19], '%Y-%m-%dT%H:%M:%S'),
                'slug': entry['slug'],
                'tags': [tag['slug'] for tag in entry['story_tags']],
                'words': words
            }
            
            self.articles[article['slug']] = article
            for tag in entry['story_tags']:
                if tag['slug'] not in self.tags:
                    self.tags[tag['slug']] = 1
                else:
                    self.tags[tag['slug']] += 1

            for word in words:
                if word not in self.words:
                    self.words[word] = 1
                else:
                    self.words[word] += 1

            print(len(self.words))

        
        self.article_list = sorted(self.articles.keys())
        self.word_list = sorted(self.words.keys())
        self.tag_list = sorted(self.tags.keys())

    def prepare(self):
        self.word_idf = np.array([math.log(len(self.article_list)/self.words[word]) for word in self.word_list])
        self.tag_idf = np.array([math.log(len(self.article_list)/self.tags[tag]) for tag in self.tag_list])

        self.tag_matrix = np.array([
            np.multiply([1 if tag in self.articles[article]['tags'] else 0 for tag in self.tag_list], self.tag_idf)
            for article in self.article_list
        ])

        self.word_embeddings = np.array([text.compute_embeddings(self.articles[article]['words'], self.words) for article in self.article_list])

        r = 0.5
        alpha = 1
        beta = alpha * (r/(1-r)) * self.word_embeddings.shape[1] / self.tag_matrix.shape[1]

        print(self.word_embeddings.shape, self.tag_matrix.shape, alpha, beta)

        norm = math.sqrt(alpha**2+beta**2)
        alpha /= norm
        beta /= norm
        
        print(alpha, beta)

        self.matrix = np.concatenate(
            (np.multiply(alpha, self.word_embeddings), np.multiply(beta, self.tag_matrix)),
            axis = 1
        )

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
similar.prepare()

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