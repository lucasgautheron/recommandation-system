import numpy as np
import json
import requests
from similar import SimilarArticles
import random

similar = SimilarArticles()
similar.method = "mixed"

similar.from_json(open('cache.json', 'r').read())
similar.matrix = np.array(json.load(open('matrix.json', 'r')))
similar.embeddings = np.array(json.load(open('embeddings.json', 'r')))

articles = json.load(open('articles.json', 'r'))

article_slug = random.choice(list(articles.keys()))
article = articles[article_slug]

closest = similar.closest(article_slug, 6)
current_algorithm = requests.get("https://api.lemediatv.fr/api/1/public/stories/by-slug/%s" % (article_slug)).json()['results']['contextual_stories']

print(article['title'])

print("Algorithme testé :")
for item in closest:
    print("%s (%.3f, %.3f)" % (articles[item['slug']]['title'], item['distance'], item['time_factor']))

print("Système actuel :")
for item in current_algorithm:
    print(item['title'])

