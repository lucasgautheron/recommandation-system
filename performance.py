import numpy as np
import json
import requests
from similar import SimilarArticles

similar = SimilarArticles()

similar.from_json(open('cache.json', 'r').read())
similar.matrix = np.array(json.load(open('matrix.json', 'r')))
similar.embeddings = np.array(json.load(open('embeddings.json', 'r')))

similar.method = "mixed"

def get_articles():
    res = requests.get(
        "https://api.lemediatv.fr/api/1/public/stories/?page=1&per_page=1000",
        headers = {
            'X-Fields': 'title,canonical_url,slug,contextual_stories'
        }
    )
    entries = res.json()['results']

    articles = {}
    for entry in entries:
        articles[entry['slug']] = {
            'title': entry['title'],
            'url': entry['canonical_url']#,
#            'contextual_stories': entry['contextual_stories']
        }

    return articles

articles = get_articles()
json.dump(articles, open('articles.json', 'w+'), indent = 2)



# print(similar.distance("convention-pour-le-climat-macron-arnaque-les-citoyens-Dk9Yx_51TruQT2kMmp8qaw", "rojava-lavenir-suspendu-6J-ixMmYTZWjKgbndIqRxA"))
# print(similar.distance("convention-pour-le-climat-macron-arnaque-les-citoyens-Dk9Yx_51TruQT2kMmp8qaw", "convention-citoyenne-pour-le-climat-macron-face-a-ses-contradictions-7GJB3OutTdaUHksYArtz8Q"))
# print(similar.distance("convention-pour-le-climat-macron-arnaque-les-citoyens-Dk9Yx_51TruQT2kMmp8qaw", "bolivie-retour-sur-un-putsch-uKOZhoppQ7ydHATA7Xo7yA"))

# slug = "convention-pour-le-climat-macron-arnaque-les-citoyens-Dk9Yx_51TruQT2kMmp8qaw"
# closest = similar.closest(slug, 10)
# print(similar.show_article(slug), " : ")
# for item in closest:
#     print("%s (%.3f)" % (similar.show_article(item['slug']), item['distance']))

# slug = "lex-agent-secret-qui-en-savait-beaucoup-trop-4-contre-la-corruption-dans-lirak-sous-tutelle-des-etats-unis-7H5duI4KRPq3Cq2r9wqLwA"
# closest = similar.closest(slug, 6)
# print(similar.show_article(slug), " : ")
# for item in closest:
#     print("%s (%.3f)" % (similar.show_article(item['slug']), item['distance']))

# slug = "startup-nation-larnaque-du-siecle-NnM1i8etQ4i-h07dMwikgQ"
# closest = similar.closest(slug, 6)
# print(similar.show_article(slug), " : ")
# for item in closest:
#     print("%s (%.3f)" % (similar.show_article(item['slug']), item['distance']))
# #print(timeit.timeit('similar.closest("convention-pour-le-climat-macron-arnaque-les-citoyens-Dk9Yx_51TruQT2kMmp8qaw", 6)', number = 100, globals=globals()))