import re
# import nltk
# import pyLDAvis
# from IPython.core.display import HTML
import wordcloud
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk import SnowballStemmer, TextCollection, Text
import os
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from gensim import corpora, models
import pyLDAvis.gensim
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
from nltk.sentiment import SentimentIntensityAnalyzer
import math


def filter_and_stem_text(path: str) -> str:
    file = open(path, 'r')
    text = file.read()
    text = re.sub(r'\'|\d|\.|@\w+|’|\(|\)|\"|,|;|:|!|\?|\*|\[|]|\{|}|#|$|&|<|>|_|“|”|\$|\||«|»', '', text)
    text = re.sub(r'-|—', ' ', text)
    result = text.split()
    result = [sb.stem(word) for word in result if word not in blacklist]
    result = ' '.join(result)
    return result


def generate_cloud(words: list[str], filename: str, color='white'):
    text_color = wordcloud.get_single_color_func(color)
    word_cloud = WordCloud(collocations=False, background_color='black',
                           max_words=50, min_word_length=4, scale=2, color_func=text_color).generate('+'.join(words))
    word_cloud.to_file('./clouds/'+filename)


sb = SnowballStemmer('english')
blacklist = set(stopwords.words('english'))

post_stem_blacklist = ['dont', 'a', 'aa', 'i', 'he', 'she', 'the', 'it', 'one', 'what', 'but', '', 'would', 'could',
                       'want', 'said', 'say']
# przygotowanie tekstu
book1 = filter_and_stem_text('./atlas-shrugged.txt')
book2 = filter_and_stem_text('./the-fountainhead.txt')

both_books = (book1 + ' ' + book2).split(' ')

docs = [*book1.split('xxyxx'), *book2.split('xxyxx')]
# [ch1, ch2, ch3, ...]

docs = [doc.split(' ') for doc in docs]
# [[w1,w2,...],[w1,w2,...],[w1,w2,...]]

docs = [[word for word in doc if word not in post_stem_blacklist] for doc in docs]

vec = CountVectorizer(lowercase=False, token_pattern=None, tokenizer=lambda x: x)
docs_vectorized = vec.fit_transform(docs)
df = pd.DataFrame(docs_vectorized.toarray().T, index=vec.get_feature_names_out())
df = df.sort_values(by=list(df.columns), axis=0, ascending=[False] * len(list(df.columns)))
print(df.head(10))

# chmury słów
if not os.path.exists('./clouds'):
    os.mkdir('./clouds')

docs_unique = []
common = set(docs[0])
print(common)
for i in range(len(docs)):
    other_docs = [' '.join(doc) for doc in docs if doc is not docs[i]]
    unique = set(docs[i]) - set((' '.join(other_docs).split(' ')))
    docs_unique.append([word for word in docs[i] if word in unique])

i = 0
for doc in docs_unique:
    generate_cloud(doc, 'doc' + str(i) + '.png')
    i += 1

for doc in docs:
    common = common & set(doc)

common_freq = [word for word in both_books if word in common]
generate_cloud(common_freq, 'common.png', 'yellow')

# 3.1. Interpretacja wyników wykorzystująca grupowanie

linkage_data = linkage(df.to_numpy().T, method='ward', metric='euclidean')
dendrogram(linkage_data)
plt.ylabel("Termy")
plt.xlabel("Rozdziały")
plt.show()

#3.2. Interpretacja wyników wykorzystująca modele wątków

dictionary = corpora.Dictionary(docs)

print('Sample word to id mappings:\n', list(dictionary.items())[:15])
print('Total Vocabulary Size:', len(dictionary))

corpus_vect = [dictionary.doc2bow(doc) for doc in docs]

num_topics = 5

ldamodel = models.ldamodel.LdaModel(corpus_vect, num_topics=num_topics, id2word=dictionary, passes=25, alpha='symmetric')

for num, topic in ldamodel.show_topics(formatted=True, num_topics=num_topics, num_words=10):
    print(str(num)+": " + topic)

print('\nPerplexity: ', ldamodel.log_perplexity(corpus_vect))
coherence_model_lda = models.CoherenceModel(model=ldamodel, corpus=corpus_vect, dictionary=dictionary, coherence='u_mass')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

vis = pyLDAvis.gensim.prepare(ldamodel, corpus_vect, dictionary)
pyLDAvis.save_html(vis, 'lda_result.html')

# modyfikacja korpusu

atlas_chapters_tokens = [chapter.split(' ') for chapter in book1.split('chapter')]
atlas_chapters_tokens = [[word for word in chapter if word not in post_stem_blacklist] for chapter in atlas_chapters_tokens]

fountain_chapters_tokens = [chapter.split(' ') for chapter in book2.split('chapter')]
fountain_chapters_tokens = [[word for word in chapter if word not in post_stem_blacklist] for chapter in fountain_chapters_tokens]

atlas_tokens = list(np.concatenate(atlas_chapters_tokens))
fountain_tokens = list(np.concatenate(fountain_chapters_tokens))

tc = TextCollection([atlas_tokens, fountain_tokens])
terms_50 = tc.vocab().most_common(50)

atlas_chap_t = [Text(chapter) for chapter in atlas_chapters_tokens]
fountain_chap_t = [Text(chapter) for chapter in fountain_chapters_tokens]

# 4.1 Budowa klasyfikatora z wagami binarnymi

matrix_bin = [[0 if ch.count(t[0]) == 0 else 1 for t in terms_50] for ch in atlas_chap_t]
[r.append('atlas') for r in matrix_bin]
matrix_bin_r = [[0 if ch.count(t[0]) == 0 else 1 for t in terms_50] for ch in atlas_chap_t]
[r.append('fountain') for r in matrix_bin_r]

matrix_bin.extend(matrix_bin_r)
df_bin = pd.DataFrame(matrix_bin)
X_train, X_test, y_train, y_test = train_test_split(df_bin.iloc[:, :-1], df_bin.iloc[:, -1], test_size=0.5)
model_b = GaussianNB()
model_b.fit(X_train, y_train)
y_pred = model_b.predict(X_test)
comparison = [pred == train for pred, train in zip(y_pred, y_test)]
print(sum(comparison) / len(comparison))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 4.2 klasyfikator logarytmiczny:

matrix_log = [[math.log2(ch.count(t[0]) + 1) for t in terms_50] for ch in atlas_chap_t]
[r.append('atlas') for r in matrix_log]
matrix_log_r = [[math.log2(ch.count(t[0]) + 1) for t in terms_50] for ch in fountain_chap_t]
[r.append('fountain') for r in matrix_log_r]
matrix_log.extend(matrix_log_r)
df_bin = pd.DataFrame(matrix_log)
X_train, X_test, y_train, y_test = train_test_split(df_bin.iloc[:, :-1], df_bin.iloc[:, -1], test_size=0.5)
model_b = GaussianNB()
model_b.fit(X_train, y_train)
y_pred = model_b.predict(X_test)
comparison = [pred == rzecz for pred, rzecz in zip (y_pred, y_test)]
print(sum(comparison) / len(comparison))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 4.3 klasyfikator z TFIDF

matrix_tfidf = [[tc.tf_idf(t[0], ch) for t in terms_50] for ch in atlas_chap_t]
[r.append('atlas') for r in matrix_tfidf]
matrix_tfidf_r = [[tc.tf_idf(t[0], ch) for t in terms_50] for ch in fountain_chap_t]
[r.append('fountain') for r in matrix_tfidf_r]
matrix_tfidf.extend(matrix_tfidf_r)
df_bin = pd.DataFrame(matrix_tfidf)
X_train, X_test, y_train, y_test = train_test_split(df_bin.iloc[:, :-1], df_bin.iloc[:, -1], test_size=0.5)
model_b = GaussianNB()
model_b.fit(X_train, y_train)
y_pred = model_b.predict(X_test)
comparison = [pred == rzecz for pred, rzecz in zip (y_pred, y_test)]
print(sum(comparison) / len(comparison))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 5. Wyznaczenie wskaźników polaryzacji nastroju

# nltk.download('vader_lexicon')

sa_vader = SentimentIntensityAnalyzer()

polar_veder = {i: sa_vader.polarity_scores(kom)["compound"] for i, kom in enumerate([' '.join(chapter) for chapter in atlas_chapters_tokens])}
plt.style.use('default')
plt.plot(polar_veder.keys(), polar_veder.values(), 'y.')
plt.show()

polar_veder = {i: sa_vader.polarity_scores(kom)["compound"] for i, kom in enumerate([' '.join(chapter) for chapter in fountain_chapters_tokens])}
plt.style.use('default')
plt.plot(polar_veder.keys(), polar_veder.values(), 'y.')
plt.show()
