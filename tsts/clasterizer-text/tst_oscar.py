# encoding: utf-8
import pandas as pd
from collections import Counter
import time
import sys
import re
import pymorphy2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans, AffinityPropagation, MiniBatchKMeans, AgglomerativeClustering
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from functools import wraps
from yellowbrick.cluster import SilhouetteVisualizer, KElbowVisualizer
import matplotlib.pyplot as plt
import numpy as np


class DenseTransformer(TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.todense()


def extractFeatures(data):
    tfidf = TfidfVectorizer(max_df=500, min_df=10).fit_transform(data)
    # tfidf = TruncatedSVD(n_components=2000, random_state=123).fit_transform(tfidf)
    return tfidf


def timethis(func):
    """
    Decorator that reports the execution time.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        print("Executing {}...".format(func.__name__))
        result = func(*args, **kwargs)
        end = time.time()
        print('{} took {}'.format(func.__name__, end - start))
        return result

    return wrapper


morph = pymorphy2.MorphAnalyzer()
count = Counter()
nrows_to_load = None
random_sample = None
num_clasters_for_kMeans = 5
engine_for_pd = 'c'
# engine_for_pd = 'python'
data_file_name = r'D:\data\oscar_oscar'


def text_cleaner(text):
    text = str(text).lower()

    # text = re.sub(r'https?://[\S]+', ' url ', text)  # замена интернет ссылок
    # text = re.sub(r'[\w\./]+\.[a-z]+', ' url ', text)

    # text = re.sub(r'\d+[-/\.]\d+[-/\.]\d+', ' date ', text)  # замена даты и времени
    # text = re.sub(r'\d+ ?гг?', ' date ', text)
    # text = re.sub(r'\d+:\d+(:\d+)?', ' time ', text)
    # text = re.sub(r'<[^>]*>', ' ', text)  # удаление html тагов
    text = re.sub(r'[\W]+', ' ', text)  # удаление лишних символов

    text = ' '.join(list(map(lambda x: morph.parse(x)[0].normal_form, text.split())))
    stw = ['в', 'по', 'на', 'из', 'и', 'или', 'не', 'но', 'за', 'над', 'под', 'то', 'для', "как",
           'a', 'at', 'on', 'of', 'and', 'or', 'in', 'for', 'at']
    remove = r'\b(' + '|'.join(stw) + ')\b'
    text = re.sub(remove, ' ', text)

    text = re.sub(r'\b\w\b', ' ', text)  # удаление отдельно стоящих букв

    # text = re.sub(r'\b\d+\b', ' digit ', text)  # замена цифр
    return text


@timethis
def load_from_csv(file):
    df = pd.read_csv(file, '\t', parse_dates=['datetime'], index_col='datetime',
                     converters={'normal_query': str},
                     nrows=nrows_to_load,
                     engine=engine_for_pd,
                     )
    # df.info()
    print("считано: {}".format(df.shape))
    return df


@timethis
def load_data_and_lemmatize(file='oscar1'):
    df = pd.read_csv(file, '\t', parse_dates=['datetime'], index_col='datetime',
                     converters={'normal_query': text_cleaner},
                     nrows=nrows_to_load,
                     engine=engine_for_pd,
                     )
    # df.info()
    return df


@timethis
def save_data(data, file='oscar2'):
    data.to_csv(file, sep='\t')


@timethis
def predict_clusters(data, text_clstz):
    data['tag'] = text_clstz.predict(data['normal_query'])
    print("количество кластеров:", len(set(data['tag'])))
    return data


@timethis
def learn_KMeans(data, num_clasters_for_kMeans):
    text_clstz = Pipeline([
        ('tfidf', TfidfVectorizer()),
        # ('km', KMeans(n_clusters=n_clusters)),
        # ( 'km', KMeans(n_clusters=n_clusters, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0) )
        ('km', KMeans(n_clusters=num_clasters_for_kMeans, n_jobs=-1))
    ])

    text_clstz.fit(data['normal_query'])
    print("Inertia %2.f" % text_clstz._final_estimator.inertia_)
    return text_clstz


def learn_and_predict_KMeans(data):
    text_clstz = learn_KMeans(data)
    data = predict_clusters(data, text_clstz)
    return data


@timethis
def plot_Silhouette(data, num_clasters_for_kMeans):
    batch_size = 7
    tfidf = extractFeatures(data['normal_query'])
    mbkm = MiniBatchKMeans(n_clusters=num_clasters_for_kMeans, batch_size=batch_size,
                           init_size=num_clasters_for_kMeans * 3)
    visualizer = SilhouetteVisualizer(mbkm)
    visualizer.fit(tfidf)
    visualizer.poof()


@timethis
def learn_MiniBatchKMeans(data, num_clasters_for_kMeans):
    batch_size = 7
    text_clstz = Pipeline([
        ('tfidf', TfidfVectorizer(max_df=500, min_df=10)),
        # ('svd', TruncatedSVD(n_components=100, random_state=123)),
        ('mbkm', MiniBatchKMeans(n_clusters=num_clasters_for_kMeans, batch_size=batch_size,
                                 init_size=num_clasters_for_kMeans * 3))
    ])

    text_clstz.fit(data['normal_query'])
    print("Inertia %2.f" % text_clstz._final_estimator.inertia_)
    return text_clstz


def learn_and_predict_MiniBatchKMeans(data, num_clasters_for_kMeans):
    text_clstz = learn_MiniBatchKMeans(data, num_clasters_for_kMeans)
    data = predict_clusters(data, text_clstz)
    return data


@timethis
def learn_and_predict_AgglomerativeClustering(data, num_clasters_for_kMeans):
    text_clstz = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('svd', TruncatedSVD(n_components=100, random_state=123)),
        # ('to_dense', DenseTransformer()),
        ('afp', AgglomerativeClustering(n_clusters=num_clasters_for_kMeans,
                                        affinity='cosine',
                                        linkage='complete'))
    ])
    data['tag'] = text_clstz.fit_predict(data['normal_query'])
    return data


@timethis
def learn_AffinityPropagation(data):
    text_clstz = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('afp', AffinityPropagation(preference=-50))
    ])
    text_clstz.fit(data['normal_query'])
    return text_clstz


def learn_and_predict_AffinityPropagation(data):
    text_clstz = learn_AffinityPropagation(data)
    data = predict_clusters(data, text_clstz)
    return data


def commons(text):
    count.clear()
    count.update(filter(lambda x: len(x) > 2, text.split()))
    text = str(count.most_common(10))
    return text


@timethis
def guess_clusters(data):
    dist = []
    min_clusters = 90
    max_clusters = 110

    @timethis
    def kkm(i):
        print(i)
        km = MiniBatchKMeans(n_clusters=i, batch_size=7, init_size=i * 3)
        km.fit(X)
        dist.append(km.inertia_)

    X = TfidfVectorizer(max_df=500, min_df=10).fit_transform(data['normal_query'])
    for i in range(min_clusters, max_clusters):
        print('Clusters: ', i)
        kkm(i)
    plt.plot(range(min_clusters, max_clusters), dist, marker='o')
    plt.xlabel('Clusters num')
    plt.ylabel('Distortion')
    plt.show()


def count_words(text):
    count.update(text.split())
    return text

@timethis
def save_file(dt, excel_file_to):
    writer = pd.ExcelWriter(excel_file_to, engine='xlsxwriter')
    dt.to_excel(writer, 'Sheet1')
    writer.save()

@timethis
def main():
    # load, lemmatize and save data (only for first use)

    # data = load_data_and_lemmatize(data_file_name)
    # save_data(data, data_file_name + 'Clean')  # save cleaned data

    # load lemmatized data filter by word oscar

    data = load_from_csv(data_file_name)
    # d = data[data['normal_query'].str.contains('oscar|оскар')].sort_index()
    # save_data(d, data_file_name + '_oscar')

    # take random sample to process if needed
    if random_sample is not None:
        data = data.sample(random_sample, random_state=20)
    # guess_clusters(data)
    # return

    data['normal_query'] = data['normal_query'].map(lambda x: count_words(x))
    save_file(pd.DataFrame(count.most_common()), r'd:\data\oscar_freq.xlsx')
    # data = learn_and_predict_AgglomerativeClustering(data, num_clasters_for_kMeans)
    # save_data(data.sort_values('tag'), r'D:\data\oscarAgglomerativeClustering')

    # data = learn_and_predict_KMeans(data, num_clasters_for_kMeans)
    # save_data(data.sort_values('tag'), r'D:\data\oscarKMeans')

    # Clusterize using KMeans
    # plot_Silhouette(data, num_clasters_for_kMeans)
    # data = learn_and_predict_MiniBatchKMeans(data, num_clasters_for_kMeans)
    # save_data(data.sort_values('tag'), r'D:\data\oscarMiniBatchKMeans')

    # data = learn_and_predict_AffinityPropagation(data)
    # save_data(data.sort_values('tag'), data_file_name + 'AffinityPropagation')

    # df = data.groupby('tag')['normal_query'].apply(lambda tags: ' '.join(tags))
    # df = pd.DataFrame(df)
    # df['normal_query'] = df['normal_query'].apply(commons)

    # save_data(df, data_file_name + 'Group')

    # silhouette(data)


if __name__ == '__main__':
    sys.exit(main())
