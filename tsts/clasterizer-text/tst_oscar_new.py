# encoding: utf-8
import pandas as pd
from collections import Counter
import time
import sys
import re
import pymorphy2
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime
from sklearn.base import TransformerMixin
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.pipeline import Pipeline
from functools import wraps


class DenseTransformer(TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.todense()


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
num_clasters_for_kMeans = 20
engine_for_pd = 'c'
# engine_for_pd = 'python'
data_file_name = r'D:\data\oscar'


def text_cleaner(text):
    text = str(text).lower()
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
def learn_and_predict_DBSCAN(data, *args):
    text_clstz = Pipeline([
        ('tfidf', TfidfVectorizer(binary=True)),
        # ('svd', TruncatedSVD(n_components=100, random_state=123)),
        # ('to_dense', DenseTransformer()),
        ('DBSCAN', DBSCAN(eps=0.2, min_samples=5, metric='euclidean'))
    ])
    data['tag'] = text_clstz.fit_predict(data['normal_query'])
    print("{} clusters".format(len(set(data['tag'].tolist()))))
    return data


@timethis
def learn_and_predict_AgglomerativeClustering(data, num_clasters_for_kMeans):
    text_clstz = Pipeline([
        ('tfidf', TfidfVectorizer(binary=True)),
        # ('svd', TruncatedSVD(n_components=100, random_state=123)),
        ('to_dense', DenseTransformer()),
        ('afp', AgglomerativeClustering(n_clusters=num_clasters_for_kMeans,
                                        affinity='cosine',
                                        linkage='complete'))
    ])
    data['tag'] = text_clstz.fit_predict(data['normal_query'])
    print("{} clusters".format(len(set(data['tag'].tolist()))))
    return data


def commons(text):
    count.clear()
    count.update(filter(lambda x: len(x) > 2, text.split()))
    text = str(count.most_common(10))
    return text


def count_words(text):
    count.update(filter(lambda x: len(x) > 2, text.split()))  # take words more than 2 symbols
    return text


@timethis
def save_file(dt, excel_file_to):
    writer = pd.ExcelWriter(excel_file_to, engine='xlsxwriter')
    dt.to_excel(writer, 'Sheet1')
    writer.save()


@timethis
def filter_data_oscar(data):
    return data[data['normal_query'].str.contains('oscar|оскар')].sort_index()


@timethis
def main():
    data = load_from_csv(data_file_name)  # load all data
    only_oscars = filter_data_oscar(data)  # filter only that contains oscars
    save_data(only_oscars, data_file_name + '_oscar')  # save obly oscar to file
    data = ''  # free memory from trash
    only_oscars = load_from_csv(data_file_name + '_oscar')  # load again only oscars from file
    only_oscars['normal_query'] = only_oscars['normal_query'].map(
        lambda x: text_cleaner(x))  # lemmatize only oscars

    save_data(only_oscars, data_file_name + '_oscar_normal')  # save lemmatized oscars
    only_oscars = only_oscars.sort_index()
    # on MOW time oscar was from 03/00 25/02/19 till 07/00 25/02/2019
    # lets assume that time in this table is MOW
    delimited = [only_oscars[:datetime(2019, 2, 25, 2, 59, 59)].copy(deep=True),  # before
                 only_oscars[datetime(2019, 2, 25, 3, 0, 0):datetime(2019, 2, 25, 6, 59, 59)].copy(
                     deep=True),  # during
                 only_oscars[datetime(2019, 2, 25, 7, 0, 0):].copy(deep=True)]  # after

    names_of_periods = {0: 'BEFORE\n\n', 1: 'DURING\n\n', 2: 'AFTER\n\n'}
    # print info about these datasets
    for (num, d) in enumerate(delimited):
        print(names_of_periods[num])
        d.info()
        print(d.head())

    for (num, d) in enumerate(delimited):
        print(names_of_periods[num])
        count.clear()
        fn = names_of_periods[num].strip().lower()
        d['normal_query'].map(lambda x: count_words(x))
        save_file(pd.DataFrame(count.most_common()),
                  r'd:\data\oscar_freq_{}.xlsx'.format(fn))  # save frequency of words to excel

        # lets try to make Hierarchical clustering
        data = learn_and_predict_AgglomerativeClustering(d, num_clasters_for_kMeans)
        save_data(data.sort_values('tag'), r'D:\data\oscarAgglomerativeClustering_{}'.format(fn))
        # most common words in clusters
        df = data.groupby('tag')['normal_query'].apply(lambda words: ' '.join(words))
        df = pd.DataFrame(df)
        df['normal_query'] = df['normal_query'].apply(commons)

        save_data(df, data_file_name + 'Agglomerative_Group_{}'.format(fn))

        #also lets try DBSCAN clustering
        data = learn_and_predict_DBSCAN(d, num_clasters_for_kMeans)
        save_data(data.sort_values('tag'), r'D:\data\oscarDBSCANClustering_{}'.format(fn))
        # most common words in clusters
        df = data.groupby('tag')['normal_query'].apply(lambda words: ' '.join(words))
        df = pd.DataFrame(df)
        df['normal_query'] = df['normal_query'].apply(commons)

        save_data(df, data_file_name + 'DBSCAN_Group_{}'.format(fn))


if __name__ == '__main__':
    sys.exit(main())
