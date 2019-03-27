# encoding: utf-8
import pandas as pd
import time
import sys
import re
import pymorphy2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AffinityPropagation, MiniBatchKMeans
from sklearn.pipeline import Pipeline

morph = pymorphy2.MorphAnalyzer()


def text_cleaner(text):
    text = text.lower()

    # text = re.sub(r'https?://[\S]+', ' url ', text)  # замена интернет ссылок
    # text = re.sub(r'[\w\./]+\.[a-z]+', ' url ', text)

    # text = re.sub(r'\d+[-/\.]\d+[-/\.]\d+', ' date ', text)  # замена даты и времени
    # text = re.sub(r'\d+ ?гг?', ' date ', text)
    # text = re.sub(r'\d+:\d+(:\d+)?', ' time ', text)
    # text = re.sub(r'<[^>]*>', ' ', text)  # удаление html тагов
    text = re.sub(r'[\W]+', ' ', text)  # удаление лишних символов

    text = ' '.join(list(map(lambda x: morph.parse(x)[0].normal_form, text.split())))
    stw = ['в', 'по', 'на', 'из', 'и', 'или', 'не', 'но', 'за', 'над', 'под', 'то',
           'a', 'at', 'on', 'of', 'and', 'or', 'in', 'for', 'at']
    remove = r'\b(' + '|'.join(stw) + ')\b'
    text = re.sub(remove, ' ', text)

    text = re.sub(r'\b\w\b', ' ', text)  # удаление отдельно стоящих букв

    # text = re.sub(r'\b\d+\b', ' digit ', text)  # замена цифр
    return text


def load_data(file='oscar1'):
    print("[i] загружаем данные...")
    start_time = time.time()
    df = pd.read_csv(file, '\t')
    # data = data.head(10)
    print("считано: {} {}s".format(df.shape, time.time() - start_time))
    return df


def clean_data(data):
    print("[i] очистка данных...")
    start_time = time.time()
    data['normal_query__'] = data['normal_query'].map(lambda x: text_cleaner(str(x)))
    print("готово: {}s".format(time.time() - start_time))
    save_data(data, r'D:\data\oscar1_1')
    return data


def save_data(data, file='oscar2'):
    print("[i] сохраняем результат...")
    start_time = time.time()
    data.to_csv(file, sep='\t')
    print("готово: {}s".format(time.time() - start_time))


def learn_and_predict_KMeans(data):
    n_clusters = 100

    print("[i] обучение кластеризатора KMeans...")
    start_time = time.time()
    text_clstz = Pipeline([
        ('tfidf', TfidfVectorizer()),
        # ('km', KMeans(n_clusters=n_clusters)),
        # ( 'km', KMeans(n_clusters=n_clusters, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0) )
        ('km', KMeans(n_clusters=n_clusters, n_jobs=-1))
    ])

    text_clstz.fit(data['normal_query__'])
    data['tag'] = text_clstz.predict(data['normal_query__'])
    print("готово: {}s".format(time.time() - start_time))
    print("количество кластеров:", len(set(data['tag'])))
    return data


def learn_and_predict_MiniBatchKMeans(data):
    n_clusters = 100
    batch_size = 45
    print("[i] обучение кластеризатора MiniBatchKMeans...")
    start_time = time.time()
    text_clstz = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('mbkm', MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, ))
    ])

    text_clstz.fit(data['normal_query__'])
    data['tag'] = text_clstz.predict(data['normal_query__'])
    print("готово: {}s".format(time.time() - start_time))
    print("количество кластеров:", len(set(data['tag'])))
    return data


def learn_and_predict_AffinityPropagation(data):
    print("[i] обучение кластеризатора...")
    start_time = time.time()
    text_clstz = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('afp', AffinityPropagation())
    ])

    text_clstz.fit(data['normal_query__'])
    data['tag'] = text_clstz.predict(data['normal_query__'])
    print("готово: {}s".format(time.time() - start_time))
    print("количество кластеров:", len(set(data['tag'])))
    return data


def main():
    data = load_data(r'D:\data\oscar')
    data = clean_data(data)

    # data2 = learn_and_predict_KMeans(data.copy(True))
    # save_data(data2.sort_values('tag'), r'D:\data\oscarKMeans')

    data = learn_and_predict_MiniBatchKMeans(data)
    save_data(data.sort_values('tag'), r'D:\data\oscarMiniBatchKMeans')

    # data = learn_and_predict_AffinityPropagation(data)


if __name__ == '__main__':
    sys.exit(main())
