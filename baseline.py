import gensim.downloader
import numpy as np
from sklearn import svm
import re

class Baseline:
    w2v : 'KeyedVectors'
    svc : 'BaseEstimator'

    def __init__(self, svc_model : svm.SVC | None, w2v_model_path = 'word2vec-google-news-300'):
        '''
        '''
        self.w2v = gensim.downloader.load(w2v_model_path)
        self.svc = svc_model if  svc_model is not None else svm.SVC()
        self._is_fitted = False

    def __sentence_preprocessing(self, X):
        '''
        Не очень оопшно, но у нас есть вектор из промптов,
        из которых мы разбивааем на вектор из слов, затем каждое корректное
        слово с точки зрения w2v мы закидываем в сам w2v, то есть для каждого
        промпта ("пользовательского запроса") мы получаем вектор из эмбеддингов каждого допустимого
        слова предложения.
        '''
        new_x = []
        for sentence in X:
            words = re.findall(r'\w+', sentence)
            vectors = []
            for word in words:
                if word.lower() in self.w2v:
                    vectors.append(self.w2v[word])
            
            new_x.append(vectors)
        return new_x

    def __vectors_merge(self, X):
        '''
        Нам приходит вектор, каждый элемент которого является массивом
        эмбеддингов для какого-то предложения. Мы складываем все вектора-
        эмбеддинги одного предложения и нормализуем их, надеясь, что таким образом сохраним 
        смысл всего предложения. Нормализация нужна для того, чтобы мы 
        не зависили от количества слов в предложении.
        '''
        merged_vectors = []
        for vectors_arr in X:
            shared_vector = np.zeros(self.w2v.vector_size)
            for vector in vectors_arr:
                shared_vector += vector

            merged_vectors.append(shared_vector / np.linalg.norm(shared_vector))

        return merged_vectors
                
    def fit(self, X, y, **kwargs):
        '''
        На вход функции fit нам приходит вектор из промптов и вектор из классов, которым эти промпты принадлежат. Возвращется
        обученная модель, которая может предсказывать класс для нового промпта.
        '''
        X = self.__sentence_preprocessing(X)
        X = self.__vectors_merge(X)

        self.svc.fit(X, y, **kwargs)
        self._is_fitted = True
        return self

    def predict(self, sentence):
        sentence = self.__sentence_preprocessing([sentence])
        sentence = self.__vectors_merge(sentence)

        if not self._is_fitted:
            raise Exception('Model is not fitted yet!')
        return self.svc.predict(sentence)

        