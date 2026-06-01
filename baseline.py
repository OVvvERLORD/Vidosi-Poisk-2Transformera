import gensim.downloader
import numpy as np
from sklearn import svm
import re
from typing import Union, Optional, List

class W2VSentenceEmbedder:
    """
    Эмбеддер предложений на базе Word2Vec.
    Инкапсулирует препроцессинг, усреднение векторов слов и нормализацию.
    Совместим с DataStorage через протокол Callable + атрибут .vector_size.
    """
    def __init__(self, model: str = "word2vec-google-news-300"):
        
        self.w2v = gensim.downloader.load(model)
        self.vector_size = self.w2v.vector_size

    def __call__(self, text: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Позволяет вызывать объект как функцию: embedder("text") или embedder(["text1", "text2"])
        """
        if isinstance(text, str):
            return self.embed_single(text)
        elif isinstance(text, list):
            return self.embed_batch(text)
        else:
            raise TypeError("Ожидается str или List[str]")

    def _preprocess(self, sentences: List[str]) -> List[List[np.ndarray]]:
        '''
        Не очень оопшно, но у нас есть вектор из промптов,
        из которых мы разбивааем на вектор из слов, затем каждое корректное
        слово с точки зрения w2v мы закидываем в сам w2v, то есть для каждого
        промпта ("пользовательского запроса") мы получаем вектор из эмбеддингов каждого допустимого
        слова предложения.
        '''
        new_x = []
        for sentence in sentences:
            words = re.findall(r'\w+', sentence.lower())
            vectors = [self.w2v[word] for word in words if word in self.w2v]
            new_x.append(vectors)
        return new_x

    def _merge(self, vectors_list: List[List[np.ndarray]]) -> List[np.ndarray]:
        '''
        Нам приходит вектор, каждый элемент которого является массивом
        эмбеддингов для какого-то предложения. Мы складываем все вектора-
        эмбеддинги одного предложения и нормализуем их, надеясь, что таким образом сохраним 
        смысл всего предложения. Нормализация нужна для того, чтобы мы 
        не зависили от количества слов в предложении.
        '''
        merged = []
        for vecs in vectors_list:
            if not vecs:
                merged.append(np.zeros(self.vector_size, dtype=np.float32))
                continue
                
            shared = np.sum(vecs, axis=0)
            norm = np.linalg.norm(shared)
            
            # Нормализуем и гарантируем float32 (требование usearch)
            merged.append((shared / norm if norm > 0 else shared).astype(np.float32))
        return merged

    def embed_single(self, text: str) -> np.ndarray:
        return self._merge(self._preprocess([text]))[0]

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        return self._merge(self._preprocess(texts))
    
class SupportModel:
    emb : Optional['W2VSentenceEmbedder']
    svc : 'BaseEstimator'

    def __init__(self, svc_model : svm.SVC | None, w2v_model_path = 'word2vec-google-news-300'):
        '''
        '''
        self.w2v = gensim.downloader.load(w2v_model_path)
        self.svc = svc_model if  svc_model is not None else svm.SVC(probability = True)
        self._is_fitted = False
                
    def fit(self, X, y, **kwargs):
        '''
        На вход функции fit нам приходит вектор из промптов и вектор из классов, которым эти промпты принадлежат. Возвращется
        обученная модель, которая может предсказывать класс для нового промпта.
        '''
        X = self.emb(X)

        self.svc.fit(X, y, **kwargs)
        self._is_fitted = True
        return self

    def predict(self, sentence):
        sentence = self.emb(sentence)

        if not self._is_fitted:
            raise Exception('Model is not fitted yet!')
        return self.svc.predict(sentence)

        