import gensim.downloader
from gensim.models import KeyedVectors
import numpy as np
from sklearn import svm
import os
import re
from typing import Union, Optional, List
from sklearn.base import BaseEstimator

class W2VSentenceEmbedder:
    """
    Эмбеддер предложений на базе Word2Vec.
    Инкапсулирует препроцессинг, усреднение векторов слов и нормализацию.
    Совместим с DataStorage через протокол Callable + атрибут .vector_size.
    """
    def __init__(self, model: str = "word2vec-google-news-300"):
        if os.path.exists(model):
            self.w2v = KeyedVectors.load_word2vec_format(model, binary = True)
        else:
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
    emb: 'W2VSentenceEmbedder'
    svc: BaseEstimator
    vector_size: int

    def __init__(self, svc_model: Optional[svm.SVC] = None, emb_model: str | W2VSentenceEmbedder = 'word2vec-google-news-300'):
        if isinstance(emb_model, str):
            self.emb = W2VSentenceEmbedder(emb_model)
        else:
            self.emb = emb_model

        self._is_fitted = False
        if svc_model is not None:
            self.svc = svc_model
            self._is_fitted = True
        else: 
            self.svc = svm.SVC(probability=True)
                
    def fit(self, X: Union[str, List[str]], y: List, **kwargs):
        X_embedded = self.emb(X) 
        self.svc.fit(X_embedded, y, **kwargs)
        self._is_fitted = True
        return self

    def predict(self, sentence: List[str]):
        if not self._is_fitted:
            raise Exception('Model is not fitted yet!')
        
        if not isinstance(sentence, list):
            raise Exception("You have to provide a list even if you want to " \
            "predict only one sentence. Uncomfortable, but it is what it is.")
        
        sentence_embedded = self.emb(sentence)
        return self.svc.predict(sentence_embedded)

    def predict_proba(self, sentence: List[str]):
        if not self._is_fitted:
            raise Exception('Model is not fitted yet!')
        
        if not isinstance(sentence, list):
            raise Exception("You have to provide a list even if you want to " \
            "predict only one sentence. Uncomfortable, but it is what it is.")
        
        sentence_embedded = self.emb(sentence)
        if isinstance(sentence_embedded, np.ndarray) and sentence_embedded.ndim == 1:
            sentence_embedded = sentence_embedded.reshape(1, -1)
        return self.svc.predict_proba(sentence_embedded)
        