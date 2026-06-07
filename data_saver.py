import numpy as np
import pandas as pd
from usearch.index import Index
from typing import Callable, Union, Literal, List, Any
from tqdm import tqdm
from baseline import SupportModel, W2VSentenceEmbedder
import os
import json
import pathlib

def _infer_embedder_dim(embedder: Any) -> int:
    """
    Универсальный способ узнать размерность вектора.
    Проверяет стандартные атрибуты -> если нет, запускает тестовый инференс.
    """
    if hasattr(embedder, 'vector_size'):
        return int(embedder.vector_size)          # gensim
    if hasattr(embedder, 'get_sentence_embedding_dimension'):
        return int(embedder.get_sentence_embedding_dimension())  # sentence-transformers
    if hasattr(embedder, 'dim'):
        return int(embedder.dim)                  # кастомные классы

    try:
        dummy_vec = embedder("test_inference")
        if isinstance(dummy_vec, np.ndarray) and dummy_vec.ndim == 1:
            return int(dummy_vec.shape[0])
    except Exception as e:
        raise RuntimeError(f"Не удалось вывести размерность эмбеддера. Убедитесь, что он возвращает 1D numpy.ndarray. Ошибка: {e}")

    raise ValueError("Эмбеддер должен возвращать 1D numpy.ndarray или иметь атрибут .dim/.vector_size")
    
class DataStorage:
    def __init__(
            self,
            root_dir: str,
            meta_path: str = "metadata.parquet",
            index_path: str = "vectors.usearch",
            metric: Union[Literal["cos", "l2", "ip"], Callable] = "cos",
            embedder: Callable = None,
            batch_size: int = 64,
            connectivity: int = 16,
            expansion_add: int = 128,
            expansion_search: int = 64,
            support_model : SupportModel = None
            ):
        """
        Инициализирует хранилище данных.
        Загружает существующую таблицу метаданных и векторный индекс с диска
        или создаёт их с нуля. Автоматически определяет размерность эмбеддинга
        и настраивает параметры графа Usearch.
        """
        self.root_dir = root_dir
        self.meta_path = meta_path
        self.index_path = index_path
        self.batch_size = batch_size

        if embedder is None:
            self.embedder = W2VSentenceEmbedder()
        else:
            self.embedder = embedder

        self.dim = _infer_embedder_dim(self.embedder)

        if os.path.exists(self.meta_path):
            self.df = pd.read_parquet(self.meta_path)
        else:
            self.df = pd.DataFrame(columns=[
                "usearch_uid", "uid", "file_path", "style", "emotion", "caption",
                "brightness", "colorfulness", "hue", "duration",
                "status"
            ])

        valid_ids = self.df["usearch_uid"].dropna()
        self._next_id = int(valid_ids.max()) + 1 if len(valid_ids) > 0 else 0

        if os.path.exists(self.index_path):
            self.index = Index()              
            self.index.load(self.index_path)
            if self.index.ndim != self.dim:
                raise ValueError(
                    f"Индекс имеет dim={self.index.ndim}, но pipeline ожидает dim={self.dim}"
                )
        else:
            self.index = Index(
                ndim=self.dim,
                metric=metric,
                connectivity=connectivity,
                expansion_add=expansion_add,
                expansion_search=expansion_search
            )

        if support_model is None:
            self.support_model = SupportModel()
        else:
            self.support_model = support_model 

    def scan_new(self):
        """
        Рекурсивно сканирует целевую директорию на наличие JSON-файлов.
        Сравнивает найденные файлы с уже обработанными (по uid) и добавляет
        только новые записи в метаданные со статусом 'pending'.
        Возвращает: int — количество успешно добавленных новых записей.
        """
        existing_uids = set(self.df['uid'])
        new_records = []

        json_paths = list(pathlib.Path(self.root_dir).rglob("*.json"))
        for p in tqdm(json_paths, desc="Сканирование"):
            try:
                with open(p, 'r', encoding="utf-8") as f:
                    raw = json.load(f)
                
                data = {k.strip(): v for k, v in raw.items()}

                uid = str(data.get("video_id", p.stem)).strip()

                if uid in existing_uids:
                    continue

                new_records.append({
                    "usearch_uid": None,
                    "uid": uid,
                    "file_path": str(p),
                    "style": data.get("style"),
                    "emotion": data.get("emotion"),
                    "caption": data.get("caption"),
                    "brightness": data.get("brightness"),
                    "colorfulness": data.get("colorfulness"),
                    "hue": data.get("hue"),
                    "duration": data.get("duration"),
                    "status": "pending"
                })

            except Exception as e:
                print(f"Ошибка {p}: {e}")

        if new_records:
            start_id = self._next_id
            for i, rec in enumerate(new_records):
                rec["usearch_uid"] = start_id + i
                self._next_id += 1

            new_df = pd.DataFrame(new_records)
            self.df = pd.concat([self.df, new_df], ignore_index=True)
            self._save_meta()
            print(f"+{len(new_records)} новых записей в метаданных")
            return len(new_records)
        
        return 0


    def embed_pending(self):
        """
        Генерирует векторные представления для всех записей со статусом 'pending'.
        Добавляет полученные векторы в индекс Usearch батчами и обновляет статус
        записей на 'ready'. Сохраняет изменения на диск.
        Возвращает: int — количество обработанных записей.
        """
        pending_mask = self.df["status"] == "pending"

        if not pending_mask.any():
            return 0
        
        pending_df = self.df[pending_mask].copy()
        uids = pending_df["usearch_uid"].tolist()
        texts = pending_df["caption"].tolist()

        for i in tqdm(range(0, len(texts), self.batch_size), desc="Эмбеддинг"):
            batch_uids = uids[i:i+self.batch_size]
            batch_texts = texts[i:i+self.batch_size]
            
            vectors = np.array([self.embedder(t) for t in batch_texts])
            
            self.index.add(batch_uids, vectors)
            self.df.loc[self.df["usearch_uid"].isin(batch_uids), "status"] = "ready"

        self._save_all()
        return len(pending_df)


    def search(self, prompt: str, K : int):
        """
        Выполняет поиск похожих записей по текстовому запросу.
        Преобразует запрос в вектор, ищет ближайшие соседи в индексе
        и возвращает результат в виде отфильтрованного DataFrame.
        K - количество возвращаемых результатов.
        """
        X = self.embedder(prompt)

        probs = self.support_model.predict_proba([prompt])
        top_3_probs = np.argsort(probs)[-3:][::-1]
        expected_classes = self.support_model.svc.classes_[top_3_probs]

        table = pd.read_parquet(self.meta_path)
        table = table[table['emotion'].isin(expected_classes)]

        index = Index()
        index.load(self.index_path)
        idxs = np.setdiff1d(
            np.array(list(index.keys), dtype = np.uint64),
            table['usearch_uid'].to_numpy(dtype = np.uint64)
        )
        index.remove(idxs) 

        output = index.search(X, count = K)
        return output


    def _save_meta(self):
        """
        Сохраняет DataFrame с метаданными в Parquet-файл с ZSTD-сжатием.
        """
        self.df.to_parquet(self.meta_path, index=False, compression="zstd")

    def _save_all(self):
        """
        Атомарно сохраняет векторный индекс Usearch и таблицу метаданных на диск.
        """
        self.index.save(self.index_path)
        self._save_meta()

    def run_pipeline(self):
        """
        Запускает полный цикл обработки данных:
        1. Сканирование директории на новые файлы.
        2. Генерация эмбеддингов для новых записей.
        3. Выполнение тестового поиска по демонстрационному запросу.
        """
        self.scan_new()
        self.embed_pending()
        print("\nДемо-поиск: 'happy man in blue shirt'")
        print(self.search("happy man in blue shirt", top_k=3))