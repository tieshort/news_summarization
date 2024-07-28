import multiprocessing as mp
import re

import nltk
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore

from typing import Iterable


class DataHandler:
    tokenizer = Tokenizer(num_words=20000, oov_token="OOV")

    @staticmethod
    def load_data(
        data_path: str = "data/data.csv", nrows: int = None, index_col: any = 0
    ) -> pd.DataFrame:
        data: pd.DataFrame

        try:
            data = pd.read_csv(data_path, nrows=nrows, index_col=index_col)
        except FileNotFoundError:
            print(f"File not found at {data_path}")
            data = None

        return data

    @staticmethod
    def data_to_csv(content: dict, filename: str = "data") -> None:
        df = pd.DataFrame(content)
        df.to_csv(f"data/{filename}.csv", index=False)

    @staticmethod
    def preprocess_data(text: str,
                        to_lower: bool = False,
                        sentence_tokenize: bool = False,
                        remove_stopwords: bool = False,
                        stem_words: bool = False) -> str:
        try:
            text = re.sub(r'<.*?>', '', text)

            pattern = r'[^\w\s\.\!\?]' if sentence_tokenize else r'[^\w\s]'
            text = re.sub(pattern, '', text)

            if to_lower:
                text = text.lower()

            if sentence_tokenize:
                text = nltk.sent_tokenize(text)
            else:
                text = nltk.word_tokenize(text)

            if remove_stopwords:
                stopwords = set(nltk.corpus.stopwords.words('english'))
                text = [word for word in text if word not in stopwords]

            if stem_words:
                stemmer = nltk.stem.SnowballStemmer('english')
                text = [stemmer.stem(word) for word in text]

            return ' '.join(text)

        except TypeError:
            print('TypeError: Input must be a string')
            print(f'    Input type: {type(text)}')
            return text

    @classmethod
    def to_sequences(cls, texts: Iterable) -> any:
        cls.tokenizer.fit_on_texts(texts)
        sequences = cls.tokenizer.texts_to_sequences(texts)

        maxlen = 10
        data = pad_sequences(sequences, maxlen=maxlen)

        return data

    @classmethod
    def tokenizer_to_json(cls, filename: str = "tokenizer") -> None:
        tokenizer_json = cls.tokenizer.to_json()
        with open(f"data/tokenizer/{filename}.json", "w") as file:
            file.write(tokenizer_json)