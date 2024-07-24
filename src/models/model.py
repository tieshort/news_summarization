import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore

class Summarization_model(Model):
    def __init__(self):
        # Параметры модели
        vocab_size = 20000  # Размер словаря
        embedding_dim = 256  # Размерность эмбеддингов
        units = 512  # Количество юнитов в LSTM

        # Энкодер
        encoder_inputs = Input(shape=(None,), name='encoder_inputs')
        encoder_embedding = Embedding(vocab_size, embedding_dim, name='encoder_embedding')(encoder_inputs)
        encoder_lstm = LSTM(units, return_state=True, name='encoder_lstm')
        encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)

        # Сохранение состояний энкодера
        encoder_states = [state_h, state_c]

        # Декодер
        decoder_inputs = Input(shape=(None,), name='decoder_inputs')
        decoder_embedding = Embedding(vocab_size, embedding_dim, name='decoder_embedding')(decoder_inputs)
        decoder_lstm = LSTM(units, return_sequences=True, return_state=True, name='decoder_lstm')
        decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
        decoder_dense = Dense(vocab_size, activation='softmax', name='decoder_dense')
        decoder_outputs = decoder_dense(decoder_outputs)

        super().__init__([encoder_inputs, decoder_inputs], decoder_outputs)