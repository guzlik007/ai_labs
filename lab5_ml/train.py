import pandas as pd
import re
import numpy as np
from bs4 import BeautifulSoup
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Attention, Concatenate
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import pickle

# Константы
VOCAB_SIZE = 25000
MAX_LEN = 50
HIDDEN = 256
EMBED = 256


def clean(text):
    if not isinstance(text, str):
        return ""
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'Пользователь \d+:', '', text).lower()
    text = re.sub(r'[^а-яёa-z0-9\s.,!?]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def get_messages(html):
    messages = []
    soup = BeautifulSoup(html, "html.parser")
    for span in soup.find_all("span"):
        cls = span.get("class", [])
        speaker = "bot" if "participant_1" in cls else "user" if "participant_2" in cls else None
        if speaker:
            text = clean(span.get_text())
            if 2 < len(text.split()) < 100:
                messages.append((speaker, text))
    return messages


def load_data(path, max_pairs=50000):
    df = pd.read_csv(path, sep="\t")
    questions, answers = [], []

    for _, row in df.iterrows():
        msgs = get_messages(row["dialogue"])
        for i in range(len(msgs) - 1):
            if msgs[i][0] == "user" and msgs[i + 1][0] == "bot":
                questions.append(msgs[i][1])
                answers.append(msgs[i + 1][1])
                if len(questions) >= max_pairs:
                    break
        if len(questions) >= max_pairs:
            break

    print(f"Собрано пар: {len(questions)}")
    return questions, answers


def prepare(questions, answers):
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<unk>')
    answers = ['<start> ' + a + ' <end>' for a in answers]
    tokenizer.fit_on_texts(questions + answers)

    enc_in = pad_sequences(tokenizer.texts_to_sequences(questions), maxlen=MAX_LEN, padding='post')
    dec_in = pad_sequences(tokenizer.texts_to_sequences(answers), maxlen=MAX_LEN, padding='post')

    dec_out = np.zeros_like(dec_in)
    dec_out[:, :-1] = dec_in[:, 1:]
    dec_out = np.expand_dims(dec_out, -1)

    vocab = min(len(tokenizer.word_index) + 1, VOCAB_SIZE)
    return enc_in, dec_in, dec_out, tokenizer, vocab


def build_model(vocab):
    # Encoder с Attention
    enc_in = Input(shape=(MAX_LEN,), name='enc_in')
    enc_emb = Embedding(vocab, EMBED, mask_zero=True)(enc_in)
    enc_emb = Dropout(0.5)(enc_emb)
    enc_lstm = LSTM(HIDDEN, return_sequences=True, return_state=True, recurrent_dropout=0.2, name='enc_lstm')
    enc_out, h, c = enc_lstm(enc_emb)

    # Decoder с Attention
    dec_in = Input(shape=(MAX_LEN,), name='dec_in')
    dec_emb = Embedding(vocab, EMBED, mask_zero=True, name='dec_emb')(dec_in)
    dec_emb = Dropout(0.5)(dec_emb)
    dec_lstm = LSTM(HIDDEN, return_sequences=True, return_state=True, recurrent_dropout=0.2, name='dec_lstm')
    dec_out, _, _ = dec_lstm(dec_emb, initial_state=[h, c])

    # Attention слой
    attention = Attention(name='attention')
    context = attention([dec_out, enc_out])

    # Объединяем decoder output с context
    concat = Concatenate()([dec_out, context])
    concat = Dropout(0.5)(concat)

    # Финальный слой
    output = Dense(vocab, activation='softmax', name='dec_dense')(concat)

    model = Model([enc_in, dec_in], output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model, enc_in, enc_out, [h, c], dec_in, dec_lstm


# Основной код
print("Загрузка данных...")
questions, answers = load_data("dialogues.tsv")

print("Подготовка данных...")
enc_in, dec_in, dec_out, tokenizer, vocab = prepare(questions, answers)

print("Создание модели...")
model, encoder_in, encoder_out, encoder_states, decoder_in, decoder_lstm = build_model(vocab)
model.summary()

print("Обучение...")
model.fit(
    [enc_in, dec_in], dec_out,
    batch_size=32, epochs=50,
    validation_split=0.2,
    callbacks=[
        ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', verbose=1),
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)
    ]
)

print("Сохранение...")
model.save('model.keras')

# Сохраняем encoder (теперь возвращает sequences)
encoder_model = Model(encoder_in, [encoder_out] + encoder_states)
encoder_model.save('encoder.keras')

# Сохраняем decoder с attention
enc_out_input = Input(shape=(MAX_LEN, HIDDEN))
state_h = Input(shape=(HIDDEN,))
state_c = Input(shape=(HIDDEN,))
dec_in_inf = Input(shape=(1,))

dec_emb_layer = model.get_layer('dec_emb')
dec_lstm_layer = model.get_layer('dec_lstm')
attention_layer = model.get_layer('attention')
dec_dense_layer = model.get_layer('dec_dense')

emb = dec_emb_layer(dec_in_inf)
out, h, c = dec_lstm_layer(emb, initial_state=[state_h, state_c])

# Attention
context = attention_layer([out, enc_out_input])
concat = Concatenate()([out, context])

output = dec_dense_layer(concat)

decoder_model = Model([dec_in_inf, enc_out_input, state_h, state_c], [output, h, c])
decoder_model.save('decoder.keras')

with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

with open('config.pkl', 'wb') as f:
    pickle.dump({'vocab': vocab, 'max_len': MAX_LEN, 'hidden': HIDDEN}, f)

print("✓ Готово! Модель с Attention сохранена!")