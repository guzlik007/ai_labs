import numpy as np
import pickle
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


def clean(text: str) -> str:
    """Предобработка входного текста пользователя."""
    text = text.lower()
    text = re.sub(r'[^а-яёa-z0-9\s.,!?]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def postprocess_response(text: str) -> str:
    """
    Чистим ответ модели:
    - убираем служебные токены (<start>, <end>, <unk>, любые <...>)
    - убираем возможное 'unk' и 'end'
    - чистим лишние пробелы
    """
    # Убираем явные спецтокены
    text = text.replace('<start>', ' ')
    text = text.replace('<end>', ' ')
    text = text.replace('<unk>', ' ')

    # Убираем любые штуки в угловых скобках типа <что_угодно>
    text = re.sub(r'<[^>]+>', ' ', text)

    # На всякий случай убираем одиночные "unk" и "end"
    text = re.sub(r'\bunk\b', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\bend\b', ' ', text, flags=re.IGNORECASE)

    # Чистим пробелы
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def beam_search(text, encoder, decoder, tokenizer, config, beam_width: int = 5, length_penalty: float = 0.7) -> str:
    """
    Генерация ответа с помощью beam search.
    length_penalty < 1.0 — поощряем чуть более длинные ответы.
    """
    # Кодируем вопрос
    seq = tokenizer.texts_to_sequences([clean(text)])
    seq = pad_sequences(seq, maxlen=config['max_len'], padding='post')

    # Encoder: enc_out для attention + начальные состояния h, c
    enc_out, h, c = encoder.predict(seq, verbose=0)

    # Индексы спецтокенов
    start_token = tokenizer.word_index.get('<start>', 1)
    end_token = tokenizer.word_index.get('<end>', 2)
    end_word_idx = tokenizer.word_index.get('end')
    unk_idx = tokenizer.word_index.get('<unk>')

    # Индекс → слово
    index_word = {i: w for w, i in tokenizer.word_index.items()}

    # Beam: [(последовательность_индексов, score, (h, c))]
    beams = [([start_token], 0.0, (h, c))]
    completed = []

    max_len = config['max_len']

    for _ in range(max_len):
        new_beams = []

        for seq_tokens, score, (st_h, st_c) in beams:
            last_idx = seq_tokens[-1]

            # Если уже дошли до конца по индексу — считаем луч завершённым
            if last_idx == end_token or (end_word_idx is not None and last_idx == end_word_idx):
                completed.append((seq_tokens, score))
                continue

            target = np.array([[last_idx]])
            preds, new_h, new_c = decoder.predict([target, enc_out, st_h, st_c], verbose=0)

            preds = preds[0, -1, :]

            # Берём top-k токенов
            top_k = np.argsort(preds)[-beam_width:]

            for idx in top_k:
                # Не расширяем <unk>-ветки по индексу
                if unk_idx is not None and idx == unk_idx:
                    continue

                # Игнорируем нулевой токен
                if idx == 0:
                    continue

                new_seq = seq_tokens + [idx]
                # Чем меньше score, тем лучше → используем -log p
                new_score = score - np.log(preds[idx] + 1e-10)
                new_beams.append((new_seq, new_score, (new_h, new_c)))

        if not new_beams:
            break

        # Оставляем top beam_width лучей по текущему score
        beams = sorted(new_beams, key=lambda x: x[1])[:beam_width]

    # Если есть завершённые лучи — выбираем лучший с учётом length penalty
    def norm_score(item):
        seq_tokens, score = item
        length = max(len(seq_tokens), 1)
        # нормализация: делим на length**length_penalty
        return score / (length ** length_penalty)

    if completed:
        best_seq = min(completed, key=lambda x: norm_score((x[0], x[1])))[0]
    else:
        # Иначе берём лучший из незавершённых
        if beams:
            best_seq = beams[0][0]
        else:
            best_seq = [start_token]

    # Декодируем индексы в слова
    result_tokens = []
    for idx in best_seq[1:]:  # пропускаем <start>
        # Стоп по индексу
        if idx == end_token or (end_word_idx is not None and idx == end_word_idx):
            break

        word = index_word.get(idx)
        if not word:
            break

        # Пропускаем явный мусор
        if word in ['<unk>', '<start>', '<end>']:
            continue
        if word.lower() == 'end':
            continue

        result_tokens.append(word)

    raw_text = ' '.join(result_tokens)
    return postprocess_response(raw_text)


# ================== ЗАГРУЗКА МОДЕЛЕЙ И ЗАПУСК ЧАТА ==================

print("Загрузка моделей...")
encoder = load_model('encoder.keras')
decoder = load_model('decoder.keras')

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('config.pkl', 'rb') as f:
    config = pickle.load(f)

print("\n" + "=" * 60)
print("🤖 ЧАТ-БОТ ГОТОВ (beam search по дефолту)!")
print("=" * 60)
print("Напиши что-нибудь.")
print("Команда: 'выход' — завершить.")
print("=" * 60 + "\n")

while True:
    user = input("Вы: ").strip()

    if not user:
        continue

    if user.lower() in ['выход', 'exit', 'quit']:
        print("Пока! 👋")
        break

    try:
        response = beam_search(user, encoder, decoder, tokenizer, config, beam_width=5)
        if not response:
            response = "не понял"
        print(f"Бот: {response}\n")
    except Exception as e:
        print(f"❌ Ошибка: {e}\n")
