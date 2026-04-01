from datasets import load_dataset
from transformers import AutoTokenizer
import json
import re
import os
from tqdm import tqdm
import random

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

print("🔹 Загрузка CSV датасета...")

dataset = load_dataset(
    "csv",
    data_files="lenta-ru-news.csv"
)

data = dataset["train"]

print(f"✅ Датасет загружен. Размер: {len(data)}")

# 🔥 реверс (идем с конца — более свежие новости)
data = data.select(range(len(data) - 1, -1, -1))
print("🔁 Данные развернуты (начинаем с новых новостей)")

# токенизатор
print("🔹 Загрузка токенизатора...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-3-mini-4k-instruct")
print("✅ Токенизатор загружен")

MIN_TOKENS = 300
MAX_TOKENS = 600
TARGET_SIZE = 3500


def split_into_sentences(text):
    return re.split(r'(?<=[.!?])\s+', text)


def clean_text(text):
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def process_text(text):
    sentences = split_into_sentences(text)

    if len(sentences) < 2:
        return None, None

    use_random_start = random.random() < 0.5

    if use_random_start:
        start_idx = random.randint(0, len(sentences) - 1)
    else:
        start_idx = 0

    current = ""

    for sent in sentences[start_idx:]:
        temp = current + " " + sent if current else sent
        token_len = len(tokenizer.encode(temp))

        if token_len <= MAX_TOKENS:
            current = temp
        else:
            break

    if not current:
        return None, None

    token_len = len(tokenizer.encode(current))

    if token_len < MIN_TOKENS:
        return None, None

    return current.strip(), use_random_start


def build_sample(text, title, style="публицистическом"):
    return {
        "messages": [
            {
                "role": "user",
                "content": f"Напиши текст в {style} стиле. Тема: {title}"
            },
            {
                "role": "assistant",
                "content": text.strip()
            }
        ]
    }


# 🔹 Основной цикл
print("🚀 Начинаем обработку новостей...")

result = []
processed = 0
kept = 0
skipped = 0
random_chunks = 0
start_chunks = 0


for idx, item in enumerate(tqdm(data)):
    text = clean_text(item.get("text", ""))
    title = item.get("title", "")

    if not text or not title:
        skipped += 1
        continue

    processed_text, is_random = process_text(text)

    if not processed_text:
        skipped += 1
        continue

    if is_random:
        random_chunks += 1
    else:
        start_chunks += 1

    if not processed_text:
        skipped += 1
        continue

    sample = build_sample(processed_text, title)
    result.append(sample)

    kept += 1
    processed += 1

    # 🔥 лимит
    if len(result) >= TARGET_SIZE:
        print(f"\n🎯 Достигнут лимит {TARGET_SIZE}")
        break

    # логи
    if idx % 5000 == 0 and idx > 0:
        print(f"\n📊 Обработано: {idx}")
        print(f"   Сохранено: {kept}")
        print(f"   Пропущено: {skipped}")

print(f"\n✅ Итог: {len(result)} примеров")

# сохранение
print("💾 Сохраняем датасет...")

with open("news_dataset1.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print("🎉 Готово!")
print("\n📊 Статистика выборки:")
print(f"   Начало статьи: {start_chunks}")
print(f"   Середина статьи: {random_chunks}")