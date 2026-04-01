from datasets import load_dataset
from transformers import AutoTokenizer
import json
import re
import os
from tqdm import tqdm


print("Загрузка датасета...")

dataset = load_dataset(
    "parquet",
    data_files="data/*.parquet"
)

data = dataset["train"]
print(f" Датасет загружен. Размер: {len(data)}")

# токенизатор
print(" Загрузка токенизатора...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-3-mini-4k-instruct")
print(" Токенизатор загружен")

MIN_TOKENS = 100
MAX_TOKENS = 300
TARGET_SIZE = 8000


def split_into_sentences(text):
    return re.split(r'(?<=[.!?])\s+', text)


def clean_text(text):
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def process_text(text):
    sentences = split_into_sentences(text)

    current = ""

    for sent in sentences:
        temp = current + " " + sent if current else sent
        token_len = len(tokenizer.encode(temp))

        if token_len <= MAX_TOKENS:
            current = temp
        else:
            break  # останавливаемся на последнем полном предложении

    if not current:
        return None

    token_len = len(tokenizer.encode(current))

    if token_len < MIN_TOKENS:
        return None

    return current.strip()


def build_sample(text, title, style="научном"):
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


#  Основной цикл
print(" Начинаем обработку...")

result = []
total_chunks = 0
kept_chunks = 0
skipped_texts = 0

for idx, item in enumerate(tqdm(data)):
    text = clean_text(item["text"])
    title = item["title"]

    if not text or not title:
        skipped_texts += 1
        continue

    processed_text = process_text(text)

    if not processed_text:
        skipped_texts += 1
        continue

    sample = build_sample(processed_text, title)
    result.append(sample)

    kept_chunks += 1
    total_chunks += 1

    # проверка лимита
    if len(result) >= TARGET_SIZE:
        print(f"\n Достигнут лимит {TARGET_SIZE}")
        break

    if len(result) >= TARGET_SIZE:
        break

    if idx % 10000 == 0 and idx > 0:
        print(f"\n Обработано: {idx}")
        print(f"   Всего чанков: {total_chunks}")
        print(f"   Сохранено: {kept_chunks}")
        print(f"   Пропущено текстов: {skipped_texts}")

# ограничение
print("🔹 Ограничиваем до 10k...")
result = result[:10000]

print(f" Финальный размер: {len(result)}")

# сохранение
print("Сохраняем датасет...")

with open("processed_dataset3.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(" Готово!")
print(f" Итоговый датасет: {len(result)} samples")
print(f" Всего чанков обработано: {total_chunks}")