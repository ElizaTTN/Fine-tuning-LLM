from transformers import AutoTokenizer
import json
import re
import os
from tqdm import tqdm

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 📁 путь к папке с авторами
DATA_PATH = "MyData"

print("🔹 Загрузка токенизатора...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-3-mini-4k-instruct")
print("✅ Токенизатор загружен")

MIN_TOKENS = 100
MAX_TOKENS = 300
TARGET_SIZE = 3500


def split_into_sentences(text):
    return re.split(r'(?<=[.!?])\s+', text)


def clean_text(text):
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text):
    """
    🔥 Главное отличие:
    - разбиваем ВЕСЬ текст на чанки
    - двигаемся по предложениям
    """
    sentences = split_into_sentences(text)

    chunks = []
    current = ""

    for sent in sentences:
        temp = current + " " + sent if current else sent
        token_len = len(tokenizer.encode(temp))

        if token_len <= MAX_TOKENS:
            current = temp
        else:
            # сохраняем текущий чанк
            if current:
                chunks.append(current.strip())
            current = sent

    if current:
        chunks.append(current.strip())

    # фильтрация по MIN_TOKENS
    valid_chunks = []
    for chunk in chunks:
        token_len = len(tokenizer.encode(chunk))
        if token_len >= MIN_TOKENS:
            valid_chunks.append(chunk)

    return valid_chunks


def build_sample(text, style="художественном"):
    return {
        "messages": [
            {
                "role": "user",
                "content": f"Напиши текст в {style} стиле. Тема: "
            },
            {
                "role": "assistant",
                "content": text.strip()
            }
        ]
    }


# 🔥 Сбор всех файлов
print("🔹 Сканируем папки с авторами...")

all_files = []

for root, dirs, files in os.walk(DATA_PATH):
    for file in files:
        if file.lower().endswith(".txt") and file.lower() != "info.txt":
            full_path = os.path.join(root, file)
            all_files.append(full_path)

print(f"✅ Найдено файлов: {len(all_files)}")


# 🔹 Основной цикл
print("🚀 Начинаем обработку литературы...")

result = []
processed_files = 0
total_chunks = 0
kept_chunks = 0

for filepath in tqdm(all_files):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        print(f"⚠️ Ошибка чтения {filepath}: {e}")
        continue

    text = clean_text(text)

    if not text:
        continue

    chunks = chunk_text(text)

    total_chunks += len(chunks)

    for chunk in chunks:
        sample = build_sample(chunk)
        result.append(sample)
        kept_chunks += 1

        # 🔥 лимит
        if len(result) >= TARGET_SIZE:
            print(f"\n🎯 Достигнут лимит {TARGET_SIZE}")
            break

    processed_files += 1

    # лог каждые N файлов
    if processed_files % 50 == 0:
        print(f"\n📊 Обработано файлов: {processed_files}")
        print(f"   Всего чанков: {total_chunks}")
        print(f"   Сохранено: {kept_chunks}")

    if len(result) >= TARGET_SIZE:
        break


print(f"\n✅ Итог: {len(result)} примеров")

# 🔹 сохранение
print("💾 Сохраняем датасет...")

with open("literature_dataset3.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print("🎉 Готово!")
print("\n📊 Финальная статистика:")
print(f"   Файлов обработано: {processed_files}")
print(f"   Всего чанков: {total_chunks}")
print(f"   Сохранено: {kept_chunks}")