import json

# Входной и выходной файлы
input_file = "input.json"
output_file = "output.json"

# Загружаем датасет
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

result = []
counter = 0

for item in data:
    # Собираем текст из 5 предложений
    text = " ".join([
        item.get("sentence1", ""),
        item.get("sentence2", ""),
        item.get("sentence3", ""),
        item.get("sentence4", ""),
        item.get("sentence5", "")
    ]).strip()

    title = item.get("storytitle", "")

    formatted_item = {
        "messages": [
            {
                "role": "user",
                "content": f"Напиши текст в художественном стиле. Тема: {title}"
            },
            {
                "role": "assistant",
                "content": text
            }
        ]
    }

    result.append(formatted_item)
    counter += 1

# Сохраняем результат
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print("Готово! Датасет сохранён в", output_file)
print("количество", counter)