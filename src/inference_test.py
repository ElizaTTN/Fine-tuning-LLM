import torch
import json
import os
from unsloth import FastLanguageModel
from datetime import datetime

LORA_ADAPTER_PATH = "/home/alexey/lizacourse/LearningCourseWork/r64/outputs/checkpoint-356"
OUTPUT_FILENAME = "test_results_64r2.json"
MODEL_NAME = "unsloth/Phi-3-mini-4k-instruct-bnb-4bit"
MAX_SEQ_LENGTH = 2048

TEST_PROMPTS = [
    "Напиши эссе о природе творчества и одержимости мастерством на примере великого скрипача, который, вернувшись после триумфального концерта в гостиницу, снова берет в руки инструмент, чтобы бесконечно оттачивать одну и ту же музыкальную фразу в поисках идеала, выполни в художественном стиле",
    "Подготовь аналитический отчет: Влияние экологической катастрофы в Мексиканском заливе на котировки British Petroleum, выполни в публицистическом стиле",
    "Подготовь развернутый обзор для изучения темы: Оценка прироста производительности труда благодаря использованию ИИ, выполни в научном стиле",
]

TEST_CONFIGS = [
    {"temperature": 0.7, "top_p": 0.9, "top_k": 50, "description": "Базовая, сбалансированная"},
    {"temperature": 0.1, "top_p": 0.9, "top_k": 50, "description": "Детерминированный, низкая T"},
    {"temperature": 1.0, "top_p": 0.9, "top_k": 50, "description": "Креативный, высокая T"},
    {"temperature": 0.7, "top_p": 0.7, "top_k": 50, "description": "Сфокусированный, низкий Top-P"},
    {"temperature": 0.7, "top_p": 0.9, "top_k": 20, "description": "Ограниченный словарь, низкий Top-K"},
    {"temperature": 0.5, "top_p": 0.5, "top_k": 80, "description": "Сфокусированный, низкий Top-P"},
    {"temperature": 0.5, "top_p": 0.8, "top_k": 50, "description": "Сфокусированный, низкий Top-P"},

]

try:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    model.load_adapter(LORA_ADAPTER_PATH)
    FastLanguageModel.for_inference(model)

    print(f"Модель успешно загружена с адаптером из: {LORA_ADAPTER_PATH}")
    print(f"Устройство: {device}")

except Exception as e:
    print(f"Ошибка при загрузке модели или адаптера. Проверьте путь ({LORA_ADAPTER_PATH}): {e}")
    exit()



def generate_and_save_results():
    all_results = []

    for prompt_id, prompt in enumerate(TEST_PROMPTS):
        for config_id, config in enumerate(TEST_CONFIGS):

            messages = [{"role": "user", "content": prompt}]
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(device)

            print(
                f"Генерация: Prompt {prompt_id + 1}/{len(TEST_PROMPTS)}, Config {config_id + 1}/{len(TEST_CONFIGS)} ({config['description']})...")

            try:
                outputs = model.generate(
                    input_ids=inputs,
                    max_new_tokens=512,
                    use_cache=True,
                    temperature=config["temperature"],
                    do_sample=True,
                    top_p=config["top_p"],
                    top_k=config["top_k"],
                )


                response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
                generated_text = response.replace(tokenizer.apply_chat_template(messages, tokenize=False), "").strip()

            except Exception as e:
                generated_text = f"ОШИБКА ГЕНЕРАЦИИ: {e}"
                print(f"Ошибка при генерации для конфигурации {config_id}: {e}")

            result_entry = {
                "timestamp": datetime.now().isoformat(),
                "test_description": config["description"],
                "input_prompt": prompt,
                "generation_params": config,
                "generated_text": generated_text,
            }
            all_results.append(result_entry)
            print("Сохранено.")

    with open(OUTPUT_FILENAME, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)

    print(f"\nВСЕ ТЕСТЫ ЗАВЕРШЕНЫ. Результаты сохранены в: {OUTPUT_FILENAME}")


if __name__ == "__main__":
    generate_and_save_results()
