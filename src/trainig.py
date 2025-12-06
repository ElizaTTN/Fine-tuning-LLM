import json
import torch
import os
import time

from unsloth import FastLanguageModel
from datasets import Dataset
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

try:
    with open("dataset.json", "r", encoding="utf-8") as f:
        file_data = json.load(f)
    print("Данные успешно загружены.")
    if len(file_data) > 1:
        print(f"Пример данных (второй элемент): {file_data[1]}")
    else:
        print("В файле dataset.json меньше двух элементов.")

except FileNotFoundError:
    print("Ошибка: Файл 'dataset.json' не найден.")
    exit()
except json.JSONDecodeError:
    print("Ошибка: Неверный формат JSON в 'dataset.json'.")
    exit()

print("-" * 30)
print(f"CUDA доступна: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Используемый GPU: {torch.cuda.get_device_name(0)}")
else:
    print("GPU не обнаружено. Обучение будет медленным.")
print("-" * 30)

model_name = "unsloth/Phi-3-mini-4k-instruct-bnb-4bit"
max_seq_length = 2048
dtype = None

print(f"Загрузка модели {model_name}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=True,
)
print("Модель и токенизатор успешно загружены.")



def format_prompt(example):
    return f"### Input: {example['input']}\n### Output: {example['output']['answer']}<|endoftext|>"


formatted_data = [format_prompt(item) for item in file_data]
full_dataset = Dataset.from_dict({"text": formatted_data})

print(f"Общее количество примеров: {len(full_dataset)}")
print(f"Пример отформатированного текста: \n{formatted_data[0][:200]}...")


def pre_tokenize_dataset(dataset, tokenizer, max_seq_length):
    print("Начинаем предварительную токенизацию...")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=max_seq_length,
            return_tensors=None,
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=os.cpu_count() // 2 if os.cpu_count() > 1 else 1,
        desc="Токенизация данных"
    )
    return tokenized_dataset


# Токенизируем данные
tokenized_full_dataset = pre_tokenize_dataset(full_dataset, tokenizer, max_seq_length)
print("Предварительная токенизация завершена.")

train_test_split = tokenized_full_dataset.train_test_split(test_size=0.05, seed=3407)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]
print(f"Размер обучающего набора: {len(train_dataset)}")
print(f"Размер валидационного набора: {len(eval_dataset)}")

print("Добавление LoRA адаптеров...")
model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", ],
    lora_alpha=128,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)
print("LoRA адаптеры добавлены.")

output_dir = "outputs"

training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    warmup_steps=10,
    num_train_epochs=2,
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir=output_dir,

    logging_steps=20,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    

    eval_strategy="steps",
    eval_steps=20,
    load_best_model_at_end=True,
    
    report_to="tensorboard",

    dataloader_pin_memory=False,
    dataloader_num_workers=0,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    max_seq_length=max_seq_length,
    dataset_num_proc=1,
    args=training_args,
)

print("Trainer успешно создан! Начинаем обучение...")
start_time = time.time()
trainer_stats = trainer.train()
end_time = time.time()
duration = end_time - start_time
print("Обучение завершено.")
print(f"Общее время обучения: {duration:.2f} секунд")

print("-" * 30)
print("Тестирование тонко-настроенной модели...")
FastLanguageModel.for_inference(model)

test_input = "Напиши увлекательную статью на тему: 5 ошибок в уходе за кожей, которые делают все"

messages = [
    {"role": "user", "content": test_input},
]

device = "cuda" if torch.cuda.is_available() else "cpu"
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to(device)

outputs = model.generate(
    input_ids=inputs,
    max_new_tokens=512,
    use_cache=True,
    temperature=0.7,
    do_sample=True,
    top_p=0.9,
    top_k=50,
)

response = tokenizer.batch_decode(outputs)[0]
print("--- Сгенерированный запрос ---")
print(test_input)
print("--- Сгенерированный ответ ---")
print(response)
print("-" * 30)

print("Сохранение модели в формате GGUF...")
model.save_pretrained_gguf("gguf_model", tokenizer, quantization_method="q4_k_m")
print("Модель сохранена в 'gguf_model'.")
