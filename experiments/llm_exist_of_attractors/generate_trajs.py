import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import pickle
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Настройка визуализации
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Проверка доступности GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используется устройство: {device}")

# Загрузка GPT-2 Small (124M параметров)
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.to(device)
model.eval()

# Добавляем pad_token если его нет
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Модель загружена: {model_name}")
print(f"Размер словаря: {len(tokenizer)}")
print(f"Размерность скрытых состояний: {model.config.n_embd}")

# Группа A: Семантически различные промпты
prompts_diverse = [
    "The gravitational constant G is approximately",
    "In a binary tree data structure, each node has",
    "The French Revolution began in the year",
    "To bake a simple sponge cake, first preheat the oven to",
    "Quantum entanglement is a phenomenon where"
]

# Группа B: Близкие вариации
prompts_similar = [
    "The cat sat on the mat",
    "A cat sat on the mat",
    "The cat sat on a mat",
    "A cat sits on the mat",
    "The dog sat on the mat",
    "The cat lay on the mat",
    "On the mat sat the cat"
]

# Группа C: Нейтральные промпты
prompts_neutral = [
    "The",
    ".",
    "Therefore,"
]

all_prompts = {
    'diverse': prompts_diverse,
    'similar': prompts_similar,
    'neutral': prompts_neutral
}

print("Промпты определены:")
for group, prompts in all_prompts.items():
    print(f"  {group}: {len(prompts)} промптов")

def prepare_input(prompt: str, max_length: int = 512) -> torch.Tensor:
    """
    Подготавливает входной промпт, дополняя до фиксированной длины.
    """
    # Токенизация
    tokens = tokenizer.encode(prompt, return_tensors='pt')
    
    # Дополнение или обрезка до max_length
    if tokens.shape[1] < max_length:
        # Дополняем слева pad токенами
        pad_length = max_length - tokens.shape[1]
        pad_tokens = torch.full((1, pad_length), tokenizer.pad_token_id)
        tokens = torch.cat([pad_tokens, tokens], dim=1)
    else:
        # Обрезаем справа
        tokens = tokens[:, -max_length:]
    
    return tokens.to(device)

def extract_hidden_state(input_ids: torch.Tensor) -> torch.Tensor:
    """
    Извлекает скрытое состояние последнего токена последнего слоя.
    """
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        # Берем последний слой, последний токен
        hidden_state = outputs.hidden_states[-1][0, -1, :]
        return hidden_state.cpu().numpy()

def generate_trajectory(prompt: str, num_steps: int = 600) -> Tuple[List[np.ndarray], List[str]]:
    """
    Генерирует траекторию скрытых состояний для заданного промпта.
    """
    # Подготовка начального состояния
    current_input = prepare_input(prompt)
    
    hidden_states = []
    generated_tokens = []
    
    for step in tqdm(range(num_steps), desc=f"Генерация для '{prompt[:30]}...'"):
        # Извлечение скрытого состояния
        hidden_state = extract_hidden_state(current_input)
        hidden_states.append(hidden_state)
        
        # Генерация следующего токена (жадное декодирование)
        with torch.no_grad():
            outputs = model(current_input)
            next_token_id = torch.argmax(outputs.logits[0, -1, :]).unsqueeze(0).unsqueeze(0)
        
        # Декодирование токена для анализа
        next_token = tokenizer.decode(next_token_id[0, 0].item())
        generated_tokens.append(next_token)
        
        # Обновление контекста (сдвиг окна)
        current_input = torch.cat([current_input[:, 1:], next_token_id], dim=1)
    
    return hidden_states, generated_tokens

print("Функции для генерации определены")

# Параметры эксперимента
NUM_STEPS = 600
CONTEXT_LENGTH = 512

# Словарь для хранения результатов
trajectories = {}
generated_texts = {}

print("Начинаем генерацию траекторий...")

for group_name, prompts in all_prompts.items():
    trajectories[group_name] = []
    generated_texts[group_name] = []
    
    print(f"\nОбработка группы: {group_name}")
    
    for i, prompt in enumerate(prompts):
        print(f"Промпт {i+1}/{len(prompts)}: '{prompt}'")
        
        hidden_states, tokens = generate_trajectory(prompt, NUM_STEPS)
        
        trajectories[group_name].append(np.array(hidden_states))
        generated_texts[group_name].append(tokens)
        
        sample_text = ''.join(tokens[:50])
        print(f"  Сгенерированный текст (первые 50 токенов): {sample_text}")

print("\nГенерация траекторий завершена!")

# Сохраняем оба словаря в один файл
data_to_save = {
    'trajectories': trajectories,
    'generated_texts': generated_texts
}

with open('experiment_results.pkl', 'wb') as f:
    pickle.dump(data_to_save, f)

print("Результаты успешно сохранены в 'experiment_results.pkl'")
