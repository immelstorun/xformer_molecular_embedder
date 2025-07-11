import torch
from torch import nn
from torch.optim import AdamW
import random
import math
import xformers.ops
 
# --- 1. Данные и Токенизатор ---
# Наш "игрушечный" мир состоит всего из 5 молекул
SMILES_DATA = [
    "CCO",      # Этанол
    "COC",      # Его изомер, диметиловый эфир
    "c1ccccc1", # Бензол
    "CCC",      # Пропан
    "CC(=O)O", # Уксусная кислота
]

print("\n Готовим токенизатор ---")
unique_chars = sorted(list(set("".join(SMILES_DATA))))
SPECIAL_TOKENS = ["[PAD]", "[MASK]"]
VOCAB = SPECIAL_TOKENS + unique_chars
VOCAB_SIZE = len(VOCAB)

# Словари 
char_to_id = {char: i for i, char in enumerate(VOCAB)}
id_to_char = {i: char for i, char in enumerate(VOCAB)}

print(f"Размер словаря: {VOCAB_SIZE}")
print(f"Словарь: {VOCAB}")

# demo model
class MolecularEmbedder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 32, n_layers: int = 2, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        # номер токена -> начальный эмбеддинг
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # 1. Создаем один слой-шаблон. PyTorch автоматически использует xFormers, если он установлен.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4, # Стандартная практика
            dropout=dropout,
            activation="relu",
            batch_first=True # Важно для работы с формой [batch, seq_len, dim]
        )
        
        # 2. Создаем стандартный энкодер PyTorch, который использует
        #    наш слой-шаблон для построения всех n_layers.
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layers
        )
        # -----------------------------------------------

        # Линейный слой для предсказания замаскированного токена
        self.predictor = nn.Linear(d_model, vocab_size)
        print("\n--- 2. Модель создана (с исправленным API) ---")

    def forward(self, token_ids):
        # 1. Превращаем номера в начальные векторы
        x = self.token_embedding(token_ids)
        # 2. Обрабатываем энкодером
        encoded_vectors = self.encoder(x)
        # 3. Делаем предсказание для каждого токена
        logits = self.predictor(encoded_vectors)
        # Возвращаем и предсказания (для обучения), и финальные векторы (наша цель)
        return logits, encoded_vectors

# --- 3. Цикл Обучения ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Используемое устройство: {device}")

# Инициализируем модель
model = MolecularEmbedder(vocab_size=VOCAB_SIZE).to(device)
optimizer = AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

print("\n--- 3. Начинаем обучение... ---")
TRAINING_STEPS = 501
for step in range(TRAINING_STEPS):
    model.train()

    smiles = random.choice(SMILES_DATA)
    tokens = list(smiles)

    mask_position = random.randint(0, len(tokens) - 1)
    
    original_token = tokens[mask_position]
    tokens[mask_position] = "[MASK]"

    input_ids = torch.tensor([[char_to_id[token] for token in tokens]], device=device)
    correct_label_id = torch.tensor([char_to_id[original_token]], device=device)

    logits, _ = model(input_ids)
    logits_for_masked_token = logits[0, mask_position]
    loss = loss_fn(logits_for_masked_token.unsqueeze(0), correct_label_id)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        predicted_id = torch.argmax(logits_for_masked_token).item()
        print(f"Шаг {step:3d} | Ошибка: {loss.item():.4f} | {''.join(tokens):<10} | Угадываем: '{id_to_char[predicted_id]}' (Правильно: '{original_token}')")

# --- 4. Получение Эмбеддингов ---
print("\n--- 4. Обучение завершено. Получаем финальные эмбеддинги ---")
model.eval()

all_embeddings = {}
with torch.no_grad():
    for smiles in SMILES_DATA:
        input_ids = torch.tensor([[char_to_id[char] for char in smiles]], device=device)
        _, token_vectors = model(input_ids)
        molecule_embedding = torch.mean(token_vectors, dim=1)
        all_embeddings[smiles] = molecule_embedding
        
        print(f"\nМолекула: {smiles}")
        print(f"Её эмбеддинг (первые 8 из 32 чисел): {molecule_embedding.cpu().numpy()[0, :8]}")

sim_isomers = nn.functional.cosine_similarity(all_embeddings['CCO'], all_embeddings['COC']).item()
sim_different = nn.functional.cosine_similarity(all_embeddings['CCO'], all_embeddings['c1ccccc1']).item()

print("\n--- Результаты ---")
print(f"Сходство векторов CCO и COC (изомеры): {sim_isomers:.4f}")
print(f"Сходство векторов CCO и c1ccccc1 (разные молекулы): {sim_different:.4f}")
if sim_isomers > sim_different:
    print("Успех! Модель считает изомеры более похожими, чем химически разные молекулы.")
else:
    print("Модель не до конца обучилась, но это нормально для прототипа.")