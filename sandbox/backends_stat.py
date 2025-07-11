import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel
import logging
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Настройка ---
logging.basicConfig(level=logging.WARNING) 
# logging.getLogger("torch.nn.attention").setLevel(logging.INFO) # для отладки

# --- Информация о GPU ---
print("="*60)
print("--- Информация о системе и GPU ---")
if torch.cuda.is_available():
    device = "cuda"
    torch.cuda.set_device(0)
    current_device_idx = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(current_device_idx)
    print(f"Тест будет проводиться на GPU {current_device_idx}: {gpu_name}")
else:
    device = "cpu"
    gpu_name = "CPU"
    print("CUDA не найдена. Тест будет проводиться на CPU.")
print("="*60)

# --- 2. модель ---
class TransEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, dim_feedforward):
        super().__init__()
        self.emb = nn.Embedding(
            vocab_size,
            d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=dim_feedforward,
            batch_first=True, 
            dropout=0.0
        )
        self.transformer_enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

    def forward(self, src):
        x = self.emb(src) # tok to vec
        output = self.transformer_enc(x)
        return output

# --- 3. бенч ---
def run_model_benchmark(model, input_data, n_repeats=50, n_warmup=5):
    for _ in range(n_warmup):
        _ = model(input_data)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    with torch.no_grad():
        for _ in range(n_repeats):
            _ = model(input_data)
    end_event.record()
    
    torch.cuda.synchronize()
    
    elapsed_time_ms = start_event.elapsed_time(end_event)
    return elapsed_time_ms / n_repeats

# --- 4. Параметры перебора ---
VOCAB_SIZE = 1000
BATCH_SIZE = 64 
N_LAYERS = 6    
dtype = torch.float16

# для итерации
SEQ_LENGTHS = [128, 512, 1024, 2048]
D_MODELS = [256, 512, 768]
N_HEADS_LIST = [4, 8, 16, 32]

# Бэкенды
BACKENDS = {
    "MATH": SDPBackend.MATH,
    "xFormers": SDPBackend.EFFICIENT_ATTENTION,
    "FlashAttention": SDPBackend.FLASH_ATTENTION
}

all_results = []

# --- 5. Главный цикл перебора и тестирования ---

print("\n--- Тест 1: Зависимость от длины последовательности (SEQ_LENGTH) ---")
D_MODEL_FIXED = 512
N_HEADS_FIXED = 8
for seq_len in SEQ_LENGTHS:
    print(f"\nТестируем SEQ_LENGTH = {seq_len}...")
    try:
        model = TransEncoder(VOCAB_SIZE, D_MODEL_FIXED, N_HEADS_FIXED, N_LAYERS, D_MODEL_FIXED * 4).to(device, dtype)
        model.eval()
        input_data = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, seq_len), device=device)
        
        for name, backend in BACKENDS.items():
            try:
                with sdpa_kernel(backend):
                    timing = run_model_benchmark(model, input_data)
                    all_results.append({"param_type": "SEQ_LENGTH", "param_value": seq_len, "backend": name, "time_ms": timing})
                    print(f"  - {name}: {timing:.4f} мс")
            except RuntimeError:
                print(f"  - {name}: Недоступен")
    except torch.cuda.OutOfMemoryError:
        print(f"  - Ошибка нехватки памяти. Пропускаем...")
        break

print("\n--- Тест 2: Зависимость от размерности модели (D_MODEL) ---")
SEQ_LENGTH_FIXED = 1024
for d_model in D_MODELS:
    # Убедимся, что d_model делится на n_heads
    if d_model % N_HEADS_FIXED != 0: continue
    print(f"\nТестируем D_MODEL = {d_model}...")
    try:
        model = TransEncoder(VOCAB_SIZE, d_model, N_HEADS_FIXED, N_LAYERS, d_model * 4).to(device, dtype)
        model.eval()
        input_data = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LENGTH_FIXED), device=device)
        
        for name, backend in BACKENDS.items():
            try:
                with sdpa_kernel(backend):
                    timing = run_model_benchmark(model, input_data)
                    all_results.append({"param_type": "D_MODEL", "param_value": d_model, "backend": name, "time_ms": timing})
                    print(f"  - {name}: {timing:.4f} мс")
            except RuntimeError:
                print(f"  - {name}: Недоступен")
    except torch.cuda.OutOfMemoryError:
        print(f"  - Ошибка нехватки памяти. Пропускаем...")
        break

print("\n--- Тест 3: Зависимость от количества голов (N_HEADS) ---")
D_MODEL_FIXED = 512
for n_heads in N_HEADS_LIST:
    if D_MODEL_FIXED % n_heads != 0: continue
    print(f"\nТестируем N_HEADS = {n_heads}...")
    try:
        model = TransEncoder(VOCAB_SIZE, D_MODEL_FIXED, n_heads, N_LAYERS, D_MODEL_FIXED * 4).to(device, dtype)
        model.eval()
        input_data = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LENGTH_FIXED), device=device)
        
        for name, backend in BACKENDS.items():
            try:
                with sdpa_kernel(backend):
                    timing = run_model_benchmark(model, input_data)
                    all_results.append({"param_type": "N_HEADS", "param_value": n_heads, "backend": name, "time_ms": timing})
                    print(f"  - {name}: {timing:.4f} мс")
            except RuntimeError:
                print(f"  - {name}: Недоступен")
    except torch.cuda.OutOfMemoryError:
        print(f"  - Ошибка нехватки памяти. Пропускаем...")
        break

# --- 6. Обработка результатов и построение графиков ---
print("\n--- Построение графиков ---")
df = pd.DataFrame(all_results)

# Создаем фигуру с 3 графиками
fig, axes = plt.subplots(1, 3, figsize=(24, 7), sharey=False)
fig.suptitle(f'Производительность Attention бэкендов на {gpu_name}', fontsize=16)

# Функция для построения одного графика
def plot_results(ax, param_type, title):
    subset_df = df[df['param_type'] == param_type]
    if subset_df.empty:
        ax.text(0.5, 0.5, 'Нет данных для этого теста', ha='center', va='center')
        ax.set_title(title)
        return

    sns.lineplot(data=subset_df, x="param_value", y="time_ms", hue="backend", marker="o", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(param_type)
    ax.set_ylabel("Среднее время (мс)")
    ax.grid(True)
    ax.legend(title="Бэкенд")

plot_results(axes[0], "SEQ_LENGTH", "Зависимость времени от длины последовательности")
plot_results(axes[1], "D_MODEL", "Зависимость времени от размерности модели")
plot_results(axes[2], "N_HEADS", "Зависимость времени от количества голов")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("backends_stat.py.png")
print("Графики сохранены в файл 'backends_stat.py.png'")
# plt.show() # если в среде с GUI (Jupyter, VSCode)