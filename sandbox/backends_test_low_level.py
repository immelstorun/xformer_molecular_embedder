import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Настройка ---
logging.basicConfig(level=logging.WARNING)

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

# --- 2. Функция бенчмаркинга (без изменений) ---
def run_sdpa_benchmark(q, k, v, n_repeats=50, n_warmup=5):
    for _ in range(n_warmup):
        _ = F.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    with torch.no_grad():
        for _ in range(n_repeats):
            _ = F.scaled_dot_product_attention(q, k, v)
    end_event.record()
    
    torch.cuda.synchronize()
    
    elapsed_time_ms = start_event.elapsed_time(end_event)
    return elapsed_time_ms / n_repeats

# --- 3. Параметры для перебора (ИДЕНТИЧНЫ ПАРАМЕТРАМ ВЫСОКОУРОВНЕВОГО ТЕСТА) ---
BATCH_SIZE = 16
dtype = torch.float16

# Списки параметров для итерации
SEQ_LENGTHS = [128, 512, 1024]
D_MODELS = [256, 512, 768]
N_HEADS_LIST = [4, 8, 16]

BACKENDS = {
    "MATH": SDPBackend.MATH,
    "xFormers": SDPBackend.EFFICIENT_ATTENTION,
    "FlashAttention": SDPBackend.FLASH_ATTENTION
}

all_results = []

# --- 4. Главный цикл перебора и тестирования ---

print("\n--- Тест 1: Зависимость от длины последовательности (SEQ_LENGTH) ---")
D_MODEL_FIXED = 512
N_HEADS_FIXED = 8
D_HEAD_FIXED = D_MODEL_FIXED // N_HEADS_FIXED
for seq_len in SEQ_LENGTHS:
    print(f"\nТестируем SEQ_LENGTH = {seq_len}...")
    try:
        q = torch.randn(BATCH_SIZE, N_HEADS_FIXED, seq_len, D_HEAD_FIXED, device=device, dtype=dtype)
        k = torch.randn(BATCH_SIZE, N_HEADS_FIXED, seq_len, D_HEAD_FIXED, device=device, dtype=dtype)
        v = torch.randn(BATCH_SIZE, N_HEADS_FIXED, seq_len, D_HEAD_FIXED, device=device, dtype=dtype)
        
        for name, backend in BACKENDS.items():
            try:
                with sdpa_kernel(backend):
                    timing = run_sdpa_benchmark(q, k, v)
                    all_results.append({"param_type": "SEQ_LENGTH", "param_value": seq_len, "backend": name, "time_ms": timing})
                    print(f"  - {name}: {timing:.4f} мс")
            except RuntimeError:
                print(f"  - {name}: Недоступен")
    except torch.cuda.OutOfMemoryError:
        print(f"  - Ошибка нехватки памяти. Пропускаем...")
        break

print("\n--- Тест 2: Зависимость от размерности модели (D_MODEL) ---")
SEQ_LENGTH_FIXED = 1024
N_HEADS_FIXED = 8
for d_model in D_MODELS:
    if d_model % N_HEADS_FIXED != 0: continue
    d_head = d_model // N_HEADS_FIXED
    print(f"\nТестируем D_MODEL = {d_model} (D_HEAD = {d_head})...")
    try:
        q = torch.randn(BATCH_SIZE, N_HEADS_FIXED, SEQ_LENGTH_FIXED, d_head, device=device, dtype=dtype)
        k = torch.randn(BATCH_SIZE, N_HEADS_FIXED, SEQ_LENGTH_FIXED, d_head, device=device, dtype=dtype)
        v = torch.randn(BATCH_SIZE, N_HEADS_FIXED, SEQ_LENGTH_FIXED, d_head, device=device, dtype=dtype)
        
        for name, backend in BACKENDS.items():
            try:
                with sdpa_kernel(backend):
                    timing = run_sdpa_benchmark(q, k, v)
                    all_results.append({"param_type": "D_MODEL", "param_value": d_model, "backend": name, "time_ms": timing})
                    print(f"  - {name}: {timing:.4f} мс")
            except RuntimeError:
                print(f"  - {name}: Недоступен")
    except torch.cuda.OutOfMemoryError:
        print(f"  - Ошибка нехватки памяти. Пропускаем...")
        break

print("\n--- Тест 3: Зависимость от количества голов (N_HEADS) ---")
SEQ_LENGTH_FIXED = 1024
D_MODEL_FIXED = 512
for n_heads in N_HEADS_LIST:
    if D_MODEL_FIXED % n_heads != 0: continue
    d_head = D_MODEL_FIXED // n_heads
    print(f"\nТестируем N_HEADS = {n_heads} (D_HEAD = {d_head})...")
    try:
        q = torch.randn(BATCH_SIZE, n_heads, SEQ_LENGTH_FIXED, d_head, device=device, dtype=dtype)
        k = torch.randn(BATCH_SIZE, n_heads, SEQ_LENGTH_FIXED, d_head, device=device, dtype=dtype)
        v = torch.randn(BATCH_SIZE, n_heads, SEQ_LENGTH_FIXED, d_head, device=device, dtype=dtype)
        
        for name, backend in BACKENDS.items():
            try:
                with sdpa_kernel(backend):
                    timing = run_sdpa_benchmark(q, k, v)
                    all_results.append({"param_type": "N_HEADS", "param_value": n_heads, "backend": name, "time_ms": timing})
                    print(f"  - {name}: {timing:.4f} мс")
            except RuntimeError:
                print(f"  - {name}: Недоступен")
    except torch.cuda.OutOfMemoryError:
        print(f"  - Ошибка нехватки памяти. Пропускаем...")
        break

# --- 5. Обработка результатов и построение графиков ---
print("\n--- Построение графиков ---")
df = pd.DataFrame(all_results)

fig, axes = plt.subplots(1, 3, figsize=(24, 7), sharey=False)
fig.suptitle(f'Производительность Attention бэкендов на {gpu_name} (низкоуровневый тест)', fontsize=16)

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
    ax.set_yscale('log') # Используем логарифмическую шкалу для наглядности
    ax.grid(True, which="both", ls="--")
    ax.legend(title="Бэкенд")

plot_results(axes[0], "SEQ_LENGTH", "Зависимость времени от длины последовательности")
plot_results(axes[1], "D_MODEL", "Зависимость времени от размерности модели")
plot_results(axes[2], "N_HEADS", "Зависимость времени от количества голов")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("backends_test_low_level.py.png")
print("Графики сохранены в файл 'backends_test_low_level.py.png'")