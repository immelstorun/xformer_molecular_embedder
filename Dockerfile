# Используем базовый образ PyTorch
FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

# --- Аргументы сборки для UID/GID и имени пользователя ---
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USERNAME=devuser

# --- Установка системных зависимостей и создание пользователя (от root) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Базовые утилиты
    curl \
    tmux \
    sudo \
    # --- >>> RDKit Draw и общие графические зависимости <<< ---
    libxrender1 \
    libfreetype6 \
    libfontconfig1 \
    libxft2 \
    libxext6 \
    # --- >>> Конец зависимостей <<< ---
    # Очистка APT кэша для уменьшения размера образа
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    # Создаем группу с переданным GID
    && groupadd -g $GROUP_ID $USERNAME \
    # Создаем пользователя с переданным UID, GID, домашней директорией и оболочкой bash
    && useradd -u $USER_ID -g $GROUP_ID -m -s /bin/bash $USERNAME \
    # Добавляем пользователя в группу sudo для удобства (опционально)
    && usermod -aG sudo $USERNAME \
    # Разрешаем sudo без пароля для этого пользователя (опционально)
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# --- Переключаемся на созданного пользователя ---
USER $USERNAME
WORKDIR /home/$USERNAME

# --- Установка пользовательских инструментов и конфигураций ---
RUN mkdir -p /home/$USERNAME/.config
COPY --chown=$USER_ID:$GROUP_ID ./starship.toml /home/$USERNAME/.config/starship.toml

RUN curl -sSL https://install.python-poetry.org | python3 - && \
    echo '' >> /home/$USERNAME/.bashrc && \
    echo '# Add Poetry to PATH' >> /home/$USERNAME/.bashrc && \
    echo 'export PATH="/home/'$USERNAME'/.local/bin:$PATH"' >> /home/$USERNAME/.bashrc

ENV PATH="/home/${USERNAME}/.local/bin:${PATH}"

RUN curl -sS https://starship.rs/install.sh | sh -s -- -y -b "/home/$USERNAME/.local/bin" && \
    echo '' >> /home/$USERNAME/.bashrc && \
    echo '# Initialize Starship prompt' >> /home/$USERNAME/.bashrc && \
    echo 'eval "$(starship init bash)"' >> /home/$USERNAME/.bashrc

# --- Настройка приложения ---
WORKDIR /app

# Копируем файлы с описанием зависимостей
COPY --chown=$USER_ID:$GROUP_ID ./pyproject.toml ./poetry.lock* .

# --- Конфигурация Poetry ---
# Создавать виртуальное окружение (по умолчанию true, но явно лучше)
# РАЗРЕШИТЬ виртуальному окружению видеть пакеты из системного Python (из базового образа)
# Опционально: создавать .venv в папке проекта (удобно для Docker)
RUN poetry config virtualenvs.create true && \
    poetry config virtualenvs.options.system-site-packages true && \
    poetry config virtualenvs.in-project true  # Рекомендуется для Docker

RUN poetry install --no-interaction --no-ansi --no-root

# принудительно устанавливаем xformers с помощью pip в это же окружение.
RUN . .venv/bin/activate && \
    pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu118

# Копируем остальной код приложения
COPY --chown=$USER_ID:$GROUP_ID ./app .

# --- Команда по умолчанию ---
CMD ["/bin/bash", "--login"]