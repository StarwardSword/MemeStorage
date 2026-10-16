# MemeStorage

Хранилище мемов (картинки, видео, аудио) с поиском по текстовому запросу. Эмбеддинги считает `google/embeddinggemma-2`, векторы лежат в ChromaDB, файлы — локально. Доступ через gRPC и CLI-клиент.

## Требования

- Python 3.13+
- FFmpeg shared-сборка в `PATH` — нужна `torchcodec` для видео:

```powershell
winget install Gyan.FFmpeg.Shared
```

## Установка

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### PyTorch

Ставится отдельно и **до** `requirements.txt`, иначе pip подтянет torch с PyPI. Выберите один вариант.

CUDA 13.2 (NVIDIA GPU):

```powershell
pip install torch==2.14.1 torchvision==0.29.1 torchcodec==0.17.0 --index-url https://download.pytorch.org/whl/cu132
```

CPU:

```powershell
pip install torch==2.14.1 torchvision==0.29.1 torchcodec==0.17.0 --index-url https://download.pytorch.org/whl/cpu
```

Проверка — должно вывести `True` для CUDA-сборки:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

### Остальные зависимости

```powershell
pip install -r requirements.txt
```

Если модель на Hugging Face закрыта лицензией — примите её на странице модели и залогиньтесь:

```powershell
hf auth login
```

## Генерация gRPC-кода

Файлы `*_pb2*` не хранятся в репозитории, их нужно сгенерировать:

```powershell
make proto
```

Без `make`:

```powershell
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. --pyi_out=. ./api/grpc/proto/service.proto
```

## Конфигурация

Сервер читает `config.yaml` из текущей директории:

```yaml
gemma:
  device: "cuda"
  dtype: "bfloat16"

chroma:
  path: "./.runtime/storage.chromadb"

files:
  root: "./.runtime/files/"
  sqlite_connection_string: "./.runtime/sqlite.db"
  chunk_size: 65536

grpc:
  port: "[::]:50051"
  max_workers: 10
```

| Ключ | Значения | По умолчанию |
|---|---|---|
| `gemma.device` | `cpu`, `cuda` (нужна CUDA-сборка PyTorch) | `cpu` |
| `gemma.dtype` | `float32`, `float16`, `bfloat16` | `float32` |
| `chroma.path` | путь к базе ChromaDB | обязательный |
| `files.root` | директория для файлов | обязательный |
| `files.sqlite_connection_string` | путь к файлу SQLite | обязательный |
| `files.chunk_size` | размер чанка в байтах при чтении файлов | `65536` |
| `grpc` | необязательная секция; без неё API не запускается | — |
| `grpc.port` | адрес `host:port` | обязательный |
| `grpc.max_workers` | число потоков gRPC-сервера | `10` |

## Запуск сервера

```powershell
python main.py
```

При первом запуске скачивается модель.

## CLI-клиент

```powershell
# Загрузить файл или папку (-r — рекурсивно, -d — описание, -j — число параллельных загрузок)
python cli.py upload ./memes -r -d "кот"

# Найти по запросу и скачать 5 лучших
python cli.py get "грустный кот" -o ./out -n 5

# Скачать по id
python cli.py download <uuid> <uuid> -o ./out
```

Другой адрес сервера: `python cli.py --server host:50051 ...`
