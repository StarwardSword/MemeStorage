# MemeStorage

Хранилище мемов (картинки, видео, аудио) с поиском по текстовому запросу. Эмбеддинги считает `google/embeddinggemma-2`, векторы лежат в ChromaDB, файлы — локально. Доступ через gRPC, HTTP (FastAPI) и CLI-клиент.

## Требования

- Python 3.14+
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
  semaphore_count: 4

chroma:
  path: "./.runtime/storage.chromadb"

files:
  root: "./.runtime/files/"
  sqlite_connection_string: "./.runtime/sqlite.db"
  chunk_size: 65536

grpc:
  port: "[::]:50051"
  max_workers: 10
  max_records: 32

http:
  host: "0.0.0.0"
  port: 8000
  max_workers: 10
```

| Ключ | Значения | По умолчанию |
|---|---|---|
| `gemma.device` | `cpu`, `cuda` (нужна CUDA-сборка PyTorch) | `cpu` |
| `gemma.dtype` | `float32`, `float16`, `bfloat16` | `float32` |
| `gemma.semaphore_count` | сколько вызовов модели выполняется одновременно, общий лимит для gRPC и HTTP | `4` |
| `chroma.path` | путь к базе ChromaDB | обязательный |
| `files.root` | директория для файлов | обязательный |
| `files.sqlite_connection_string` | путь к файлу SQLite | обязательный |
| `files.chunk_size` | размер чанка в байтах при чтении файлов | `65536` |
| `grpc` | необязательная секция; без неё gRPC не запускается | — |
| `grpc.port` | адрес `host:port` | обязательный |
| `grpc.max_workers` | число потоков gRPC-сервера | `10` |
| `grpc.max_records` | наибольший `record_count` в поисковых запросах gRPC | `32` |
| `http` | необязательная секция; без неё HTTP не запускается | — |
| `http.host` | адрес, на котором слушает HTTP-сервер | `127.0.0.1` |
| `http.port` | порт HTTP-сервера | `8000` |
| `http.max_workers` | число потоков для обработчиков HTTP-запросов | `10` |

Нужна хотя бы одна из секций `grpc` и `http`.

## Запуск сервера

```powershell
python main.py
```

При первом запуске скачивается модель. Если в конфиге есть обе секции, gRPC и HTTP работают в одном процессе с общей моделью. Ctrl+C останавливает оба сервера.

## HTTP API

Спецификация лежит в [`api/http/openapi.yaml`](api/http/openapi.yaml). Запущенный сервер отдаёт её же по адресу `/openapi.json`, а Swagger UI — по адресу `/docs`.

| Метод | Путь | Что делает |
|---|---|---|
| `POST` | `/files` | загрузить файл (`multipart/form-data`) → `201 {"id": ...}` |
| `POST` | `/files/batch` | загрузить несколько файлов (`multipart/form-data`) → `200 {"failed": [...]}` |
| `GET` | `/files/{id}` | скачать файл; `?download=true` — отдать как вложение |
| `GET` | `/files/{id}/info` | имя и тип файла |
| `GET` | `/search?q=...&count=10` | поиск по одному запросу |
| `POST` | `/search/query` | поиск по нескольким запросам: `{"queries": [...], "count": 10}` |
| `POST` | `/search/embeddings` | поиск по готовым эмбеддингам: `{"embeddings": [[...]], "count": 10}` |
| `GET` | `/health` | проверка, что сервер жив |

Поля формы `POST /files`:

- `file` — обязательное, картинка, видео или аудио. Если тип не передан или равен `application/octet-stream`, сервер определяет его по имени файла. На другие типы сервер отвечает `415`.
- `description` — текстовое описание, файл находится и по нему.
- `embeddings` и `text_embeddings` — готовые эмбеддинги файла и описания в виде JSON-массива чисел. Размерность `embeddings` должна совпадать с моделью сервера.

`POST /files/batch` принимает файлы в повторяющемся поле `files` и считает эмбеддинги для всех файлов за один вызов модели. Тип каждого файла определяется так же, как в `POST /files`. Описания и готовые эмбеддинги здесь не передаются. Если файл не картинка, не видео и не аудио или его не удалось сохранить, запрос не падает: имя файла попадает в `failed`, остальные файлы сохраняются. id сохранённых файлов не возвращаются, файлы находятся поиском.

Ошибки возвращаются в виде `{"detail": ...}`:

| Код | Когда |
|---|---|
| `404` | файла с таким id нет |
| `415` | файл не картинка, не видео и не аудио (`POST /files`) |
| `422` | запрос не прошёл валидацию: битый uuid, `count` вне `1..100`, размерность эмбеддинга не совпадает с моделью и т. п. |

Для `422` поле `detail` — список ошибок по полям в формате FastAPI, для остальных кодов — строка.

```powershell
# Загрузить файл с описанием
curl.exe -F "file=@cat.png" -F "description=грустный кот" http://localhost:8000/files

# Загрузить несколько файлов одним запросом
curl.exe -F "files=@cat.png" -F "files=@dog.mp4" -F "files=@meow.mp3" http://localhost:8000/files/batch

# Найти 5 файлов по запросу
curl.exe -G http://localhost:8000/search --data-urlencode "q=грустный кот" -d count=5

# Скачать файл
curl.exe -o cat.png http://localhost:8000/files/<uuid>
```

После изменения HTTP API спецификацию нужно сгенерировать заново:

```powershell
make openapi
```

Без `make` (gRPC-код должен быть уже сгенерирован):

```powershell
python -m api.http.openapi
```

## gRPC API

Контракт лежит в [`api/grpc/proto/service.proto`](api/grpc/proto/service.proto).

| Метод | Что делает |
|---|---|
| `Save` | загрузить файл: поток из `info`, затем чанков `data` → `uuid` |
| `SaveMultiple` | загрузить несколько файлов одним потоком → `failed` |
| `GetFile` | скачать файл: поток из `info`, затем чанков `chunk` |
| `GetByQuery` | поиск по текстовым запросам |
| `GetByEmbeddings` | поиск по готовым эмбеддингам |

В `SaveMultiple` каждое сообщение `info` начинает новый файл, следующие за ним `data` — его содержимое. В `info` есть только имя файла и необязательный `content_type`: описания и готовые эмбеддинги здесь не передаются. Эмбеддинги всех файлов считаются за один вызов модели после того, как сохранён весь поток. Неподдерживаемый тип файла или ошибка сохранения не завершают вызов ошибкой: имя файла попадает в `failed`, остальные файлы сохраняются. uuid сохранённых файлов не возвращаются, файлы находятся поиском.

Ошибки приходят как статусы gRPC, текст ошибки — в `details`:

| Статус | Когда |
|---|---|
| `INVALID_ARGUMENT` | первое сообщение `Save` или `SaveMultiple` не `info`; в `Save` следующее сообщение не `data`; неподдерживаемый тип файла или тип не определяется по имени (`Save`); размерность эмбеддинга не совпадает с моделью; битый uuid; пустой список запросов или эмбеддингов; `record_count` меньше 1 или больше `grpc.max_records` |
| `NOT_FOUND` | файла с таким uuid нет (`GetFile`) |
| `INTERNAL` | поиск нашёл записи, файлов которых нет в хранилище (хранилища рассинхронизированы) |

## CLI-клиент

```powershell
# Загрузить файл или папку (-r — рекурсивно, -b — файлов в одном вызове, -j — число параллельных вызовов)
python cli.py upload ./memes -r -b 32 -j 4

# Загрузить с описанием (-d), файлы уходят по одному
python cli.py upload ./memes -r -d "кот"

# Найти по запросу и скачать 5 лучших
python cli.py get "грустный кот" -o ./out -n 5

# Скачать по id
python cli.py download <uuid> <uuid> -o ./out
```

`upload` отправляет файлы пачками через `SaveMultiple`, по умолчанию по 32 файла (`-b`). В одну пачку не попадают два файла с одинаковым именем: сервер сообщает о несохранённых файлах по имени. Сервер не возвращает uuid для пачки, поэтому клиент выводит только путь сохранённого файла. `SaveMultiple` не принимает описание, поэтому с `-d` файлы уходят по одному через `Save`, и клиент выводит uuid каждого файла. Файлы, тип которых не определяется как картинка, видео или аудио, клиент пропускает и не отправляет.

Если хотя бы один id в `download` не является uuid, клиент ничего не скачивает и выводит список неправильных id.

Другой адрес сервера: `python cli.py --server host:50051 ...`

## Известные проблемы

- **Ошибка записи одного файла роняет весь батч** (`SaveMultiple`, `POST /files/batch`, `cli.py upload` без `-d`). `LocalFileStorage.save` при ошибке уже удаляет файл, а повторный `file_storage.remove` в `Application.save_multiple` бросает `FileNotFoundError`. Вызов падает с `UNKNOWN` в gRPC и `500` в HTTP. Файлы батча, сохранённые до ошибки, остаются на диске без записей в векторной базе и поиском не находятся. До остальных файлов батча дело не доходит.
- **Нет отката при ошибке эмбеддингов или векторной базы в батче.** Если модель или ChromaDB падают на батче, все его файлы остаются на диске без записей в векторной базе.