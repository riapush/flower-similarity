## Инструкция по запуску
1. Клонируйте репозиторий.
2. Перейдите в директорию:
`cd flower-search`
3. Соберите Docker-образ:
`docker build -t flower-search .`
4. Запустите контейнер:
`docker run -p 8000:8000 flower-search`

## Пример использования
```bash
curl -X POST -F "file=@test_flower.jpg" http://localhost:8000/search