# semaforos-ia

Herramienta IA para semáforos.

## Stack

- **Backend:** FastAPI (Python 3.11)
- **Base de datos:** PostgreSQL
- **ORM:** SQLAlchemy (async)
- **Infraestructura:** Docker + docker-compose

## Requisitos

- Python 3.11+
- Docker y docker-compose (opcional pero recomendado)

## Setup local

```bash
# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Instalar dependencias
pip install -r requirements-dev.txt

# Copiar variables de entorno
cp .env.example .env
# Editar .env con tus valores

# Levantar la base de datos
docker-compose up db -d

# Iniciar el servidor
uvicorn src.main:app --reload
```

## Con Docker

```bash
docker-compose up --build
```

La API estará disponible en: http://localhost:8000

Documentación interactiva: http://localhost:8000/docs

## Scripts útiles

```bash
# Linting
ruff check .

# Formateo
black .

# Tests
pytest
```
