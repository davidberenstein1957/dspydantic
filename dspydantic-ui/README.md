# DSPydantic Labeling UI

A web-based UI for managing DSPy optimization tasks with Pydantic schemas.

## Quick Start

1. Copy `.env.example` to `.env` and configure:
```bash
cp .env.example .env
```

2. If you have an existing database volume with SSL issues, delete it first:
```bash
docker-compose down -v
```

3. Start services with Docker Compose:
```bash
docker-compose up
```

4. Access the application:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

**Note:** If you encounter PostgreSQL connection errors about SSL/pg_hba.conf, delete the database volume with `docker-compose down -v` and restart.

## Development

### Backend

```bash
cd backend
pip install -r requirements.txt
# Install dspydantic from parent directory
pip install -e ../..
uvicorn app.main:app --reload
```

Note: The backend requires `dspydantic` to be installed. If running locally, install it from the parent directory with `pip install -e ../..` or install it from PyPI if available.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

## Project Structure

- `backend/` - FastAPI backend application
- `frontend/` - React frontend application
- `docker-compose.yml` - Docker orchestration
