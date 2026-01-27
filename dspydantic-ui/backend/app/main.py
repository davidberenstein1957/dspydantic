from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database import init_db
from app.routers import examples, optimization, prompts, tasks, evaluation

app = FastAPI(title="DSPydantic Labeling UI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(tasks.router)
app.include_router(examples.router)
app.include_router(optimization.router)
app.include_router(evaluation.router)
app.include_router(prompts.router)


@app.on_event("startup")
async def startup_event():
    init_db()


@app.get("/")
async def root():
    return {"message": "DSPydantic Labeling UI API"}


@app.get("/health")
async def health():
    return {"status": "healthy"}
