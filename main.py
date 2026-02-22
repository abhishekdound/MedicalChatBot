from src import *
from fastapi import FastAPI , Request , Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

llm = LLM()

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})


@app.post('/get')
async def get(msg: str = Form(...)):
    print("User:", msg)

    ans = llm.rag_result(msg)

    print("Bot:", ans)
    return ans