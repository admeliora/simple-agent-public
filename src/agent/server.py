import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent.core import make_agent
from agent.memory import MemoryCoordinator, MemoryStore, memory_scope_id
from agent.memory_mode import parse_memory_mode

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEFAULT_MODEL = "anthropic:claude-haiku-4-5-20251001"
DEFAULT_CONVERSATION_ID = "default"

memory_mode = parse_memory_mode(os.getenv("MEMORY_MODE"))
model_str = os.getenv("MODEL", DEFAULT_MODEL)
memory_store = MemoryStore(os.getenv("MEMORY_DB_PATH", "memory.db"))
memory = MemoryCoordinator(
    mode=memory_mode,
    store=memory_store,
    summary_model_str=os.getenv("SUMMARY_MODEL", model_str),
)

agent = make_agent(model_str=model_str, memory_mode=memory_mode)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    conversation_id: str | None = None
    user_id: str | None = None


@app.post("/chat")
def chat(req: ChatRequest):
    conversation_id = req.conversation_id or DEFAULT_CONVERSATION_ID
    scope_id = memory_scope_id(req.user_id, conversation_id)
    incoming_messages = [{"role": m.role, "content": m.content} for m in req.messages]
    request_messages = memory.build_prompt_messages(scope_id, incoming_messages)

    result = agent.invoke({"messages": request_messages})
    ai_msg = result["messages"][-1]

    latest_user = next(
        (
            {"role": m["role"], "content": m["content"]}
            for m in reversed(incoming_messages)
            if m["role"] == "user"
        ),
        None,
    )
    if latest_user is not None:
        memory.persist_exchange(
            scope_id,
            latest_user,
            {"role": "assistant", "content": ai_msg.content},
        )

    return {"reply": ai_msg.content}


def main():
    import uvicorn

    uvicorn.run("agent.server:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
