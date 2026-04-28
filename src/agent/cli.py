import argparse
import os

from dotenv import load_dotenv

from agent.core import make_agent
from agent.memory import MemoryCoordinator, MemoryStore
from agent.memory_mode import MemoryMode, parse_memory_mode


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="CLI Chat Agent")
    parser.add_argument(
        "--model",
        default="anthropic:claude-haiku-4-5-20251001",
        help="Model string, e.g. openai:gpt-4o, anthropic:claude-haiku-4-5-20251001, google_genai:gemini-2.5-flash",
    )
    parser.add_argument(
        "--system",
        default=None,
        help="Custom system prompt",
    )
    parser.add_argument(
        "--memory-mode",
        choices=[mode.value for mode in MemoryMode],
        default=MemoryMode.NONE.value,
        help="Cross-conversation memory mode: none, raw, summary",
    )
    parser.add_argument(
        "--conversation-id",
        default="default",
        help="Conversation identifier used for cross-session memory retrieval",
    )
    args = parser.parse_args()

    memory_mode = parse_memory_mode(args.memory_mode)
    agent = make_agent(args.model, args.system, memory_mode=memory_mode)

    memory_store = MemoryStore(os.getenv("MEMORY_DB_PATH", "memory.db"))
    memory = MemoryCoordinator(
        mode=memory_mode,
        store=memory_store,
        summary_model_str=os.getenv("SUMMARY_MODEL", args.model),
    )

    conversation_id = args.conversation_id
    messages = []

    print(
        f"Chat started (memory mode: {memory_mode.value}, conversation_id: {conversation_id}). "
        "Type 'quit' to exit.\n"
    )

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            break

        user_message = {"role": "user", "content": user_input}

        if memory_mode == MemoryMode.NONE:
            messages.append(user_message)
            request_messages = messages
        else:
            request_messages = memory.build_prompt_messages(conversation_id, [user_message])

        result = agent.invoke({"messages": request_messages})
        ai_msg = result["messages"][-1]
        print(f"\nAssistant: {ai_msg.content}\n")

        assistant_message = {"role": "assistant", "content": ai_msg.content}
        if memory_mode == MemoryMode.NONE:
            messages = result["messages"]
        else:
            memory.persist_exchange(conversation_id, user_message, assistant_message)


if __name__ == "__main__":
    main()
