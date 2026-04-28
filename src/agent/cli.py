import argparse

from dotenv import load_dotenv

from agent.core import make_agent
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
        help="Cross-conversation memory mode scaffold: none, raw, summary",
    )
    args = parser.parse_args()

    memory_mode = parse_memory_mode(args.memory_mode)
    agent = make_agent(args.model, args.system, memory_mode=memory_mode)
    messages = []

    print(f"Chat started (memory mode: {memory_mode.value}). Type 'quit' to exit.\n")

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

        messages.append({"role": "user", "content": user_input})
        result = agent.invoke({"messages": messages})
        ai_msg = result["messages"][-1]
        print(f"\nAssistant: {ai_msg.content}\n")
        messages = result["messages"]


if __name__ == "__main__":
    main()
