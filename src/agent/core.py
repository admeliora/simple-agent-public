from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent

from agent.memory_mode import MemoryMode


def make_agent(
    model_str: str = "anthropic:claude-haiku-4-5-20251001",
    system_prompt: str | None = None,
    memory_mode: MemoryMode = MemoryMode.NONE,
):
    """Create a deep agent with the specified model provider.

    Args:
        model_str: Provider and model in "provider:model" format.
                   Examples: "openai:gpt-4o", "anthropic:claude-haiku-4-5-20251001",
                   "google_genai:gemini-2.5-flash"
        system_prompt: Optional system prompt override.
        memory_mode: Cross-conversation memory mode toggle. Currently a config
            switch used by the CLI/server and harness scaffolding.

    Returns:
        A compiled LangGraph agent supporting .invoke(), .stream(), .astream().
    """
    model = init_chat_model(model_str)
    kwargs = {}
    if system_prompt:
        kwargs["system_prompt"] = system_prompt

    _ = memory_mode  # Reserved for upcoming memory implementation wiring.
    return create_deep_agent(model=model, **kwargs)
