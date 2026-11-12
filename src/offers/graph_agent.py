import asyncio
from typing import Annotated, Any, Literal, TypedDict

from google.genai import types
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool, StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from mcp import Client
from mcp.types import TextContent, Tool

from offers.llm import build_http_options
from offers.mcp_server import create_server, provide_deps
from offers.observability import record_step, record_tokens, timed
from offers.pyai_agent import ANSWER_INSTRUCTION, ANSWER_TOOL, MAX_MODEL_REQUESTS, AgentAnswer
from offers.tools import AgentDeps

type ChatRunnable = Runnable[LanguageModelInput, AIMessage]

ANSWER_TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": ANSWER_TOOL,
        "description": "Gives the final answer to the user. Call it once you have the data.",
        "parameters": AgentAnswer.model_json_schema(),
    },
}


class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    answer: AgentAnswer | None


def as_langchain_tool(client: Client, tool: Tool) -> StructuredTool:
    async def call(**args: Any) -> Any:
        result = await client.call_tool(tool.name, args)
        if result.is_error:
            return " ".join(part.text for part in result.content if isinstance(part, TextContent))
        return result.structured_content

    return StructuredTool.from_function(
        coroutine=call,
        name=tool.name,
        description=tool.description or "",
        args_schema=tool.input_schema,
    )


def record_model_call(message: AIMessage) -> None:
    calls = [
        (call["name"], call["args"]) for call in message.tool_calls if call["name"] != ANSWER_TOOL
    ]
    record_step("graph", calls)
    usage = message.usage_metadata
    if usage:
        thinking = usage.get("output_token_details", {}).get("reasoning", 0)
        record_tokens(
            "graph.step", usage["input_tokens"], usage["output_tokens"] - thinking, thinking
        )


def build_graph(llm: ChatRunnable, tools: list[BaseTool]) -> CompiledStateGraph[Any]:
    async def call_model(state: GraphState) -> dict[str, Any]:
        message = await llm.ainvoke([SystemMessage(ANSWER_INSTRUCTION), *state["messages"]])
        record_model_call(message)
        return {"messages": [message]}

    def next_node(state: GraphState) -> Literal["tools", "answer"]:
        message = state["messages"][-1]
        assert isinstance(message, AIMessage)
        names = [call["name"] for call in message.tool_calls]
        return "tools" if names and ANSWER_TOOL not in names else "answer"

    def read_answer(state: GraphState) -> dict[str, Any]:
        message = state["messages"][-1]
        assert isinstance(message, AIMessage)
        for call in message.tool_calls:
            if call["name"] == ANSWER_TOOL:
                return {"answer": AgentAnswer.model_validate(call["args"])}
        return {"answer": AgentAnswer(answer=message.text, offers=[])}

    graph = StateGraph(GraphState)
    graph.add_node("model", call_model)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("answer", read_answer)
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", next_node)
    graph.add_edge("tools", "model")
    graph.add_edge("answer", END)
    return graph.compile()


def build_model(deps: AgentDeps) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=deps.settings.gemini_model,
        google_api_key=deps.settings.gemini_api_key,
        thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
    )


async def run_graph(deps: AgentDeps, question: str, model: BaseChatModel) -> AgentAnswer:
    async with Client(create_server(provide_deps(deps))) as client:
        listed = await client.list_tools()
        tools: list[BaseTool] = [as_langchain_tool(client, tool) for tool in listed.tools]
        llm = model.bind_tools([*tools, ANSWER_TOOL_SPEC], tool_choice="any").bind(
            http_options=build_http_options(deps.settings)
        )
        graph = build_graph(llm, tools)
        state = await graph.ainvoke(
            {"messages": [HumanMessage(question)], "answer": None},
            {"recursion_limit": 2 * MAX_MODEL_REQUESTS},
        )
    answer: AgentAnswer = state["answer"]
    return answer


def innermost(error: BaseException) -> BaseException:
    while isinstance(error, BaseExceptionGroup) and len(error.exceptions) == 1:
        error = error.exceptions[0]
    return error


def ask_graph(deps: AgentDeps, question: str, model: BaseChatModel | None = None) -> AgentAnswer:
    with timed("graph.run"):
        try:
            return asyncio.run(run_graph(deps, question, model or build_model(deps)))
        except ExceptionGroup as group:
            raise innermost(group) from None
