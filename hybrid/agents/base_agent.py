"""
Abstract Base Class for All Specialist Agents

Every specialist agent in the system (OrderAgent, AccountAgent, BillingAgent)
inherits from this class.  BaseAgent encapsulates the reasoning loop that is
common to all LLM-based agents: build a prompt from the current workflow
context, call the LLM, handle any tool calls the LLM requests, feed the tool
results back into the conversation, and iterate until the LLM produces a plain
text response (i.e. it has finished its reasoning).

This design follows the ReAct (Reasoning + Acting) pattern,
in which the LLM alternates between producing reasoning text and issuing
structured tool calls.  The loop is capped at five iterations per agent
invocation; in practice it rarely exceeds two (one tool call + one summary
response).

The invoke_rule_based method defined in each subclass overrides this LLM
loop with deterministic Python logic when the system runs in hybrid mode.
This eliminates one LLM call per task, reducing latency and API cost while
keeping the same tool-based interface.

"""

import sys
import time
from pathlib import Path
from typing import Any
from langchain_core.messages import SystemMessage, HumanMessage

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import build_llm


class BaseAgent:
    """Reusable LLM reasoning loop shared by all specialist agents.

    Subclasses customise behaviour by overriding three attributes in their
    __init__  method:

    Attributes
    ----------
    name : str
        Identifier string used in log records (e.g. "order_agent").
    system_prompt : str
        Domain-specific instructions prepended to every LLM call.  These
        define the agent's role, scope, and any business rules it must follow.
    tools : list[Tool]
        LangChain @tool decorated functions this agent is permitted to call.
        The LLM is only told about these tools, enforcing domain separation.
    """

    def __init__(self):
        self.name = "base_agent"
        self.system_prompt = ""
        self.tools = []
        self._llm = None  # Lazily initialised on first invocation to avoid upfront API calls

    def _get_llm(self):
        """Return the LLM instance, initialising it on first call (lazy init).

        The LLM is bound to this agent's specific tool set so the model knows
        which functions it may call and what parameters they expect.
        """
        if self._llm is None:
            self._llm = build_llm(bind_tools=self.tools if self.tools else None)
        return self._llm

    def invoke(self, state: dict) -> dict:
        """Execute the ReAct reasoning loop for the current task.

        The loop works as follows:
        1. Build the initial prompt from the current workflow context.
        2. Call the LLM.
        3. If the LLM response contains tool calls, execute each one and
           append the results to the conversation as ToolMessage objects.
        4. Call the LLM again with the updated conversation.
        5. Repeat until the LLM produces a plain-text response (no tool calls)
           or the iteration cap (5) is reached.

        The loop is intentionally simple: it does not implement backtracking,
        retries on tool errors, or multi-agent delegation.  Those concerns are
        handled at the orchestration level by the LangGraph graph in
        orchestration/graph.py.

        Parameters
        ----------
        state : dict
            The current WorkflowState dictionary, which includes the customer
            request, extracted entities, the task description for this agent,
            previously completed steps, and any errors recorded so far.

        Returns
        -------
        dict
            A result dictionary with keys:
            - agent: name of this agent
            - tool_calls: list of {tool, args, result} records
            - llm_call_log: list of {agent, latency_ms, …} records for every LLM call made
            - response: final text produced by the LLM
            - success: True if no tool returned an error status
        """
        from context.context_manager import ContextManager

        llm = self._get_llm()

        # Build the context string injected into the prompt.
        # When context_disabled is True (H2 control condition), the agent
        # receives only the bare customer request with no knowledge of what
        # previous agents have done.  When context is enabled, it receives the
        # full accumulated summary including prior results and remaining tasks.
        if state.get("context_disabled"):
            ctx = "Customer request: " + state.get("request", "")
        else:
            ctx = ContextManager.get_context_summary(state)

        # Compose the initial two-message conversation: system role description
        # and the human-turn prompt that includes context + task + available tools.
        msgs = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=(
                f"Context:\n{ctx}\n\n"
                f"Task: {state.get('current_task', 'Handle the customer request.')}\n\n"
                f"Available tools: {[t.name for t in self.tools]}"
            ))
        ]

        llm_call_log = []   # Records timing and content of every LLM API call
        tool_call_log = []  # Records every tool invocation and its result

        # ReAct loop: at most 5 iterations to prevent excessive API usage
        for _ in range(5):
            # Capture the full conversation text for the log (useful for debugging)
            request_text = "\n".join(m.content if hasattr(m, "content") else str(m) for m in msgs)

            # LLM call with timing
            t0 = time.perf_counter()
            resp = llm.invoke(msgs)
            t1 = time.perf_counter()
            msgs.append(resp)  # Add the LLM response to the running conversation

            # Record metadata about this API call for later analysis
            llm_call_log.append({
                "agent": self.name,
                "latency_ms": round((t1 - t0) * 1000, 2),
                "request_text": request_text,
                "response_text": resp.content if hasattr(resp, "content") else str(resp),
                "has_tool_calls": bool(resp.tool_calls),
                "tools_called": [tc["name"] for tc in resp.tool_calls] if resp.tool_calls else [],
            })

            # If the LLM produced no tool calls, it has finished its reasoning
            # and the last message is the final natural-language response.
            if not resp.tool_calls:
                break

            # Execute each tool call requested by the LLM and feed results back
            for tc in resp.tool_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]

                result = self._execute_tool(tool_name, tool_args)
                tool_call_log.append({"tool": tool_name, "args": tool_args, "result": result})

                # ToolMessage must carry the same tool_call_id that the LLM produced
                # so that the API can associate the result with the correct request.
                from langchain_core.messages import ToolMessage
                msgs.append(ToolMessage(
                    content=str(result),
                    tool_call_id=tc["id"]
                ))

        # The last message in the conversation is the agent's final answer
        final = msgs[-1].content if msgs and hasattr(msgs[-1], "content") else ""

        return {
            "agent": self.name,
            "tool_calls": tool_call_log,
            "llm_call_log": llm_call_log,
            "response": final,
            # The task is considered successful if none of the tool calls
            # returned a dict with {"status": "error"}
            "success": not any(
                e.get("result", {}).get("status") == "error"
                for e in tool_call_log
            ),
        }

    def _execute_tool(self, tool_name: str, tool_args: dict) -> Any:
        """Look up and invoke a tool by name.

        Parameters
        ----------
        tool_name : str
            The name of the tool as returned by the LLM's tool_call object.
        tool_args : dict
            The argument dictionary provided by the LLM.

        Returns
        -------
        Any
            The tool's return value, or an error dict if the tool is not found
            or raises an exception during execution.
        """
        for t in self.tools:
            if t.name == tool_name:
                try:
                    return t.invoke(tool_args)
                except Exception as e:
                    # Surface exceptions as structured error dicts so the LLM
                    # can include the failure in its reasoning and response.
                    return {"status": "error", "message": str(e)}
        return {"status": "error", "message": f"Tool '{tool_name}' not found."}