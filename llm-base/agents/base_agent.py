import sys
import time
from pathlib import Path
from typing import Any
from langchain_core.messages import SystemMessage, HumanMessage

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import build_llm


class BaseAgent:

    def __init__(self):
        self.name = "base_agent"
        self.system_prompt = ""
        self.tools = []
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            self._llm = build_llm(bind_tools=self.tools if self.tools else None)
        return self._llm

    def invoke(self, state: dict) -> dict:
        from context.context_manager import ContextManager

        llm = self._get_llm()

        if state.get("context_disabled"):
            ctx = "Customer request: " + state.get("request", "")
        else:
            ctx = ContextManager.get_context_summary(state)

        msgs = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=(
                f"Context:\n{ctx}\n\n"
                f"Task: {state.get('current_task', 'Handle the customer request.')}\n\n"
                f"Available tools: {[t.name for t in self.tools]}"
            ))
        ]

        llm_call_log = []
        tool_call_log = []
        for _ in range(5):
            request_text = "\n".join(m.content if hasattr(m, "content") else str(m) for m in msgs)

            t0 = time.perf_counter()
            resp = llm.invoke(msgs)
            t1 = time.perf_counter()
            msgs.append(resp)

            llm_call_log.append({
                "agent": self.name,
                "latency_ms": round((t1 - t0) * 1000, 2),
                "request_text": request_text,
                "response_text": resp.content if hasattr(resp, "content") else str(resp),
                "has_tool_calls": bool(resp.tool_calls),
                "tools_called": [tc["name"] for tc in resp.tool_calls] if resp.tool_calls else [],
            })

            if not resp.tool_calls:
                break

            for tc in resp.tool_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]

                result = self._execute_tool(tool_name, tool_args)
                tool_call_log.append({"tool": tool_name, "args": tool_args, "result": result})

                from langchain_core.messages import ToolMessage
                msgs.append(ToolMessage(
                    content=str(result),
                    tool_call_id=tc["id"]
                ))

        final = msgs[-1].content if msgs and hasattr(msgs[-1], "content") else ""

        return {
            "agent": self.name,
            "tool_calls": tool_call_log,
            "llm_call_log": llm_call_log,
            "response": final,
            "success": not any(
                e.get("result", {}).get("status") == "error"
                for e in tool_call_log
            ),
        }

    def _execute_tool(self, tool_name: str, tool_args: dict) -> Any:
        for t in self.tools:
            if t.name == tool_name:
                try:
                    return t.invoke(tool_args)
                except Exception as e:
                    return {"status": "error", "message": str(e)}
        return {"status": "error", "message": f"Tool '{tool_name}' not found."}