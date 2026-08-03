class ContextManager:

    @staticmethod
    def create_initial_state(customer_message: str, context_disabled: bool = False) -> dict:
        return {
            "customer_id": None,
            "request": customer_message,
            "intent": None,
            "entities": {},
            "task_queue": [],
            "completed_tasks": [],
            "errors": [],
            "escalated": False,
            "step_count": 0,
            "messages": [],
            "final_response": None,
            "context_disabled": context_disabled,
            "llm_call_log": [],
        }

    @staticmethod
    def get_context_summary(state: dict) -> str:
        lines = [f"Customer request: {state.get('request', 'N/A')}"]

        if state.get("customer_id"):
            lines.append(f"Customer ID: {state['customer_id']}")
        if state.get("intent"):
            lines.append(f"Intent: {state['intent']}")
        if state.get("entities"):
            lines.append(f"Entities: {state['entities']}")

        done = state.get("completed_tasks", [])
        if done:
            lines.append(f"\nCompleted steps ({len(done)}):")
            for t in done:
                status = "OK" if t.get("success") else "FAILED"
                lines.append(f"  [{status}] {t['agent']}: {t['task_type']}")
                if t.get("result"):
                    r = str(t["result"])
                    lines.append(f"    result: {r[:200]}{'...' if len(r) > 200 else ''}")

        remaining = state.get("task_queue", [])
        if remaining:
            lines.append(f"\nStill to do: {remaining}")

        errs = state.get("errors", [])
        if errs:
            lines.append("\nErrors so far:")
            for e in errs:
                lines.append(f"  {e['agent']}: {e['message']}")

        return "\n".join(lines)