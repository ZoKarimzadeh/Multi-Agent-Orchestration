"""
Shared Workflow State Management

The ContextManager is the mechanism by which the multi-agent system implements
shared context, the core feature examined by Hypothesis H2 of the thesis
("Shared workflow context reduces error rates in multi-step customer-service
tasks compared to context-disabled execution").

In a conventional request-response system, each component receives only its
immediate inputs and is unaware of what other components have already done.
The ContextManager breaks this isolation: it maintains a running summary of
the entire workflow — what was asked, which agents have acted, what they found,
what remains to be done, and any errors encountered — and injects that summary
into every subsequent agent's prompt.  This gives each specialist agent full
situational awareness without requiring direct agent-to-agent communication.

The class is stateless (all methods are static) because the state itself lives
in the LangGraph WorkflowState dictionary, which is passed around by the graph
engine.  The ContextManager simply reads from and writes to that dictionary.

"""


class ContextManager:
    """Static utility class for creating and summarising multi-agent workflow state.

    All methods are class-level static methods because there is no instance
    data to maintain; the state is owned by the LangGraph graph, not by this class.
    """

    @staticmethod
    def create_initial_state(customer_message: str, context_disabled: bool = False, hybrid: bool = False) -> dict:
        """Construct the initial WorkflowState dictionary for a new customer request.

        This dictionary is the single source of truth for the entire workflow.
        LangGraph passes it from node to node, and each node returns a partial
        update dictionary; LangGraph merges the update into the existing state
        before calling the next node.

        Parameters
        ----------
        customer_message : str
            The raw natural-language request received from the customer.
        context_disabled : bool
            When True, each agent receives only the bare customer request rather
            than the accumulated context summary.  This models the no-context
            experimental condition (H2 control group).
        hybrid : bool
            When True, specialist agents use deterministic rule-based tool dispatch
            instead of a second LLM call, reducing latency and API costs.

        Returns
        -------
        dict
            A fully initialised WorkflowState-compatible dictionary.
        """
        return {
            "customer_id": None,          # Populated by AgentManager once extracted from message
            "request": customer_message,  # Original, unmodified customer message
            "intent": None,               # Intent label assigned by AgentManager (e.g. "refund_request")
            "entities": {},               # Structured data extracted from the message (order IDs, emails, …)
            "task_queue": [],             # Ordered list of {agent, action, params} dicts to execute
            "completed_tasks": [],        # Tasks that have already been executed (success or failure)
            "errors": [],                 # Structured error records from failed tool calls
            "escalated": False,           # True if the workflow was handed off to a human agent
            "step_count": 0,              # Number of graph node executions so far (for MAX_REASONING_STEPS guard)
            "messages": [],               # LangChain message objects (used by the LangGraph message reducer)
            "final_response": None,       # The customer-facing response string produced at the end
            "context_disabled": context_disabled,
            "hybrid": hybrid,
            "llm_call_log": [],           # Detailed log of every LLM API call (agent, latency, tokens, …)
        }

    @staticmethod
    def get_context_summary(state: dict) -> str:
        """Produce a human-readable summary of the current workflow state.

        This summary is prepended to every specialist agent's prompt when
        context_disabled is False.  It replaces what would otherwise be
        agent-to-agent message passing: instead of agents sending messages to
        each other, they all read the same growing summary and can therefore
        avoid redundant tool calls and make informed decisions.

        For example, if OrderAgent has already confirmed that an order exists
        and is in "return_initiated" status, BillingAgent can read that fact
        from the context summary and skip its own redundant order lookup before
        processing the refund.

        Parameters
        ----------
        state : dict
            The current WorkflowState dictionary.

        Returns
        -------
        str
            A multi-line plain-text summary suitable for inclusion in an LLM prompt.
        """
        lines = [f"Customer request: {state.get('request', 'N/A')}"]

        # Include identifying information if already extracted by AgentManager
        if state.get("customer_id"):
            lines.append(f"Customer ID: {state['customer_id']}")
        if state.get("intent"):
            lines.append(f"Intent: {state['intent']}")
        if state.get("entities"):
            lines.append(f"Entities: {state['entities']}")

        # Summarise tasks that have already been completed in this workflow run.
        # Each entry shows the agent responsible, the task description, and a
        # truncated version of the tool result, enough for a downstream agent
        # to make a decision without being overwhelmed by raw data.
        done = state.get("completed_tasks", [])
        if done:
            lines.append(f"\nCompleted steps ({len(done)}):")
            for t in done:
                status = "OK" if t.get("success") else "FAILED"
                lines.append(f"  [{status}] {t['agent']}: {t['task_type']}")
                if t.get("result"):
                    r = str(t["result"])
                    # Truncate long results to keep the context summary token-efficient
                    lines.append(f"    result: {r[:200]}{'...' if len(r) > 200 else ''}")

        # Show what still needs to be done so agents understand the broader plan
        remaining = state.get("task_queue", [])
        if remaining:
            lines.append(f"\nStill to do: {remaining}")

        # Surface any errors so agents can adapt their behaviour accordingly
        errs = state.get("errors", [])
        if errs:
            lines.append("\nErrors so far:")
            for e in errs:
                lines.append(f"  {e['agent']}: {e['message']}")

        return "\n".join(lines)