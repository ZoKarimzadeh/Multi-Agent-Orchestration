#!/usr/bin/env python3
"""List runs where a tool returned the exact 'Status must be one of' error.

Writes `scripts/failed_status_tool_errors.json` with one entry per run.
"""
import json
from pathlib import Path

LLM_BASE_FILES = [
    "code/llm-base/evaluation/results/gemma4-31b/evaluation_results.json",
    "code/llm-base/evaluation/results/gpt-4o-mini/evaluation_results.json",
    "code/llm-base/evaluation/results/gpt-5.4-mini/evaluation_results.json",
    "code/llm-base/evaluation/results/qwen2.5-7b_local/evaluation_results.json",
]

TARGET = "Status must be one of"


def scan_file(path):
    out = []
    data = json.load(open(path))
    runs = data if isinstance(data, list) else data.get("runs", [data])
    for r in runs:
        found = False
        details = []
        for task in r.get("completed_tasks", []):
            for call in task.get("tool_calls", []) or []:
                if isinstance(call, dict):
                    res = call.get("result")
                    if isinstance(res, dict):
                        msg = res.get("message") or ""
                        if isinstance(msg, str) and TARGET in msg:
                            found = True
                            details.append({
                                "tool": call.get("tool"),
                                "args": call.get("args"),
                                "message": msg,
                            })
        if found:
            out.append({
                "file": str(path),
                "run_number": r.get("run_number") or r.get("id"),
                "scenario_id": r.get("scenario_id"),
                "condition": r.get("condition"),
                "error_count": int(r.get("error_count", 0) or 0),
                "details": details,
            })
    return out


def main():
    all_found = []
    for f in LLM_BASE_FILES:
        p = Path(f)
        if not p.exists():
            continue
        all_found.extend(scan_file(p))

    outp = Path("scripts/failed_status_tool_errors.json")
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(all_found, indent=2))

    print(f"Found {len(all_found)} runs with '{TARGET}' across llm-base files.")
    for e in all_found[:20]:
        print(f"file={Path(e['file']).name} run={e['run_number']} scenario={e.get('scenario_id')} err={e['error_count']} hits={len(e['details'])}")


if __name__ == '__main__':
    main()
