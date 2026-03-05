from typing import Any, Dict, List, Set, Tuple


ALLOWED_ROLES: Set[str] = {"system", "user", "assistant", "tool"}


def validate_openai_tool_topology(sample: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate OpenAI-style tool-calling topology for a trajectory sample.

    Rules (minimal hard checks):
    1) Each message role must be one of {system,user,assistant,tool}.
    2) assistant message with tool_calls must be followed by matching tool messages.
    3) Every tool message must include tool_call_id.
    """
    errors: List[str] = []
    messages = sample.get("messages", [])

    if not isinstance(messages, list) or not messages:
        return False, ["messages must be a non-empty list"]

    idx = 0
    while idx < len(messages):
        msg = messages[idx]
        if not isinstance(msg, dict):
            errors.append(f"messages[{idx}] is not an object")
            idx += 1
            continue

        role = msg.get("role")
        if role not in ALLOWED_ROLES:
            errors.append(f"messages[{idx}] has invalid role: {role}")

        if role == "tool" and not msg.get("tool_call_id"):
            errors.append(f"messages[{idx}] tool message missing tool_call_id")

        if role == "assistant" and msg.get("tool_calls"):
            tool_calls = msg.get("tool_calls")
            if not isinstance(tool_calls, list) or not tool_calls:
                errors.append(f"messages[{idx}] assistant.tool_calls must be a non-empty list")
                idx += 1
                continue

            expected_ids = {
                c.get("id") for c in tool_calls if isinstance(c, dict) and c.get("id")
            }
            if not expected_ids:
                errors.append(f"messages[{idx}] assistant.tool_calls missing valid ids")
                idx += 1
                continue

            seen_ids = set()
            probe = idx + 1
            while probe < len(messages):
                nxt = messages[probe]
                if not isinstance(nxt, dict):
                    break
                if nxt.get("role") != "tool":
                    break
                tool_call_id = nxt.get("tool_call_id")
                if tool_call_id in expected_ids:
                    seen_ids.add(tool_call_id)
                probe += 1

            missing_ids = expected_ids - seen_ids
            if missing_ids:
                errors.append(
                    f"messages[{idx}] missing tool responses for ids: {sorted(missing_ids)}"
                )

            idx = probe
            continue

        idx += 1

    return len(errors) == 0, errors
