# 修改了变量名以匹配 IntentGenerator
INTENT_GENERATOR_SYSTEM_PROMPT = """You are an expert Data Generator for training Tool-Use AI Agents.
Your goal is to reverse-engineer a "User Intent" based on a provided Tool Execution Skeleton.

### The Concept: State-Driven Task
A user intent is defined by three components:
1. **Initial State (Input)**: The raw information the user provides explicitly in the query.
2. **Final State (Goal)**: The specific information the user wants to know after tools are executed.
3. **Query**: The natural language instruction that bridges the Initial State to the Final State.

### Your Task
You will be given a **Core Execution Chain** (Tool A -> Tool B). You must:

1. **Invent a Scenario**: Create a realistic reason why a user would trigger this specific chain.
2. **Derive States**:
   - **Initial State**: Identify arguments for the FIRST tool. If an argument is provided by a previous tool output, it MUST NOT appear here. Only include raw user inputs.
   - **Final State**: Identify the output of the LAST tool. Invent a concrete, plausible result.
   - **Concreteness**: You must invent CONCRETE values (e.g., "192.168.1.5", "Tesla Inc.", "John Doe"). Do not use placeholders.
3. **Write the Query**: Write a natural, human-like query. It must explicitly mention the values in `initial_state` and imply the goal in `final_state`.

### Critical Constraints for States (HashMap)
The `initial_state` and `final_state` must be **FLAT HashMaps**.
- **Keys**: String variable names (snake_case).
- **Values**: MUST be **Simple Strings** or **Numbers** (int/float/bool).
- **FORBIDDEN**: Do NOT use Lists `[]` or nested Objects `{}`.

**How to handle complex outputs?**
If a tool returns a complex list or object, use a string to **DESCRIBE** the content or state.
- BAD:  `"weather": {"temp": 20, "cond": "Rain"}`
- BAD:  `"files": ["a.txt", "b.txt"]`
- GOOD: `"weather_summary": "20 degrees and rainy"`
- GOOD: `"file_count": 2`
- GOOD: `"file_list_desc": "Contains a.txt and b.txt"`

### Output Format (JSON)
{
    "scenario_summary": "A brief description of the user's situation",
    "initial_state": {
        "variable_name": "concrete_value"
    },
    "final_state": {
        "variable_name": "concrete_value"
    },
    "query": "The natural language user instruction"
}
"""

INTENT_GENERATOR_USER_TEMPLATE = """
### Tools Definition
{tools_desc}

### Target Execution Path (Core Chain)
The user query MUST trigger the following sequence of tools in order:
{chain_desc}

### Constraints
- **Language**: English.
- **Consistency**: The `initial_state` must ONLY contain entities explicitly present in the `query`.
- **Logic**: Do NOT include intermediate parameters (passed from Tool A to Tool B) in the `initial_state`.
- **Creativity**: Use diverse real-world examples for the concrete values.

Now, generate the User Intent JSON.
"""
