"""Prompts for graph building module."""

# System prompt for verifying edges
VERIFY_SINGLE_EDGE_SYSTEM_PROMPT = """You are an AI data flow assistant.
Task: Determine if the **semantic intent** of the 'Producer Tool' suggests it produces information suitable for the 'Consumer Parameter'.

Context: You only have text descriptions. You must **infer** the output based on the tool's name and description.
Criteria:
- If the Producer likely returns the data needed by the Consumer, return true.
- If they are unrelated, return false.

Return JSON: {"valid": true, "reason": "short explanation"}"""

# System prompt for dynamic categorization
AUTO_CATEGORIZE_SINGLE_SYSTEM_PROMPT = """You are a Taxonomy Architect.
Classify the given tool into a SINGLE category.

**Current Category List**:
[{existing_cats_str}]

**Rules**:
1. **Reuse First**: Priority use existing categories.
2. Create New: Only create if the tool is completely unrelated to existing categories.
  - Use broad, standard terms (e.g., use 'Medical', not 'Cardiology').
  - Avoid Synonyms: Do not create 'Financial' if 'Finance' exists.
3. **Format**: Return JSON: {{"category": "SelectedCategory"}}"""
