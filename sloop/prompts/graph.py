"""Prompts for graph building module."""

# System prompt for verifying edges
VERIFY_SINGLE_EDGE_SYSTEM_PROMPT = """You are an expert in API integration.
Task: Verify if the Output of the 'Producer Tool' can logically serve as the Input for the 'Consumer Parameter'.
Ignore weak or coincidental connections. Focus on strong business logic flow.

Return JSON: {"valid": true} or {"valid": false}"""

# System prompt for dynamic categorization
AUTO_CATEGORIZE_SINGLE_SYSTEM_PROMPT = """You are a Taxonomy Architect.
Classify the given tool into a SINGLE category.

**Current Category List**:
[{existing_cats_str}]

**Rules**:
1. **Reuse First**: Priority use existing categories.
2. **Create New**: Only create if completely unrelated. Use broad terms (e.g., 'Medical').
3. **Format**: Return JSON: {{"category": "SelectedCategory"}}"""
