TOPIC_CHOICE_PROMPT = """
Goal:
Select the topics from the numbered candidates that are directly relevant to the user's question.
Choose between {min_topics} and {max_topics} distinct topic IDs.
Return only one compact JSON object containing integer IDs.

Instructions:
1. Use only the integer IDs in the allowed candidates below.
2. Read the question in {{question}}.
3. Read the labels to make your choice, but return IDs only. Do not copy labels or explain.
4. If too many topics are relevant, keep the most useful ones.
5. If very few topics are relevant, still return at least {min_topics} IDs by picking the closest matches.
6. List each ID once. Stop immediately after the closing brace.

Output format:
{
  "topics": [1, 2]
}

Question:
{{question}}

Allowed topic IDs and labels:
{{TOPIC_LIST}}
"""


def get_topic_choice_prompt() -> str:
    return TOPIC_CHOICE_PROMPT
