SUBTOPIC_CHOICE_PROMPT = """
Goal:
For the given topic, select the numbered subtopic candidates that are most useful for answering the user's question.
Choose between {min_subtopics} and {max_subtopics} distinct subtopic IDs.
Return only one compact JSON object containing integer IDs.

Instructions:
1. Use only the integer IDs in the allowed candidates below.
2. Read the question in {question}.
3. Read the labels to make your choice, but return IDs only. Do not copy labels or explain.
4. If too many subtopics are relevant, keep the most useful ones.
5. If very few subtopics are relevant, still return at least {min_subtopics} IDs by picking the closest matches.
6. List each ID once. Stop immediately after the closing brace.

Output format:
{
  "subtopics": [1, 2]
}

Topic:
{{TOPIC_LABEL}}

Question:
{question}

Allowed subtopic IDs and labels:
{{SUBTOPIC_LIST}}
"""
