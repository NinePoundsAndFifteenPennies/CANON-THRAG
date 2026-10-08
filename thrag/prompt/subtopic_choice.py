# Topic and candidate list stay fixed across questions for the same topic.
SUBTOPIC_CHOICE_PROMPT = """
Goal:
For the given topic, select the subtopics from the provided list that are most useful for answering the user's question.
Choose between {min_subtopics} and {max_subtopics} subtopics.
Return only valid JSON.

Instructions:
1. Use only the subtopics provided in the "Allowed subtopics" section below.
2. Read the question in the "Question" section below.
3. Preserve the original subtopic strings exactly as provided.
4. If too many subtopics are relevant, keep the most useful ones.
5. If very few subtopics are relevant, still return at least {min_subtopics} subtopics by picking the closest matches.

Output format:
{
  "subtopics": ["SubtopicLabel1", "SubtopicLabel2"]
}

Topic:
{{TOPIC_LABEL}}

Allowed subtopics:
{{SUBTOPIC_LIST}}

Question:
{question}
"""
