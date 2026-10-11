EXTRACTION_PROMPT = """
You are an information extraction system.

Goal:
Extract factual (subject, relation, object) triples from the document and assign a subject and object subtopic plus main topic.

Instructions:
1. Read the full document.
2. Extract all factual triples grounded in specific sentences.
3. Resolve pronouns so the triples use explicit referents.
4. For the subject and object of each triple, assign:
   - subtopic: a specific category
   - main_topic: a broader category
5. Cover all distinct supported facts, including details that may be needed to
   answer later questions. Do not impose a fixed limit on the number of triples.
6. Do not repeat the same fact with different wording. Do not invent facts or
   include unsupported or speculative statements.
7. Return only valid JSON and finish the complete JSON object.

Expected JSON format:
{
  "triples": [
    {
      "triple": ["subject", "relation", "object"],
      "sentence": "Supporting sentence from the document.",
      "subject": {
        "subtopic": "Specific category",
        "main_topic": "Broader category"
      },
      "object": {
        "subtopic": "Specific category",
        "main_topic": "Broader category"
      }
    }
  ]
}

Example document:
"Inez is an Estonian new media artist. She lives in Finland. She has composed electronic music."

Example JSON output:
{
  "triples": [
    {
      "triple": ["Inez", "is", "an Estonian new media artist"],
      "sentence": "Inez is an Estonian new media artist.",
      "subject": {
        "subtopic": "New Media Artist",
        "main_topic": "Art"
      },
      "object": {
        "subtopic": "Nationality",
        "main_topic": "Culture"
      }
    },
    {
      "triple": ["Inez", "lives in", "Finland"],
      "sentence": "She lives in Finland.",
      "subject": {
        "subtopic": "New Media Artist",
        "main_topic": "Art"
      },
      "object": {
        "subtopic": "Country",
        "main_topic": "Geography"
      }
    }
  ]
}

Input document:
{{document}}
"""
