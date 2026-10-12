MAX_TRIPLES = 110

TRIPLES_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["triples"],
    "properties": {
        "triples": {
            "type": "array",
            "maxItems": MAX_TRIPLES,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["triple", "sentence", "subject", "object"],
                "properties": {
                    "triple": {
                        "type": "array",
                        "minItems": 3,
                        "maxItems": 3,
                        "items": {"$ref": "#/$defs/text"},
                    },
                    "sentence": {"$ref": "#/$defs/text"},
                    "subject": {"$ref": "#/$defs/category"},
                    "object": {"$ref": "#/$defs/category"},
                },
            },
        },
    },
    "$defs": {
        "text": {"type": "string", "minLength": 1},
        "category": {
            "type": "object",
            "additionalProperties": False,
            "required": ["subtopic", "main_topic"],
            "properties": {
                "subtopic": {"$ref": "#/$defs/text"},
                "main_topic": {"$ref": "#/$defs/text"},
            },
        },
    },
}

SYSTEM_PROMPT = f"""You are an information extraction system.
Return only a complete JSON object matching the supplied schema.
Extract at most {MAX_TRIPLES} triples per chunk; an empty triples array is valid.
Every fact must be supported by the input document, and sentence must contain
the original text supporting that fact. Do not invent facts."""

EXTRACTION_PROMPT = """
You are an information extraction system.

Goal:
Extract factual (subject, relation, object) triples from the document and assign a subject and object subtopic plus main topic.

Instructions:
1. Read the full document.
2. Extract at most {{max_triples}} factual triples grounded in specific sentences.
3. Resolve pronouns so the triples use explicit referents.
4. For the subject and object of each triple, assign:
   - subtopic: a specific category
   - main_topic: a broader category
5. Choose distinct supported facts within this limit, including useful details.
   Set sentence to original supporting text from the document.
6. Do not repeat the same fact with different wording. Do not invent facts or
   include unsupported or speculative statements.
7. Return only valid JSON matching the schema and finish the complete JSON object.
   Return {"triples": []} when no supported facts can be extracted.

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
""".replace("{{max_triples}}", str(MAX_TRIPLES))
