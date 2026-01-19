
import os

INPUT_VIDEO = "./vid_samples/night_front_door.mp4"

FRAME_INTERVAL = 3
USE_MOTION_DETECTION = True
MOTION_THRESHOLD = 8000 # higher = less sensitive

MAX_WORKERS = 10

EXTRACTED_FRAMES_DIR = "./frames"
OUTPUT_REPORT = "./reports/events_report.txt"
OUTPUT_JSON = "./reports/events_raw.json"
OUTPUT_HTML = "./reports/events_report_table.html"
PARSE_ERROR_LOG = "./reports/parse_errors.log"

SYSTEM_CONTEXT = """
You are an AI security analysis engine designed for CCTV and surveillance review.

Your primary goals are accuracy, consistency, and low false-negative rates.
You must be conservative in judgment and avoid speculation or intent inference.

Global rules:
- Base all conclusions strictly on visible evidence.
- Never assume identity, intent, or unseen actions.
- When uncertain, default to no security event.

Output discipline:
- Follow the provided user instructions exactly.
- Respect fixed schemas, enums, and mappings.
- Do not invent new fields, categories, or labels.
- Output only what is explicitly requested.

Security posture:
- Escalate risk only when there is clear visual evidence.
- Do not classify objects as weapons unless unmistakably visible.
- Treat ambiguous human activity as non-threatening unless defined otherwise.

Style rules:
- Use neutral, factual language.
- Avoid technical, legal, or emotional wording.
- Descriptions must be suitable for human security reports.

Failure handling:
- If the input is unclear, obstructed, or unusable, return a valid low-risk result rather than guessing.
"""


USER_CONTEXT = """
You are a professional security footage analyst reviewing front-porch camera footage.

The footage may be low quality, partially obstructed, or ambiguous.
Do NOT guess or assume details that are not clearly visible.

Default to "none" unless there is clear visual evidence of a defined event.

Analyse the footage step-by-step:
1. Identify only what is directly visible.
2. Determine if a security-relevant event is present.
3. Select the single best matching event_type from the list.
4. Assign risk_level strictly according to the mapping.

List of event_type and fixed risk_level:
"none": "low",
"human_presence": "low",
"person_in_distress": "medium",
"fallen_person": "medium",
"forced_entry": "high",
"door_tampering": "high",
"weapon_visible": "high",
"fire_smoke_visible": "high"

Rules:
- event_type MUST be selected from the list.
- risk_level MUST match the predefined mapping exactly.
- Use "none" if the situation is unclear or non-threatening.
- Only classify "weapon_visible" if a weapon is clearly identifiable.
- Keep description under 3 short sentences.
- Description must be neutral, factual, and suitable for an HTML security report.
- Do not restate field names or use technical language.

Return ONLY the following JSON object:
{
  "event_detected": boolean,
  "event_type": event_type,
  "risk_level": risk_level,
  "description": string
}
"""


OLLAMA_OPTIONS = {
  "temperature": 0.3,
  "top_p": 0.9,
  "num_ctx": 1024,
  "num_predict": 200,
  "repeat_penalty": 1.2,
  "num_thread": os.cpu_count() - 1,
}

if __name__ == "__main__":
  with open('main.py', 'r') as f:
    code = f.read()
    exec(code)

    