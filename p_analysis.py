import os
import ollama
from p_utility_function import parse_json_response, format_timestamp

def analyze_frame_worker(frame, timestamps, extracted_frames_dir, system_context, user_context, ollama_options):
    path = os.path.join(extracted_frames_dir, frame)

    if not os.path.exists(path):
        return None, f"Missing file: {frame}"

    try:
        response = ollama.chat(
            model="llava:7b",
            messages=[
                {"role": "system", "content": system_context},
                {"role": "user", "content": user_context, "images": [path]},
            ],
            options=ollama_options,
        )

        parsed = parse_json_response(response["message"]["content"])
        if not parsed:
            return None, response["message"]["content"]

        parsed.update({
            "frame_file": frame,
            "timestamp": format_timestamp(timestamps.get(frame, 0)),
        })

        return parsed, None

    except Exception as e:
        return None, str(e)
