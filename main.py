import os
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from p_config import *
from p_analysis import *
from p_frame_extract import *
from p_utility_function import *
from p_write_reports import *


if __name__ == "__main__":
    start = time.perf_counter()

    print("=" * 60)
    print("VIDEO ANALYSIS PIPELINE")
    print("=" * 60)

    # extract frames block
    print("\n[1] Extracting frames...")
    timestamps = extract_frames(INPUT_VIDEO, EXTRACTED_FRAMES_DIR, interval_sec=FRAME_INTERVAL)

    # analysis block
    print("\n[2] Analyzing frames...")
    frames = sorted(
        (f for f in os.listdir(EXTRACTED_FRAMES_DIR) if f.endswith(".jpg")),
        key=read_num_in_filename,
    )

    events, errors = [], 0

        # analysis process pool sub block
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        print(f"process pool max workers: {executor._max_workers}")
        futures = {
            executor.submit(analyze_frame_worker, f, timestamps, EXTRACTED_FRAMES_DIR, SYSTEM_CONTEXT, USER_CONTEXT, OLLAMA_OPTIONS): f
            for f in frames
        }

        for i, future in enumerate(as_completed(futures), 1):
            frame = futures[future]
            parsed, error = future.result()

            if parsed:
                events.append(parsed)
                print(f"[OK] {i}/{len(frames)} {frame}")
            else:
                errors += 1
                log_parse_error(frame, error, output=PARSE_ERROR_LOG)
                print(f"[FAIL] {i}/{len(frames)} {frame}")

    events.sort(key=lambda e: e["timestamp"])

    # report block
    print("\n[3] Generating report...")
    with open(OUTPUT_JSON, "w") as f: # write events_raw.json
        json.dump(events, f, indent=2)

    json2html_convert(OUTPUT_JSON, OUTPUT_HTML) # convert events_raw.json to events_report_table.html

    elapsed = time.perf_counter() - start
    print("\nCompleted")
    print(f"Frames analyzed: {len(events)}")
    print(f"Errors: {errors}")
    print(f"Total time: {elapsed:.2f}s ({elapsed/60:.1f} min)") # timed entire script
