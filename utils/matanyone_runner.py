"""
MatAnyone2 inference runner
Wraps the MatAnyone2 inference script for segment mask generation
"""

import subprocess
import os
import sys
import json
import time
import tempfile
from typing import Callable


MATANYONE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "MatAnyone2")
sys.path.insert(0, MATANYONE_DIR)


def run_matanyone_inference(jobs: list[dict], on_segment_done: Callable[[str], None] = None) -> list[str]:
    """
    Run MatAnyone inference on a list of segment jobs.
    
    All segments are processed in a single subprocess invocation
    Progress/tqdm output goes to the terminal via inherited stderr
    
    Args:
        jobs: list of job dicts, each containing:
            input_path, mask_path, output_path, max_size, warmup,
            erode, dilate, suffix, fps, op_num, total_ops, label, duration
        on_segment_done: optional callback called with output path after each segment
        
    Returns: list of output alpha video paths in job order
    """
    inference_script = os.path.join(MATANYONE_DIR, "inference_matanyone2.py")
    
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    
    # retry logic for intermittent crashes
    max_retries = 4
    remaining_jobs = list(jobs)
    completed_paths = []
    
    for attempt in range(max_retries):
        # write remaining jobs to a temp JSON file
        jobs_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False, dir=os.path.dirname(inference_script)
        )
        try:
            json.dump(remaining_jobs, jobs_file)
            jobs_file.close()
            
            cmd = [sys.executable, inference_script, '--jobs', jobs_file.name]
            
            # stdout=PIPE to capture DONE sentinels, stderr inherited for tqdm/status display
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env, text=True)
            
            batch_completed = []
            for line in process.stdout:
                line = line.strip()
                if line.startswith("DONE:"):
                    output_path = line[5:]
                    batch_completed.append(output_path)
                    if on_segment_done:
                        on_segment_done(output_path)
            
            process.wait()
            
            completed_paths.extend(batch_completed)
            
            if process.returncode == 0:
                break
            
            # process crashed — retry with remaining jobs
            remaining_jobs = remaining_jobs[len(batch_completed):]
            
            if not remaining_jobs:
                break  # all jobs completed before crash
            
            if attempt < max_retries - 1:
                # re-number op_num for remaining jobs so status display stays correct
                start_op = completed_paths.__len__() + 1
                for i, job in enumerate(remaining_jobs):
                    job['op_num'] = start_op + i
                time.sleep(3.0)
            else:
                raise RuntimeError(
                    f"❌ MatAnyone inference failed after {max_retries} attempts "
                    f"(code {process.returncode}, {len(remaining_jobs)} segments remaining)"
                )
        finally:
            if os.path.exists(jobs_file.name):
                os.remove(jobs_file.name)
    
    return completed_paths