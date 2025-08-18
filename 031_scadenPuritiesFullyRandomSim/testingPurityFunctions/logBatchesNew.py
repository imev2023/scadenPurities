import os

LOG_FILE = "completed_batches_hvgs.log"

def load_completed_batches():
    if not os.path.exists(LOG_FILE):
        return set()
    with open(LOG_FILE, "r") as f:
        return set(line.strip() for line in f)

def mark_batch_completed(batch_id):
    with open(LOG_FILE, "a") as f:
        f.write(batch_id + "\n")
