"""Sample GPU utilization for N seconds and report average."""
import subprocess
import time

duration = 30  # seconds
interval = 0.2

utils = []
print(f"Sampling GPU for {duration}s every {interval*1000:.0f}ms...")
start = time.time()
while time.time() - start < duration:
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True)
    util, mem = r.stdout.strip().split(", ")
    utils.append(int(util))
    time.sleep(interval)

avg = sum(utils) / len(utils)
peak = max(utils)
low = min(utils)
samples = len(utils)
print(f"\n=== {samples} samples over {duration}s ===")
print(f"  Min:  {low}%")
print(f"  Max:  {peak}%")
print(f"  Avg:  {avg:.1f}%")
