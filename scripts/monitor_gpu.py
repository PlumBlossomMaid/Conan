"""Monitor GPU utilization every 200ms. Ctrl+C to stop."""
import subprocess, time
try:
    while True:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True)
        util, mem = r.stdout.strip().split(", ")
        print(f"GPU: {util}%  MEM: {mem}MB")
        time.sleep(0.2)
except KeyboardInterrupt:
    pass
