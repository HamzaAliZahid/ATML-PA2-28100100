"""One-cell unattended run for Kaggle "Save & Run All" (or any 2-GPU machine).

Runs both GPU queues in parallel, keeps going past individual failures (KEEP_GOING=1),
then builds the figures/tables and zips results into /kaggle/working/pa2_results.zip.
Usage (from the repo root):  python run/kaggle_all.py
"""
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
os.makedirs("logs", exist_ok=True)
procs = {}
for gpu in ["0", "1"]:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1", KEEP_GOING="1")
    procs[gpu] = subprocess.Popen(["bash", f"run/gpu{gpu}.sh"], stdout=open(f"logs/gpu{gpu}.log", "a"),
                                  stderr=subprocess.STDOUT, env=env, start_new_session=True)
print("started", {g: p.pid for g, p in procs.items()}, flush=True)

t0 = time.time()
while any(p.poll() is None for p in procs.values()):
    time.sleep(300)
    done = sorted(os.listdir("results/.done")) if os.path.isdir("results/.done") else []
    print(f"[{(time.time()-t0)/60:6.1f} min] finished steps: {len(done)}/25  ->  {done[-1] if done else '-'}", flush=True)
    for g in ["0", "1"]:
        try:
            print(f"   gpu{g}: {open(f'logs/gpu{g}.log', errors='ignore').read().splitlines()[-1][:150]}", flush=True)
        except Exception:
            pass

print("queues exited:", {g: p.returncode for g, p in procs.items()}, flush=True)
fails = [l for g in ["0", "1"] for l in open(f"logs/gpu{g}.log", errors="ignore").read().splitlines() if "FAILED" in l]
print("FAILED steps:", fails or "none", flush=True)
subprocess.run([sys.executable, "-m", "scripts.make_figures"])
out = "/kaggle/working/pa2_results" if os.path.isdir("/kaggle/working") else os.path.join(ROOT, "pa2_results")
subprocess.run(["zip", "-qr", out + ".zip", "results", "report", "logs"])
print("Saved", out + ".zip", flush=True)
