from pathlib import Path
import subprocess, sys, time, json

ROOT=Path(__file__).resolve().parent
steps=["01_prepare_data.py","02_baseline_cv.py","03_tune_stacking.py","04_final_evaluation.py","05_generate_figures.py"]
t0=time.perf_counter()
for step in steps:
    print("\n"+"="*80+f"\nRUNNING {step}\n"+"="*80,flush=True)
    subprocess.run([sys.executable,str(ROOT/step)],cwd=ROOT,check=True)
total=time.perf_counter()-t0
log=ROOT/"outputs/logs/end_to_end_runtime.json"
log.parent.mkdir(parents=True,exist_ok=True)
log.write_text(json.dumps({"complete_pipeline_seconds":total,"steps":steps},indent=2),encoding="utf-8")
print(f"\nCOMPLETE END-TO-END PIPELINE RUNTIME: {total:.3f} seconds")
print(f"Runtime saved to: {log}")
