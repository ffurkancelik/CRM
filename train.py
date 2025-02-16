import os
import subprocess

base_path = os.path.dirname(os.path.abspath(__file__))

scripts = ["merge_dataset.py", "clean_dataset.py", "run_clv.py", "run_churn.py", "segmentation.py"]

for e in scripts:
    try:
        print("Script Running: ", e)
        result = subprocess.run(["python", os.path.join(base_path, e)], capture_output=True, text=True)
    except Exception as err:
        print("Error: ", err)
