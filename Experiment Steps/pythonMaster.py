import os
import subprocess


for f in os.listdir():
    if f.startswith("PPO") or f.startswith("A2C"):
        if (f != "PPO_25000_bzip2__instcount_plainmodel"):
            os.chdir(f)
            print(os.getcwd())
            print("Running.. experiment in"+f)
            subprocess.run(["/bin/python3", "ExperimentalSetup.py"])
            os.chdir("..")
            print(os.getcwd())