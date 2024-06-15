import json
from tools.datautils import DataUtils
from os.path import join,split, exists
import subprocess, sys, time
from tools.debug import Debug


debug    = Debug()
dutils   = DataUtils()


dict_path  = join(dutils.DEVPATH,"Analytics","tractography","progress.json")
with open(dict_path, 'r') as file:
    progress_dict = json.load(file)

EXEC_PATH = join(dutils.DEVPATH,"Analytics","tractography","struct_connectivity.py")
ENV_PATH = "/home/flucchetti/miniconda3/envs/analytics_env/bin/python3"
for subject_id,sessions in progress_dict.items():
    for session in sessions.keys():
        state = progress_dict[subject_id][session]["progress"]
        if state!=3:
            try:
                subprocess.run([ENV_PATH, EXEC_PATH,subject_id,session])
            except subprocess.CalledProcessError:
                debug.error("An error occurred during execution.")
                continue
            except KeyboardInterrupt:
                debug.error("KeyboardInterrupt")
                sys.exit()
        else:
            debug.success(f"Already processed: sub-{subject_id}_ses-{session}")



