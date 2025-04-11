import subprocess
import logging
import warnings
from core import logging_config
from misc.misc import *
warnings.filterwarnings('ignore')

GET_DATA = read_json("parameter.json")["get_data"]

task_list = [
    "DataAnalysis.py ",
    "ForecastModeling.py ",
    "Visualization.py "
]
if GET_DATA:
    task_list.insert(
        0, "GetAndProcessData.py "
    )
else:
    pass

for task in task_list:
    cmd = "python " + task
    logging.info(task)
    subprocess.run(cmd, shell=True)

