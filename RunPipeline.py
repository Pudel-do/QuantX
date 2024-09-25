import subprocess
import logging
from core import logging_config

task_list = [
    "DataAnalysis.py ",
    "ForecastModeling.py "
]

for task in task_list:
    cmd = "python " + task
    logging.info(task)
    subprocess.run(cmd, shell=True)

