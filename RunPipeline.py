import subprocess
import logging
from core import logging_config

task_list = [
    #"GetAndProcessData.py ",
    "DataAnalysis.py ",
    "ForecastModeling.py ",
    "PortfolioConstruction.py ",
    "Visualization.py "
]

for task in task_list:
    cmd = "python " + task
    logging.info(task)
    subprocess.run(cmd, shell=True)

