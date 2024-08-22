import pandas as pd
import json
import os
import logging
from core import logging_config
from datetime import datetime

def read_json(file_name):
    """
    :param file_name: json file name to read
    :type file_name: string
    :return: Loaded json file
    :rtype: dictionary
    """
    try:
        with open (os.path.join(os.getcwd(), "config", file_name)) as file:
            config = json.load(file)
        return config
    except FileNotFoundError:
        logging.error(f"File {file_name} not found")
        return None

def get_business_day(date):
    """
    :param date: Base date for calculating the previous business day
    :type date: string or date object
    :return: Previous business day for given date. 
    If param date is still a business day, the function returns param date
    :rtype: string
    """
    if not isinstance(date, datetime):
        date = pd.to_datetime(date)
    is_bday = pd.bdate_range(start=date, end=date).shape[0] > 0
    if not is_bday:
        bday = date - pd.offsets.BDay(1)
        bday = bday.strftime(format="%Y-%m-%d")
    else:
        bday = date.strftime(format="%Y-%m-%d")
    return bday

def get_last_business_day():
    """Function calculates the last business day
    compared to actual date

    :return: Last busines day based on actual date
    :rtype: String
    """
    now = datetime.now()
    last_bday = now - pd.offsets.BDay(1)
    last_bday = last_bday.strftime(format="%Y-%m-%d")
    return last_bday