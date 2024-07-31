import json
import os

def read_json(file_name):
    """
    :param file_name: json file name to read
    :type file_name: string
    :return: Loaded json file
    :rtype: dictionary
    """
    with open (os.path.join(os.getcwd(), "config", file_name)) as file:
        config = json.load(file)

    return config