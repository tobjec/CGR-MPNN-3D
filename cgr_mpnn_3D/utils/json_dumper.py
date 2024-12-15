import json
import os


def json_dumper(fpath: str, dictionary: dict, add_training: str = None) -> None:
    """
    Routine to dump dictionary into .json file.

    Args:
        fpath (str): Path to the to be saved file.
        dictionary (dict): To be saved dictionary.
        add_training (str, optional): Name of saved pytorch model
                                      to add also training data.
                                      Defaults to None.
    """

    try:
        with open(fpath, "r") as file:
            data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}

    if add_training:
        dic_name = os.path.basename(add_training).rsplit(".", maxsplit=1)[0]

        data[dic_name].update(dictionary)
    else:
        data.update(dictionary)

    with open(fpath, "w") as file:
        json.dump(data, file, indent=4)
