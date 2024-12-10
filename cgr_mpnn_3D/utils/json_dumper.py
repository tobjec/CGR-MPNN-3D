import json
import os

def json_dumper(fpath: str, dictionary: dict, add_training: str=None) -> None:
    
    try:
        with open(fpath, 'r') as file:
            data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}
    
    if add_training:
        dic_name = os.path.basename(add_training).rsplit('.', maxsplit=1)[0]
        
        data[dic_name].update(dictionary)
    else:
        data.update(dictionary)

    with open(fpath, 'w') as file:
        json.dump(data, file, indent=4)