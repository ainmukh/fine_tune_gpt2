import json
from collections import OrderedDict
from pathlib import Path


def read_json(file_name):
    file_name = Path(file_name)
    with file_name.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def create_config(args):
    if not isinstance(args, tuple):
        args = args.parse_args()

    msg_no_cfg = "Configuration file need to be specified. Add '-c config.json', for example."
    assert args.config is not None, msg_no_cfg
    cfg_file_name = Path(args.config)

    config = read_json(cfg_file_name)
    return config
