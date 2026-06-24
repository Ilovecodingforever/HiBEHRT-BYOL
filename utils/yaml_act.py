import yaml


def yaml_load(path):
    with open(path, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


def yaml_save(cfg, path):
    with open(path, 'w') as ymlfile:
        yaml.dump(
            cfg,
            ymlfile,
            default_style=None,
            default_flow_style=None,
            line_break=12
        )