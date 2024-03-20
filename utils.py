import yaml
from yacs.config import CfgNode

def get_config_from_yaml(yaml_file):
    def get_node(dict_val):
        if not isinstance(dict_val, dict):
            return dict_val 
        
        res = CfgNode()
        for key, val in dict_val.items():
            res[key] = get_node(val)
        return res 

    with open(yaml_file) as f:
        cfg_dict = yaml.safe_load(f.read())
        cfg_node = get_node(cfg_dict)

    return cfg_node.clone() 
