import hashlib

def get_hash(s):
    return hashlib.md5(s.encode('utf-8')).digest().hex()

def node_id(name, label=None):
    if label:
        return get_hash(f"{name.lower().replace(' ', '_')}::{label.lower().replace(' ', '_')}")
    else:
        return get_hash(f"{name.lower().replace(' ', '_')}")
    
def node_result(name, properties=['*']):
    return f"""{name}: {name}{{{', '.join('.{}'.format(p) for p in properties)}}}"""
    

def get_source_value(properties, default):
    keys = ['file_path', 'url', 'source']
    for key in keys:
        if key in properties:
            return properties[key]
    return default
    