import inspect 
import re 

NAMESPACE='MaraScott'
ICON='🐰'

def _get_version():
    frame = inspect.currentframe()
    try:
        class_name = frame.f_back.f_globals.get('__qualname__', None)
        if not class_name:
            class_name = frame.f_back.f_code.co_name
    finally:
        del frame

    version_match = re.search(r'_v(\d+)', class_name) if class_name else None
    version = version_match.group(1) if version_match else None
    
    return version

def get_name(name, shortcut = "", vendor = ""):

    version = _get_version()
    v = f" - v{version}" if version is not None else ""
    s = f" /{shortcut}" if shortcut != "" else ""
    vd = f" (from {vendor})" if vendor != "" else ""
    
    return '{} {}{}{}{}'.format(ICON, name, v, s, vd)

def get_category(sub_dirs = None):
    if sub_dirs is None:
        return NAMESPACE
    else:
        return "{}/{}".format(f"{ICON} {NAMESPACE}", sub_dirs)
