
NAMESPACE='MaraScott'
ICON='🐰'

def get_name(name, version = "1", shortcut = "m", vendor = ""):
    v = f" - v{version}"
    s = f" /{shortcut}"
    vd = f" (from {vendor})"
    return '{} {}{}{}{}'.format(ICON, name, v, s, vd)

def get_category(sub_dirs = None):
    if sub_dirs is None:
        return NAMESPACE
    else:
        return "{}/{}".format(f"{ICON} {NAMESPACE}", sub_dirs)
