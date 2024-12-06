
NAMESPACE='MaraScott'

def get_name(name):
    return '{} ({})'.format(name, NAMESPACE)

def get_category(sub_dirs = None):
    if sub_dirs is None:
        return NAMESPACE
    else:
        return "\ud83d\udc30 {}/{}".format(NAMESPACE, sub_dirs)
