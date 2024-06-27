import os
import pickle
from .... import __CACHE_DIR__
from ...utils.log import log

class MS_Cache():
    
    def isset(key):
        if not os.path.exists(os.path.join(__CACHE_DIR__, key)):
            return False
        return True
    
    def set(key, value):
        with open(os.path.join(__CACHE_DIR__, key), 'wb') as f:
            pickle.dump(value, f)

    def get(key, default_value = None):
        value = default_value
        if MS_Cache.isset(key):
            with open(os.path.join(__CACHE_DIR__, key), 'rb') as f:
                value = pickle.load(f)
        return value
    
    def cache_delete(key):
        cache_path = os.path.join(__CACHE_DIR__, key)
        if os.path.exists(cache_path):
            os.remove(cache_path)
        return True
