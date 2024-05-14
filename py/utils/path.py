import os
from .log import *

_base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
base_dir = os.path.abspath(_base_dir)
_root_dir = os.path.join(_base_dir, "..", "..")
root_dir = os.path.abspath(_root_dir)
_web_dir = os.path.join(_root_dir, "web", "extensions", "marascott")
web_dir = os.path.realpath(_web_dir)

os.makedirs(web_dir, exist_ok=True)
WEB_DIR = web_dir

sessions_dir = os.path.join(WEB_DIR, "sessions")
os.makedirs(sessions_dir, exist_ok=True)
SESSIONS_DIR = sessions_dir

profiles_dir = os.path.join(WEB_DIR, "profiles")
os.makedirs(profiles_dir, exist_ok=True)
PROFILES_DIR = profiles_dir

