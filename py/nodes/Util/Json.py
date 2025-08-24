import json
from ...utils.constants import get_name, get_category
from ...utils.log import log

class JsonList2JsonObj_v1:

    NAME = "Json List to Json Object"
    SHORTCUT = "u"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_list": ("STRING",)   # input always string from UI
            },
            "optional": {}
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json_obj",)
    FUNCTION = "fn"

    CATEGORY = get_category("Utils")

    def fn(self, json_list):
        json_obj = {}  # default to empty

        try:
            parsed = json.loads(json_list) if isinstance(json_list, str) else json_list

            if isinstance(parsed, list):
                json_obj = {i: item for i, item in enumerate(parsed)}
            else:
                log.warning("Input is not a JSON list")

        except Exception as e:
            log.warning(f"Invalid input: {e}")

        return (json.dumps(json_obj),)
