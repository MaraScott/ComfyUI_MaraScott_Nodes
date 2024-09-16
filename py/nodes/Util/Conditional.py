from ...utils.helper import AlwaysEqualProxy

any_type = AlwaysEqualProxy("*")

class IsNone_v1:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any": (any_type,)
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("boolean",)
    FUNCTION = "fn"
    CATEGORY = "MaraScott/Util"

    def fn(self, any):
        return (True if any is None else False,)

class IsEmpty_v1:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any": (any_type,)
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("boolean",)
    FUNCTION = "fn"
    CATEGORY = "MaraScott/Util"

    def fn(self, any):
        return (True if any == "" else False,)

class IsEmptyOrNone_v1:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any": (any_type,)
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("boolean",)
    FUNCTION = "fn"
    CATEGORY = "MaraScott/Util"

    def fn(self, any):
        isEmptyOrNone = False
        is_empty = IsEmpty_v1().fn(any=any)[0]
        is_none = IsNone_v1().fn(any=any)[0]
        if is_empty or is_none:
            isEmptyOrNone = True
        return (isEmptyOrNone,)
