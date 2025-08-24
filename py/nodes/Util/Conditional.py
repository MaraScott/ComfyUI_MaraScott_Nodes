from ...utils.constants import get_name, get_category
from ...utils.helper import AlwaysEqualProxy, to_bool
from ...utils.log import log

any_type = AlwaysEqualProxy("*")

class IsTrue_v1:

    NAME = "Is True ?"
    SHORTCUT = "c"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "boolean": (any_type,),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("boolean",)
    FUNCTION = "fn"
    CATEGORY = get_category("Utils")

    def fn(self, boolean=None):
        return (to_bool(boolean),)

class IsEqual_v1:

    NAME = "Is Equal ?"
    SHORTCUT = "c"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any_1": (any_type,),
                "any_2": (any_type,)
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("boolean",)
    FUNCTION = "fn"
    CATEGORY = get_category("Utils")

    def fn(self, any_1=None, any_2=None):
        return (True if any_1 == any_2 else False,)

class IsNone_v1:
    
    NAME = "Is None ?"
    SHORTCUT = "c"
    
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
    CATEGORY = get_category("Utils")

    def fn(self, any):
        return (True if any is None else False,)

class IsEmpty_v1:
    
    NAME = "Is Empty ?"
    SHORTCUT = "c"
    
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
    CATEGORY = get_category("Utils")

    def fn(self, any):
        return (True if any == "" else False,)

class IsEmptyOrNone_v1:
    
    NAME = "Is Empty Or None ?"
    SHORTCUT = "c"
    
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
    CATEGORY = get_category("Utils")

    def fn(self, any):
        isEmptyOrNone = False
        is_empty = IsEmpty_v1().fn(any=any)[0]
        is_none = IsNone_v1().fn(any=any)[0]
        if is_empty or is_none:
            isEmptyOrNone = True
        return (isEmptyOrNone,)
