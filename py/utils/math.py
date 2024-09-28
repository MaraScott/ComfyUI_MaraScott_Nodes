from .helper import AlwaysEqualProxy

any_type = AlwaysEqualProxy("*")

COMPARE_FUNCTIONS = {
    "a == b": lambda a, b: a == b,
    "a != b": lambda a, b: a != b,
    "a < b": lambda a, b: a < b,
    "a > b": lambda a, b: a > b,
    "a <= b": lambda a, b: a <= b,
    "a >= b": lambda a, b: a >= b,
}

class Compare:
    @classmethod
    def INPUT_TYPES(s):
        compare_functions = list(COMPARE_FUNCTIONS.keys())
        return {
            "required": {
                "a": (any_type, {"default": 0}),
                "b": (any_type, {"default": 0}),
                "comparison": (compare_functions, {"default": "a == b"}),
            },
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("boolean",)
    FUNCTION = "compare"
    CATEGORY = "EasyUse/Logic"

    def compare(self, a, b, comparison):
        return (COMPARE_FUNCTIONS[comparison](a, b),)
    
class mathIntOperation:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": ("INT", {"default": 0, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff, "step": 1}),
                "b": ("INT", {"default": 0, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff, "step": 1}),
                "operation": (["add", "subtract", "multiply", "divide", "modulo", "power"],),
            },
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "fn"

    CATEGORY = "MaraScott/Math"

    def fn(self, a, b, operation):
        if operation == "add":
            return (a + b,)
        elif operation == "subtract":
            return (a - b,)
        elif operation == "multiply":
            return (a * b,)
        elif operation == "divide":
            return (a // b,)
        elif operation == "modulo":
            return (a % b,)
        elif operation == "power":
            return (a ** b,)