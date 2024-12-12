import torch
try:
    from comfy_execution.graph_utils import GraphBuilder, is_link
except:
    GraphBuilder = None
from .tools import VariantSupport
from ...utils.constants import get_name, get_category
from ...utils.helper import AlwaysEqualProxy


MAX_FLOW_NUM = 10
any_type = AlwaysEqualProxy("*")

NUM_FLOW_SOCKETS = 5
@VariantSupport()
class ForLoopOpen_v1:
    
    def __init__(self):
        pass

    NAME = get_name("For Loop Open", "l")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "remaining": ("INT", {"default": 1, "min": 0, "max": 100000, "step": 1}),
            },
            "optional": {
                f"initial_value{i}": ("*",) for i in range(1, NUM_FLOW_SOCKETS)
            },
            "hidden": {
                "initial_value0": ("*",)
            }
        }

    RETURN_TYPES = tuple(["FLOW_CONTROL", "INT",] + ["*"] * (NUM_FLOW_SOCKETS-1))
    RETURN_NAMES = tuple(["flow_control", "remaining"] + [f"value{i}" for i in range(1, NUM_FLOW_SOCKETS)])
    FUNCTION = "fn"

    CATEGORY = get_category("Loop/Flow")

    def fn(self, remaining, **kwargs):
        graph = GraphBuilder()
        if "initial_value0" in kwargs:
            remaining = kwargs["initial_value0"]
        while_open = graph.node("MaraScottForLoopWhileOpen_v1", condition=remaining, initial_value0=remaining, **{(f"initial_value{i}"): kwargs.get(f"initial_value{i}", None) for i in range(1, NUM_FLOW_SOCKETS)})
        outputs = [kwargs.get(f"initial_value{i}", None) for i in range(1, NUM_FLOW_SOCKETS)]
        return {
            "result": tuple(["stub", remaining] + outputs),
            "expand": graph.finalize(),
        }

@VariantSupport()
class ForLoopClose_v1:
    
    def __init__(self):
        pass

    NAME = get_name("For Loop Close", "l")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flow_control": ("FLOW_CONTROL", {"rawLink": True}),
            },
            "optional": {
                f"initial_value{i}": ("*",{"rawLink": True}) for i in range(1, NUM_FLOW_SOCKETS)
            },
        }

    RETURN_TYPES = tuple(["*"] * (NUM_FLOW_SOCKETS-1))
    RETURN_NAMES = tuple([f"value{i}" for i in range(1, NUM_FLOW_SOCKETS)])
    FUNCTION = "fn"

    CATEGORY = get_category("Loop/Flow")

    def fn(self, flow_control, **kwargs):
        graph = GraphBuilder()
        while_open = flow_control[0]
        sub = graph.node("MaraScottForLoopIntMathOperation_v1", operation="subtract", a=[while_open,1], b=1)
        cond = graph.node("MaraScottForLoopToBoolNode_v1", value=sub.out(0))
        input_values = {f"initial_value{i}": kwargs.get(f"initial_value{i}", None) for i in range(1, NUM_FLOW_SOCKETS)}
        while_close = graph.node("MaraScottForLoopWhileClose_v1",
                flow_control=flow_control,
                condition=cond.out(0),
                initial_value0=sub.out(0),
                **input_values)
        return {
            "result": tuple([while_close.out(i) for i in range(1, NUM_FLOW_SOCKETS)]),
            "expand": graph.finalize(),
        }

@VariantSupport()
class ForLoopWhileOpen_v1:
    
    def __init__(self):
        pass

    NAME = get_name("For Loop While Open", "l")

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "condition": ("BOOLEAN", {"default": True}),
            },
            "optional": {
            },
        }
        for i in range(NUM_FLOW_SOCKETS):
            inputs["optional"][f"initial_value{i}"] = ("*",)
        return inputs

    RETURN_TYPES = tuple(["FLOW_CONTROL"] + ["*"] * NUM_FLOW_SOCKETS)
    RETURN_NAMES = tuple(["FLOW_CONTROL"] + [f"value{i}" for i in range(NUM_FLOW_SOCKETS)])
    FUNCTION = "while_loop_open"

    CATEGORY = get_category("Loop/Flow")

    def while_loop_open(self, condition, **kwargs):
        values = []
        for i in range(NUM_FLOW_SOCKETS):
            values.append(kwargs.get(f"initial_value{i}", None))
        return tuple(["stub"] + values)

@VariantSupport()
class ForLoopWhileClose_v1:
    
    def __init__(self):
        pass

    NAME = get_name("For Loop While Close", "l")

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "flow_control": ("FLOW_CONTROL", {"rawLink": True}),
                "condition": ("BOOLEAN", {"forceInput": True}),
            },
            "optional": {
            },
            "hidden": {
                "dynprompt": "DYNPROMPT",
                "unique_id": "UNIQUE_ID",
            }
        }
        for i in range(NUM_FLOW_SOCKETS):
            inputs["optional"][f"initial_value{i}"] = ("*",)
        return inputs

    RETURN_TYPES = tuple(["*"] * NUM_FLOW_SOCKETS)
    RETURN_NAMES = tuple([f"value{i}" for i in range(NUM_FLOW_SOCKETS)])
    FUNCTION = "fn"

    CATEGORY = get_category("Loop/Flow")

    def explore_dependencies(self, node_id, dynprompt, upstream):
        node_info = dynprompt.get_node(node_id)
        if "inputs" not in node_info:
            return
        for k, v in node_info["inputs"].items():
            if is_link(v):
                parent_id = v[0]
                if parent_id not in upstream:
                    upstream[parent_id] = []
                    self.explore_dependencies(parent_id, dynprompt, upstream)
                upstream[parent_id].append(node_id)

    def collect_contained(self, node_id, upstream, contained):
        if node_id not in upstream:
            return
        for child_id in upstream[node_id]:
            if child_id not in contained:
                contained[child_id] = True
                self.collect_contained(child_id, upstream, contained)


    def fn(self, flow_control, condition, dynprompt=None, unique_id=None, **kwargs):
        assert dynprompt is not None
        if not condition:
            # We're done with the loop
            values = []
            for i in range(NUM_FLOW_SOCKETS):
                values.append(kwargs.get(f"initial_value{i}", None))
            return tuple(values)

        # We want to loop
        upstream = {}
        # Get the list of all nodes between the open and close nodes
        self.explore_dependencies(unique_id, dynprompt, upstream)

        contained = {}
        open_node = flow_control[0]
        self.collect_contained(open_node, upstream, contained)
        contained[unique_id] = True
        contained[open_node] = True

        # We'll use the default prefix, but to avoid having node names grow exponentially in size,
        # we'll use "Recurse" for the name of the recursively-generated copy of this node.
        graph = GraphBuilder()
        for node_id in contained:
            original_node = dynprompt.get_node(node_id)
            node = graph.node(original_node["class_type"], "Recurse" if node_id == unique_id else node_id)
            node.set_override_display_id(node_id)
        for node_id in contained:
            original_node = dynprompt.get_node(node_id)
            node = graph.lookup_node("Recurse" if node_id == unique_id else node_id)
            assert node is not None
            for k, v in original_node["inputs"].items():
                if is_link(v) and v[0] in contained:
                    parent = graph.lookup_node(v[0])
                    assert parent is not None
                    node.set_input(k, parent.out(v[1]))
                else:
                    node.set_input(k, v)
        new_open = graph.lookup_node(open_node)
        assert new_open is not None
        for i in range(NUM_FLOW_SOCKETS):
            key = f"initial_value{i}"
            new_open.set_input(key, kwargs.get(key, None))
        my_clone = graph.lookup_node("Recurse")
        assert my_clone is not None
        result = map(lambda x: my_clone.out(x), range(NUM_FLOW_SOCKETS))
        return {
            "result": tuple(result),
            "expand": graph.finalize(),
        }

class ForLoopIntMathOperation_v1:
    
    def __init__(self):
        pass

    NAME = get_name("For Loop IntMathOperation", "l")

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

    CATEGORY = get_category("Loop/Logic")

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

class ForLoopToBoolNode_v1:
    
    def __init__(self):
        pass

    NAME = get_name("For Loop ToBoolNode", "l")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("*",),
            },
            "optional": {
                "invert": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("BOOLEAN",)
    FUNCTION = "fn"

    CATEGORY = get_category("Loop/Logic")

    def fn(self, value, invert = False):
        if isinstance(value, torch.Tensor):
            if value.max().item() == 0 and value.min().item() == 0:
                result = False
            else:
                result = True
        else:
            try:
                result = bool(value)
            except:
                # Can't convert it? Well then it's something or other. I dunno, I'm not a Python programmer.
                result = True

        if invert:
            result = not result

        return (result,)