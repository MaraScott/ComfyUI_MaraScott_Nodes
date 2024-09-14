from ...utils.helper import AlwaysEqualProxy, ByPassTypeTuple
from ...utils.cache import cache, update_cache, remove_cache
from ...utils.math import Compare, mathIntOperation
try:
    from comfy_execution.graph_utils import GraphBuilder, is_link
except:
    GraphBuilder = None

MAX_FLOW_NUM = 10
any_type = AlwaysEqualProxy("*")

class whileLoopStart:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "condition": ("BOOLEAN", {"default": True}),
            },
            "optional": {
            },
        }
        for i in range(MAX_FLOW_NUM):
            inputs["optional"]["initial_value%d" % i] = ("*",)
        return inputs

    RETURN_TYPES = ByPassTypeTuple(tuple(["FLOW_CONTROL"] + ["*"] * MAX_FLOW_NUM))
    RETURN_NAMES = ByPassTypeTuple(tuple(["flow"] + ["value%d" % i for i in range(MAX_FLOW_NUM)]))
    FUNCTION = "fn"

    CATEGORY = "MaraScott/Loop"

    def fn(self, condition, **kwargs):
        values = []
        for i in range(MAX_FLOW_NUM):
            values.append(kwargs.get("initial_value%d" % i, None))
        return tuple(["stub"] + values)

class whileLoopEnd:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "flow": ("FLOW_CONTROL", {"rawLink": True}),
                "condition": ("BOOLEAN", {}),
            },
            "optional": {
            },
            "hidden": {
                "dynprompt": "DYNPROMPT",
                "unique_id": "UNIQUE_ID",
            }
        }
        for i in range(MAX_FLOW_NUM):
            inputs["optional"]["initial_value%d" % i] = (AlwaysEqualProxy('*'),)
        return inputs

    RETURN_TYPES = ByPassTypeTuple(tuple([AlwaysEqualProxy('*')] * MAX_FLOW_NUM))
    RETURN_NAMES = ByPassTypeTuple(tuple(["value%d" % i for i in range(MAX_FLOW_NUM)]))
    FUNCTION = "fn"

    CATEGORY = "MaraScott/Loop"

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


    def fn(self, flow, condition, dynprompt=None, unique_id=None, **kwargs):
        if not condition:
            # We're done with the loop
            values = []
            for i in range(MAX_FLOW_NUM):
                values.append(kwargs.get("initial_value%d" % i, None))
            return tuple(values)

        # We want to loop
        # this_node = dynprompt.get_node(unique_id)
        upstream = {}
        # Get the list of all nodes between the open and close nodes
        self.explore_dependencies(unique_id, dynprompt, upstream)

        contained = {}
        open_node = flow[0]
        self.collect_contained(open_node, upstream, contained)
        contained[unique_id] = True
        contained[open_node] = True

        graph = GraphBuilder()
        for node_id in contained:
            original_node = dynprompt.get_node(node_id)
            node = graph.node(original_node["class_type"], "Recurse" if node_id == unique_id else node_id)
            node.set_override_display_id(node_id)
        for node_id in contained:
            original_node = dynprompt.get_node(node_id)
            node = graph.lookup_node("Recurse" if node_id == unique_id else node_id)
            for k, v in original_node["inputs"].items():
                if is_link(v) and v[0] in contained:
                    parent = graph.lookup_node(v[0])
                    node.set_input(k, parent.out(v[1]))
                else:
                    node.set_input(k, v)

        new_open = graph.lookup_node(open_node)
        for i in range(MAX_FLOW_NUM):
            key = "initial_value%d" % i
            new_open.set_input(key, kwargs.get(key, None))
        my_clone = graph.lookup_node("Recurse")
        result = map(lambda x: my_clone.out(x), range(MAX_FLOW_NUM))
        return {
            "result": tuple(result),
            "expand": graph.finalize(),
        }


class forLoopStart:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "total": ("INT", {"default": 1, "min": 1, "max": 100000, "step": 1}),
            },
            "optional": {
                "initial_value%d" % i: (any_type,) for i in range(1, MAX_FLOW_NUM)
            },
            "hidden": {
                "initial_value0": (any_type,),
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID"
            }
        }

    RETURN_TYPES = ByPassTypeTuple(tuple(["FLOW_CONTROL", "INT"] + [any_type] * (MAX_FLOW_NUM - 1)))
    RETURN_NAMES = ByPassTypeTuple(tuple(["flow", "index"] + ["value%d" % i for i in range(1, MAX_FLOW_NUM)]))
    FUNCTION = "fn"

    CATEGORY = "MaraScott/Loop"

    def fn(self, total, prompt=None, extra_pnginfo=None, unique_id=None, **kwargs):
        graph = GraphBuilder()
        i = 0
        unique_id = unique_id.split('.')[len(unique_id.split('.'))-1] if "." in unique_id else unique_id
        update_cache('forloop'+str(unique_id), 'forloop', total)
        if "initial_value0" in kwargs:
            i = kwargs["initial_value0"]

        # initial_values = {("initial_value%d" % num): kwargs.get("initial_value%d" % num, None) for num in range(1, MAX_FLOW_NUM)}
        # while_open = graph.node("easy whileLoopStart", condition=total, initial_value0=i, **initial_values)
        outputs = [kwargs.get("initial_value%d" % num, None) for num in range(1, MAX_FLOW_NUM)]
        return {
            "result": tuple(["stub", i] + outputs),
            "expand": graph.finalize(),
        }

class forLoopEnd:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flow": ("FLOW_CONTROL", {"rawLink": True}),
            },
            "optional": {
                "initial_value%d" % i: (any_type, {"rawLink": True}) for i in range(1, MAX_FLOW_NUM)
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID"
            },
        }

    RETURN_TYPES = ByPassTypeTuple(tuple([any_type] * (MAX_FLOW_NUM - 1)))
    RETURN_NAMES = ByPassTypeTuple(tuple(["value%d" % i for i in range(1, MAX_FLOW_NUM)]))
    FUNCTION = "fn"

    CATEGORY = "MaraScott/Loop"

    def fn(self, flow, prompt=None, extra_pnginfo=None, unique_id=None, **kwargs):
        def get_last_segment(s):
            """Helper function to retrieve the last segment after the final dot."""
            if isinstance(s, str) and '.' in s:
                return s.split('.')[-1]
            return s  # Return as-is if it's not a string with dots
        # Process flow: Extract the last segment of flow[0] if it's a string
        if isinstance(flow, list) and len(flow) > 0:
            processed_flow = [get_last_segment(flow[0])] + flow[1:]
        else:
            processed_flow = flow  # Handle cases where flow is not a list or is empty
            
        # Process unique_id: Extract the last segment if it's a string
        processed_unique_id = get_last_segment(unique_id)
        
        # Process kwargs: Extract the last segment of the first element in each value list
        processed_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, list) and len(value) > 0:
                # Assume the first element is the string to process
                processed_first_element = get_last_segment(value[0])
                # Reconstruct the list with the processed first element
                processed_kwargs[key] = [processed_first_element] + value[1:]
            else:
                # If the value isn't a list or is empty, keep it unchanged
                processed_kwargs[key] = value
                
        # Reassign processed values
        flow = processed_flow
        unique_id = processed_unique_id
        kwargs = processed_kwargs

        graph = GraphBuilder()
        while_open = flow[0]
        total = None
        
        if "forloop"+str(while_open) in cache:
            total = cache['forloop'+str(while_open)][1]
        elif extra_pnginfo:
            all_nodes = extra_pnginfo['workflow']['nodes']
            start_node = next((x for x in all_nodes if x['id'] == int(while_open)), None)
            total = start_node['widgets_values'][0] if "widgets_values" in start_node else None
        if total is None:
            raise Exception("Unable to get parameters for the start of the loop")
        sub = mathIntOperation(operation="add", a=[while_open, 1], b=1)
        cond = Compare(a=sub.out(0), b=total, comparison='a < b')
        input_values = {("initial_value%d" % i): kwargs.get("initial_value%d" % i, None) for i in 
                        range(1, MAX_FLOW_NUM)}
        while_close = whileLoopEnd(flow=flow, condition=cond.out(0), initial_value0=sub.out(0), **input_values)
        return {
            "result": tuple([while_close.out(i) for i in range(1, MAX_FLOW_NUM)]),
            "expand": graph.finalize(),
        }
