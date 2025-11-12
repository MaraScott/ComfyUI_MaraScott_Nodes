from ...utils.constants import get_category
from ...utils.helper import AlwaysEqualProxy

any_type = AlwaysEqualProxy("*")


class Mara_AnyBus_v2:
    """
    AnyBus v2 - Dynamic bus connection system

    This node acts as a data hub that can receive multiple inputs and forward them
    to its outputs, while also passing through a BUS connection that carries all
    slot values as a tuple.

    Profile management is handled automatically by the frontend based on BUS connections.
    """

    NAME = "AnyBus v2"
    SHORTCUT = "ab"

    @classmethod
    def INPUT_TYPES(cls):
        # Define all 24 slots in optional
        optional_inputs = {
            "bus": ("ANYBUS_v2",),
            "getset_source": ("STRING", {
                "default": "",
                "multiline": False,
            }),
        }

        # Add 24 dynamic slots
        for i in range(1, 25):
            optional_inputs[f"* {str(i).zfill(2)}"] = (any_type,)

        return {
            "required": {
                "num_slots": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 24,
                    "step": 1,
                    "display": "number",
                }),
                "mode": (["bus", "getset"], {
                    "default": "bus",
                }),
            },
            "optional": optional_inputs
        }

    RETURN_TYPES = ("ANYBUS_v2",) + (any_type,) * 24
    RETURN_NAMES = ("bus",) + tuple(f"* {str(i).zfill(2)}" for i in range(1, 25))
    FUNCTION = "execute"
    CATEGORY = get_category("Bus")

    def execute(self, num_slots, mode="bus", bus=None, getset_source="", **kwargs):
        """
        Execute the bus node:
        1. Collect all input values from slots
        2. Create a tuple of all slot values
        3. Return the tuple as bus and individual values as outputs
        Mode:
        - bus: Use direct BUS connections
        - getset: Use virtual Get/Set connections (getset_source specifies the source node)

        Note: Profile management is now handled entirely in the frontend
        """

        # Collect input values from all slots
        slot_values = []
        for i in range(1, num_slots + 1):
            input_key = f"* {str(i).zfill(2)}"
            value = kwargs.get(input_key, None)
            slot_values.append(value)

        # If we received a bus input (only in bus mode), merge its values with our inputs
        # Priority: local inputs > bus values
        if mode == "bus" and bus is not None and isinstance(bus, (tuple, list)):
            for i in range(len(bus)):
                if i < len(slot_values) and slot_values[i] is None:
                    slot_values[i] = bus[i]

        # Create the bus output tuple
        bus_output = tuple(slot_values)

        # Prepare return values: bus_output + individual slot values + None for unused slots
        return_values = [bus_output]

        # Add slot values (24 slots)
        for i in range(24):
            if i < len(slot_values):
                return_values.append(slot_values[i])
            else:
                return_values.append(None)

        return tuple(return_values)
