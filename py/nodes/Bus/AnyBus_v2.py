from ...utils.constants import get_category
from ...utils.helper import AlwaysEqualProxy

any_type = AlwaysEqualProxy("*")


class AnyBus_v2:
    """
    AnyBus v2 - Dynamic bus connection system

    This node acts as a data hub that can receive multiple inputs and forward them
    to its outputs, while also passing through a BUS connection that carries all
    slot values as a tuple for profile-based synchronization.
    """

    NAME = "🐰 AnyBus v2"
    SHORTCUT = "ab"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "num_slots": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 20,
                    "step": 1,
                    "display": "number",
                }),
                "profile": ("STRING", {
                    "default": "default",
                    "multiline": False,
                }),
                "mode": (["bus", "getset"], {
                    "default": "bus",
                }),
            },
            "optional": {
                "bus": ("ANYBUS_v2",),
                "getset_source": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
                # Dynamic slots will be handled by frontend
                "* 01": (any_type,),
                "* 02": (any_type,),
                "* 03": (any_type,),
            }
        }

    RETURN_TYPES = ("ANYBUS_v2",) + (any_type,) * 20
    RETURN_NAMES = ("bus",) + tuple(f"* {str(i).zfill(2)}" for i in range(1, 21))
    FUNCTION = "execute"
    CATEGORY = get_category("Bus")

    def execute(self, num_slots, profile, mode="bus", bus=None, getset_source="", **kwargs):
        """
        Execute the bus node:
        1. Collect all input values from slots
        2. Create a tuple of all slot values
        3. Return the tuple as bus and individual values as outputs
        Mode:
        - bus: Use direct BUS connections
        - getset: Use virtual Get/Set connections (getset_source specifies the source node)
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

        # Add slot values
        for i in range(20):
            if i < len(slot_values):
                return_values.append(slot_values[i])
            else:
                return_values.append(None)

        return tuple(return_values)
