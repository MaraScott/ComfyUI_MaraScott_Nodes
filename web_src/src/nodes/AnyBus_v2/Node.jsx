// Node operations for AnyBus_v2

// Helper: Apply slot order to a node
export function applySlotOrderToNode(node, slotOrder) {
    if (!node || !node.inputs || !node.outputs) return;

    // Skip if no custom order
    if (Object.keys(slotOrder).length === 0) return;

    // Create new arrays for reordered inputs/outputs
    const newInputs = [node.inputs[0]]; // Keep BUS input at position 0
    const newOutputs = [node.outputs[0]]; // Keep BUS output at position 0

    // Build reverse mapping (newIndex -> originalIndex)
    const reverseOrder = {};
    for (const [orig, newIdx] of Object.entries(slotOrder)) {
        reverseOrder[newIdx] = parseInt(orig);
    }

    // Reorder inputs/outputs based on slot order
    for (let i = 1; i < node.inputs.length; i++) {
        const originalIndex = reverseOrder[i] || i;
        if (node.inputs[originalIndex]) {
            newInputs[i] = { ...node.inputs[originalIndex] };
        }
        if (node.outputs[originalIndex]) {
            newOutputs[i] = { ...node.outputs[originalIndex] };
        }
    }

    // Update node
    node.inputs = newInputs;
    node.outputs = newOutputs;
    node.setDirtyCanvas(true, true);
}

// Helper: Update node's slot count
export function updateNodeSlots(node, numSlots) {
    if (!node || numSlots === undefined) return;

    const expectedSlots = parseInt(numSlots);

    if (!node.inputs || !node.outputs) return;

    // Expected structure:
    // Inputs: bus (optional) + getset_source (optional) + slot inputs (1 to numSlots)
    // Outputs: bus + slot outputs (1 to numSlots)

    // Collect information about existing slot inputs
    const slotInputs = [];
    for (let i = 0; i < node.inputs.length; i++) {
        const input = node.inputs[i];
        if (input.widget) continue; // Skip widgets
        if (input.name === "bus" || input.name === "getset_source") continue;

        const match = input.name.match(/\* (\d+)/);
        if (match) {
            const slotNum = parseInt(match[1]);
            slotInputs.push({ index: i, slotNum: slotNum, input: input });
        }
    }

    // Sort by slot number
    slotInputs.sort((a, b) => a.slotNum - b.slotNum);

    // Calculate how many slot inputs we currently have
    const currentSlotInputs = slotInputs.length;

    // Remove extra slot inputs (from the highest slot number down)
    if (currentSlotInputs > expectedSlots) {
        for (let i = slotInputs.length - 1; i >= expectedSlots; i--) {
            node.removeInput(slotInputs[i].index);
        }
    }

    // Add missing slot inputs
    if (currentSlotInputs < expectedSlots) {
        for (let slotNum = currentSlotInputs + 1; slotNum <= expectedSlots; slotNum++) {
            const slotName = `* ${String(slotNum).padStart(2, '0')}`;
            const slotLabel = `* ${String(slotNum).padStart(2, '0')}`;
            // addInput(name, type, extra_info)
            node.addInput(slotName, "*", { label: slotLabel });
        }
    }

    // Handle outputs: BUS (slot 0) + slot outputs (1 to numSlots)
    const expectedOutputs = 1 + expectedSlots;

    // Remove extra outputs
    while (node.outputs.length > expectedOutputs) {
        node.removeOutput(node.outputs.length - 1);
    }

    // Add missing outputs
    while (node.outputs.length < expectedOutputs) {
        const slotNum = node.outputs.length; // 0 = BUS, 1+ = slots
        if (slotNum === 0) {
            node.addOutput("bus", "ANYBUS_v2");
        } else {
            const slotLabel = `* ${String(slotNum).padStart(2, '0')}`;
            node.addOutput(slotLabel, "*");
        }
    }

    node.setSize(node.computeSize());
    node.setDirtyCanvas(true, true);
}// Reset labels for disconnected slot inputs/outputs to default
export function resetNodeDisconnectedSlots(node) {
    if (!node) return;

    // Reset slot input labels
    if (node.inputs) {
        for (let i = 0; i < node.inputs.length; i++) {
            const input = node.inputs[i];
            if (input.widget) continue;
            if (input.name === "bus" || input.name === "getset_source") continue;

            const match = input.name.match(/\* (\d+)/);
            if (match) {
                const slotNum = parseInt(match[1]);
                // If no connection, reset label to default
                if (!input.link) {
                    input.label = `* ${String(slotNum).padStart(2, '0')}`;
                }
            }
        }
    }

    // Reset slot output labels
    if (node.outputs) {
        for (let i = 0; i < node.outputs.length; i++) {
            const output = node.outputs[i];
            if (output.name === "bus") continue;

            const match = output.name.match(/\* (\d+)/);
            if (match) {
                const slotNum = parseInt(match[1]);
                // If no connections, reset label to default
                if (!output.links || output.links.length === 0) {
                    output.label = `* ${String(slotNum).padStart(2, '0')}`;
                }
            }
        }
    }
}
