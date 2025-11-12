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

    // Initialize inputs array if it doesn't exist
    if (!node.inputs) {
        node.inputs = [];
    }

    // Initialize outputs array if it doesn't exist
    if (!node.outputs) {
        node.outputs = [];
    }

    // Expected: 1 BUS input + numSlots any-type inputs
    const expectedInputs = 1 + parseInt(numSlots);
    const expectedOutputs = 1 + parseInt(numSlots);

    // Ensure BUS input exists
    if (node.inputs.length === 0) {
        node.addInput("bus", "ANYBUS_v2");
    }

    // Ensure BUS output exists
    if (node.outputs.length === 0) {
        node.addOutput("bus", "ANYBUS_v2");
    }

    // Add missing inputs (start from current length)
    while (node.inputs.length < expectedInputs) {
        const slotNum = node.inputs.length; // Current length gives us the next slot number
        const label = `* ${String(slotNum).padStart(2, '0')}`;
        node.addInput(label, "*", { label });
    }

    // Remove extra inputs (but keep BUS input at slot 0)
    while (node.inputs.length > expectedInputs) {
        node.removeInput(node.inputs.length - 1);
    }

    // Add missing outputs
    while (node.outputs.length < expectedOutputs) {
        const slotNum = node.outputs.length;
        const label = `* ${String(slotNum).padStart(2, '0')}`;
        node.addOutput(label, "*");
    }

    // Remove extra outputs (but keep BUS output at slot 0)
    while (node.outputs.length > expectedOutputs) {
        node.removeOutput(node.outputs.length - 1);
    }

    node.setSize(node.computeSize());
    node.setDirtyCanvas(true, true);
}

// Helper: Reset disconnected slots to default labels
export function resetNodeDisconnectedSlots(node) {
    if (!node || !node.inputs || !node.outputs) return false;

    let hasChanges = false;

    // Reset labels for disconnected slots
    for (let i = 1; i < node.inputs.length; i++) {
        const input = node.inputs[i];
        // Only reset if not connected
        if (!input.link) {
            const defaultLabel = `* ${String(i).padStart(2, '0')}`;
            if (input.label !== defaultLabel || input.type !== "*") {
                input.label = defaultLabel;
                input.type = "*";
                hasChanges = true;
            }

            // Reset corresponding output
            if (node.outputs[i]) {
                if (node.outputs[i].label !== defaultLabel || node.outputs[i].type !== "*") {
                    node.outputs[i].label = defaultLabel;
                    node.outputs[i].type = "*";
                    hasChanges = true;
                }
            }
        }
    }

    if (hasChanges) {
        node.setDirtyCanvas(true, true);
    }

    return hasChanges;
}
