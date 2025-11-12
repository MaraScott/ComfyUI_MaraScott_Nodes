// Bus operations for AnyBus_v2
import { getProfileEntry, profileSlotOrders } from "./State.jsx";
import { applySlotOrderToNode, resetNodeDisconnectedSlots } from "./Node.jsx";

// Helper: Get all BUS-connected nodes recursively
export function getBusConnectedNodes(node, visited = new Set()) {
    if (!node || visited.has(node.id)) return visited;
    visited.add(node.id);

    // Check all inputs for BUS connections
    if (node.inputs) {
        for (let i = 0; i < node.inputs.length; i++) {
            const input = node.inputs[i];
            if (input.type === "ANYBUS_v2" && input.link) {
                const link = node.graph?.links[input.link];
                if (link) {
                    const sourceNode = node.graph?.getNodeById(link.origin_id);
                    if (sourceNode && !visited.has(sourceNode.id)) {
                        getBusConnectedNodes(sourceNode, visited);
                    }
                }
            }
        }
    }

    // Check all outputs for BUS connections
    if (node.outputs) {
        for (let i = 0; i < node.outputs.length; i++) {
            const output = node.outputs[i];
            if (output.type === "ANYBUS_v2" && output.links) {
                for (const linkId of output.links) {
                    const link = node.graph?.links[linkId];
                    if (link) {
                        const targetNode = node.graph?.getNodeById(link.target_id);
                        if (targetNode && !visited.has(targetNode.id)) {
                            getBusConnectedNodes(targetNode, visited);
                        }
                    }
                }
            }
        }
    }

    return visited;
}

// Helper: Synchronize labels and types across BUS-connected nodes
export function syncConnectedNodesLabelsAndTypes(node) {
    const connectedNodeIds = getBusConnectedNodes(node);

    // Collect all slot information from all connected nodes
    const slotInfo = {}; // slotIndex -> { type, label, hasConnection }

    for (const nodeId of connectedNodeIds) {
        const connectedNode = node.graph?.getNodeById(nodeId);
        if (!connectedNode || !connectedNode.inputs) continue;

        for (let i = 1; i < connectedNode.inputs.length; i++) {
            const input = connectedNode.inputs[i];

            if (!slotInfo[i]) {
                slotInfo[i] = { type: "*", label: null, hasConnection: false };
            }

            // Track if this slot has a connection
            if (input.link) {
                slotInfo[i].hasConnection = true;

                // Get the connected type
                const link = connectedNode.graph?.links[input.link];
                if (link) {
                    const sourceNode = connectedNode.graph?.getNodeById(link.origin_id);
                    if (sourceNode && sourceNode.outputs) {
                        const sourceOutput = sourceNode.outputs[link.origin_slot];
                        if (sourceOutput && sourceOutput.type && sourceOutput.type !== "*") {
                            slotInfo[i].type = sourceOutput.type;
                        }
                    }
                }
            }

            // Collect custom labels (not default ones)
            if (input.label && !input.label.startsWith("* ")) {
                slotInfo[i].label = input.label;
            }
        }
    }

    // Apply collected information to all connected nodes
    for (const nodeId of connectedNodeIds) {
        const connectedNode = node.graph?.getNodeById(nodeId);
        if (!connectedNode) continue;

        // Update inputs
        if (connectedNode.inputs) {
            for (let i = 1; i < connectedNode.inputs.length; i++) {
                const input = connectedNode.inputs[i];
                const info = slotInfo[i];

                if (info) {
                    // Update type
                    if (input.type !== info.type) {
                        input.type = info.type;
                    }

                    // Update label: show type if connected, otherwise custom label or default
                    let newLabel;
                    if (info.hasConnection && info.type !== "*") {
                        newLabel = info.type; // Show the connected type as label
                    } else if (info.label) {
                        newLabel = info.label; // Use custom label
                    } else {
                        newLabel = `* ${String(i).padStart(2, '0')}`; // Default label
                    }

                    if (input.label !== newLabel) {
                        input.label = newLabel;
                    }
                }
            }
        }

        // Update outputs to match inputs
        if (connectedNode.outputs) {
            for (let i = 1; i < connectedNode.outputs.length; i++) {
                const output = connectedNode.outputs[i];
                const input = connectedNode.inputs ? connectedNode.inputs[i] : null;

                if (input) {
                    if (output.type !== input.type) {
                        output.type = input.type;
                    }
                    if (output.label !== input.label) {
                        output.label = input.label;
                    }
                }
            }
        }

        connectedNode.setDirtyCanvas(true, true);
    }
}

// Helper: Update profile for all BUS-connected nodes
export function updateConnectedNodesProfile(node, newProfile) {
    const connectedNodeIds = getBusConnectedNodes(node);

    for (const nodeId of connectedNodeIds) {
        const connectedNode = node.graph?.getNodeById(nodeId);
        if (!connectedNode) continue;

        const profileWidget = connectedNode.widgets?.find(w => w.name === "profile");
        if (profileWidget && profileWidget.value !== newProfile) {
            const oldProfile = profileWidget.value;

            // Remove from old profile registry
            if (oldProfile) {
                getProfileEntry(oldProfile).delete(connectedNode);
            }

            // Update widget value
            profileWidget.value = newProfile;
            connectedNode._anybus_profile = newProfile;

            // Add to new profile registry
            getProfileEntry(newProfile).add(connectedNode);

            // Mark node as dirty to update UI
            connectedNode.setDirtyCanvas(true, true);
        }
    }

    // Sync labels after all profile updates
    syncConnectedNodesLabelsAndTypes(node);
}

// Helper: Update num_slots for all BUS-connected nodes
export function updateConnectedNodesSlots(node, newNumSlots, updateNodeSlotsFn) {
    const connectedNodeIds = getBusConnectedNodes(node);

    for (const nodeId of connectedNodeIds) {
        const connectedNode = node.graph?.getNodeById(nodeId);
        if (!connectedNode) continue;

        const numSlotsWidget = connectedNode.widgets?.find(w => w.name === "num_slots");
        if (numSlotsWidget && numSlotsWidget.value !== newNumSlots) {
            // Update widget value
            numSlotsWidget.value = newNumSlots;

            // Use the passed function to avoid circular dependency
            if (updateNodeSlotsFn) {
                updateNodeSlotsFn(connectedNode, newNumSlots);
            }
        }
    }
}

// Helper: Reset all disconnected slots for a profile
export function resetProfileDisconnectedSlots(profileName, graphApp) {
    const profileNodes = getProfileEntry(profileName);
    if (!profileNodes) return;

    let hasAnyChanges = false;

    for (const node of profileNodes) {
        const changed = resetNodeDisconnectedSlots(node);
        if (changed) {
            hasAnyChanges = true;
        }
    }

    // Sync changes across all nodes if there were any changes
    if (hasAnyChanges && profileNodes.size > 0) {
        const firstNode = Array.from(profileNodes)[0];
        syncConnectedNodesLabelsAndTypes(firstNode);
    }

    return hasAnyChanges;
}
