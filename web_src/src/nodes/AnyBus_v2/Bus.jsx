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
    const slotInfo = {}; // slotIndex -> { type, label, hasConnection, customLabel }

    for (const nodeId of connectedNodeIds) {
        const connectedNode = node.graph?.getNodeById(nodeId);
        if (!connectedNode || !connectedNode.inputs) continue;

        for (let i = 1; i < connectedNode.inputs.length; i++) {
            const input = connectedNode.inputs[i];

            if (!slotInfo[i]) {
                slotInfo[i] = { type: "*", label: null, hasConnection: false, customLabel: null };
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
                            // Prioritize non-wildcard types
                            if (slotInfo[i].type === "*" || slotInfo[i].type === sourceOutput.type) {
                                slotInfo[i].type = sourceOutput.type;
                            }
                        }
                    }
                }
            }

            // Collect labels - prioritize custom labels over type names
            if (input.label) {
                // Check if it's a custom label (not a default "* XX" pattern)
                const isDefaultLabel = input.label.match(/^\* \d{2}$/);

                if (!isDefaultLabel) {
                    // Check if it's a custom label (different from the type name)
                    const isCustomLabel = input.label !== input.type;

                    if (isCustomLabel && !slotInfo[i].customLabel) {
                        // Prioritize custom labels
                        slotInfo[i].customLabel = input.label;
                        slotInfo[i].label = input.label;
                    } else if (!slotInfo[i].label) {
                        // Store type-based label as fallback
                        slotInfo[i].label = input.label;
                    }
                }
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

                    // Update label: prioritize custom label > type > default
                    let newLabel;
                    if (info.customLabel) {
                        newLabel = info.customLabel; // Use synced custom label
                    } else if (info.label) {
                        newLabel = info.label; // Use type-based label
                    } else if (info.type !== "*") {
                        newLabel = info.type; // Use type as label
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

// Helper: Forward data through BUS-connected nodes by creating slot connections
export function forwardDataThroughBus(node) {
    if (!node || !node.graph) return;

    const connectedNodeIds = getBusConnectedNodes(node);
    const nodes = Array.from(connectedNodeIds).map(id => node.graph.getNodeById(id)).filter(n => n);

    if (nodes.length < 2) return; // Need at least 2 nodes to forward

    // For each slot (excluding BUS slot 0), find if any node has an input connection
    // and forward it to outputs of other nodes in the profile
    const numSlots = Math.max(...nodes.map(n => n.inputs?.length || 0));

    for (let slotIdx = 1; slotIdx < numSlots; slotIdx++) {
        // Find nodes with input connections at this slot
        const sourceConnections = [];

        for (const busNode of nodes) {
            if (busNode.inputs && busNode.inputs[slotIdx] && busNode.inputs[slotIdx].link) {
                const link = busNode.graph.links[busNode.inputs[slotIdx].link];
                if (link && link.origin_id !== busNode.id) { // Not a self-connection
                    sourceConnections.push({
                        node: busNode,
                        link: link,
                        sourceNodeId: link.origin_id,
                        sourceSlot: link.origin_slot
                    });
                }
            }
        }

        // If we have input connections, forward them to outputs of other BUS nodes
        if (sourceConnections.length > 0) {
            // Use the first source connection for forwarding
            const source = sourceConnections[0];
            const sourceNode = node.graph.getNodeById(source.sourceNodeId);

            if (!sourceNode) continue;

            // Connect this source to all other BUS nodes' outputs at the same slot
            for (const targetBusNode of nodes) {
                // Skip if it's the node that already has the input
                if (targetBusNode.id === source.node.id) continue;

                // Check if output slot exists and is not already connected to this source
                if (targetBusNode.outputs && targetBusNode.outputs[slotIdx]) {
                    const output = targetBusNode.outputs[slotIdx];

                    // Check if already connected to the same source
                    let alreadyConnected = false;
                    if (output.links) {
                        for (const linkId of output.links) {
                            const existingLink = node.graph.links[linkId];
                            if (existingLink && existingLink.origin_id === source.sourceNodeId &&
                                existingLink.origin_slot === source.sourceSlot) {
                                alreadyConnected = true;
                                break;
                            }
                        }
                    }

                    if (!alreadyConnected) {
                        // Create a virtual connection from the source to this output
                        // We connect the sourceNode output to an input of targetBusNode, then
                        // targetBusNode output can be used downstream
                        // Actually, we just mark the output as having the same data by connecting
                        // the original source to any nodes connected to this output

                        // Forward the connection: if targetBusNode.outputs[slotIdx] has links,
                        // replace them to point to sourceNode instead
                        if (output.links && output.links.length > 0) {
                            for (const linkId of [...output.links]) {
                                const downstreamLink = node.graph.links[linkId];
                                if (downstreamLink) {
                                    // Reconnect downstream nodes directly to the source
                                    const downstreamNode = node.graph.getNodeById(downstreamLink.target_id);
                                    const downstreamSlot = downstreamLink.target_slot;

                                    if (downstreamNode) {
                                        // Remove old link
                                        targetBusNode.disconnectOutput(slotIdx, downstreamNode);

                                        // Create new link from source
                                        sourceNode.connect(source.sourceSlot, downstreamNode, downstreamSlot);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
