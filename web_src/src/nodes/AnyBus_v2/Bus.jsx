// Bus operations for AnyBus_v2
import {
    getProfileEntry,
    profileSlotOrders,
    getProfileMasterState,
    syncNodeToMasterState,
    propagateMasterStateToNodes
} from "./State.jsx";
import { applySlotOrderToNode, resetNodeDisconnectedSlots, updateNodeSlots } from "./Node.jsx";

// Track hidden getset connections
// Map: nodeId -> { sourceNodeId, hiddenLink }
const hiddenGetSetConnections = new Map();

// Helper: Get all BUS-connected nodes recursively (including hidden getset connections)
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

    // Check for hidden getset connections (nodes connected to this node)
    for (const [getsetNodeId, connection] of hiddenGetSetConnections.entries()) {
        if (connection.sourceNodeId === node.id && !visited.has(getsetNodeId)) {
            const getsetNode = node.graph?.getNodeById(getsetNodeId);
            if (getsetNode) {
                getBusConnectedNodes(getsetNode, visited);
            }
        }
    }

    // Check if this node itself has a hidden connection to another node
    const myHiddenConnection = hiddenGetSetConnections.get(node.id);
    if (myHiddenConnection && !visited.has(myHiddenConnection.sourceNodeId)) {
        const sourceNode = node.graph?.getNodeById(myHiddenConnection.sourceNodeId);
        if (sourceNode) {
            getBusConnectedNodes(sourceNode, visited);
        }
    }

    return visited;
}

/**
 * Create a hidden BUS connection for getset mode
 * This allows getset nodes to sync with their source without visible links
 */
export function createHiddenGetSetConnection(getsetNode, sourceNode) {
    if (!getsetNode || !sourceNode || !getsetNode.graph) return false;

    // Check if already connected
    const existing = hiddenGetSetConnections.get(getsetNode.id);
    if (existing && existing.sourceNodeId === sourceNode.id) {
        return true; // Already connected
    }

    // Remove any existing hidden connection
    removeHiddenGetSetConnection(getsetNode);

    // Find BUS output slot (slot 0) on source node
    const sourceOutput = sourceNode.outputs?.[0];
    const getsetInput = getsetNode.inputs?.[0];

    if (!sourceOutput || !getsetInput || sourceOutput.type !== "ANYBUS_v2") {
        console.warn('[AnyBus] Cannot create hidden connection: invalid BUS slots');
        return false;
    }

    // Create a real internal link in the graph
    // This makes ComfyUI aware of the connection for execution purposes
    // but we'll hide it from the UI
    const linkId = getsetNode.graph.last_link_id + 1;
    getsetNode.graph.last_link_id = linkId;

    const link = {
        id: linkId,
        origin_id: sourceNode.id,
        origin_slot: 0,
        target_id: getsetNode.id,
        target_slot: 0,
        type: "ANYBUS_v2"
    };

    // Add to graph links
    getsetNode.graph.links[linkId] = link;

    // Connect the nodes
    getsetInput.link = linkId;
    if (!sourceOutput.links) {
        sourceOutput.links = [];
    }
    sourceOutput.links.push(linkId);

    // Store the hidden connection info
    hiddenGetSetConnections.set(getsetNode.id, {
        sourceNodeId: sourceNode.id,
        linkId: linkId
    });

    // Mark the link as hidden so we can style it differently or hide it
    getsetNode._anybus_hidden_connection = true;
    getsetNode._anybus_hidden_link_id = linkId;

    console.log(`[AnyBus] Created hidden connection: ${sourceNode.id} -> ${getsetNode.id} (link ${linkId})`);

    return true;
}

/**
 * Remove hidden BUS connection for getset node
 */
export function removeHiddenGetSetConnection(getsetNode) {
    if (!getsetNode) return;

    const existing = hiddenGetSetConnections.get(getsetNode.id);
    if (existing) {
        const { sourceNodeId, linkId } = existing;

        // Remove the real link from the graph
        if (getsetNode.graph && linkId) {
            const link = getsetNode.graph.links[linkId];
            if (link) {
                // Remove from source node's output links
                const sourceNode = getsetNode.graph.getNodeById(sourceNodeId);
                if (sourceNode && sourceNode.outputs && sourceNode.outputs[0]) {
                    const output = sourceNode.outputs[0];
                    if (output.links) {
                        const idx = output.links.indexOf(linkId);
                        if (idx !== -1) {
                            output.links.splice(idx, 1);
                        }
                    }
                }

                // Remove from getset node's input
                if (getsetNode.inputs && getsetNode.inputs[0]) {
                    getsetNode.inputs[0].link = null;
                }

                // Remove from graph
                delete getsetNode.graph.links[linkId];
            }
        }

        hiddenGetSetConnections.delete(getsetNode.id);
        delete getsetNode._anybus_hidden_connection;
        delete getsetNode._anybus_hidden_link_id;
        console.log(`[AnyBus] Removed hidden connection for node ${getsetNode.id}`);
    }
}

/**
 * Get source node for a getset node (via hidden connection)
 */
export function getHiddenGetSetSource(getsetNode) {
    if (!getsetNode) return null;

    const connection = hiddenGetSetConnections.get(getsetNode.id);
    if (!connection) return null;

    return getsetNode.graph?.getNodeById(connection.sourceNodeId);
}

/**
 * Update hidden connection when getset_source widget changes
 */
export function updateGetSetConnection(getsetNode) {
    if (!getsetNode || !getsetNode.graph) return;

    const getsetWidget = getsetNode.widgets?.find(w => w.name === "getset_source");
    const modeWidget = getsetNode.widgets?.find(w => w.name === "mode");

    // Only process if in getset mode
    if (modeWidget?.value !== "getset") {
        removeHiddenGetSetConnection(getsetNode);
        return;
    }

    const sourceIdentifier = getsetWidget?.value;
    if (!sourceIdentifier || sourceIdentifier === "") {
        removeHiddenGetSetConnection(getsetNode);
        return;
    }

    // Find source node by identifier (title or "Node X")
    let sourceNode = null;
    for (const node of getsetNode.graph._nodes) {
        if (node.type === "MaraScottAnyBus_v2" && node.id !== getsetNode.id) {
            const nodeIdentifier = node.title || `Node ${node.id}`;
            if (nodeIdentifier === sourceIdentifier) {
                sourceNode = node;
                break;
            }
        }
    }

    if (sourceNode) {
        // Get source node's profile ID
        const sourceProfileId = sourceNode._anybus_profileId;
        if (!sourceProfileId) {
            console.warn('[AnyBus] Source node has no profile ID');
            return;
        }

        // Update getset node's profile to match source
        const oldProfileId = getsetNode._anybus_profileId;
        if (oldProfileId !== sourceProfileId) {
            // Remove from old profile
            if (oldProfileId) {
                getProfileEntry(oldProfileId).delete(getsetNode);
            }

            // Assign source's profile ID
            getsetNode._anybus_profileId = sourceProfileId;

            // Add to source's profile
            getProfileEntry(sourceProfileId).add(getsetNode);
        }

        // Sync number of slots with source
        const sourceNumSlots = sourceNode.widgets?.find(w => w.name === "num_slots")?.value;
        const getsetNumSlots = getsetNode.widgets?.find(w => w.name === "num_slots");

        if (sourceNumSlots && getsetNumSlots && getsetNumSlots.value !== sourceNumSlots) {
            getsetNumSlots.value = sourceNumSlots;
            updateNodeSlots(getsetNode, sourceNumSlots);
        }

        // Create hidden connection
        createHiddenGetSetConnection(getsetNode, sourceNode);

        // Sync this node with the source's profile
        syncConnectedNodesLabelsAndTypes(sourceNode);
    } else {
        removeHiddenGetSetConnection(getsetNode);
    }
}

// Helper: Synchronize labels and types across BUS-connected nodes
// NOW USES CENTRALIZED STATE: Updates master state then propagates to all nodes
export function syncConnectedNodesLabelsAndTypes(node) {
    if (!node || !node._anybus_profileId) return;

    const profileId = node._anybus_profileId;
    const connectedNodeIds = getBusConnectedNodes(node);
    const masterState = getProfileMasterState(profileId);

    // STEP 1: Collect information from all connected nodes and update master state
    let masterStateChanged = false;

    for (const nodeId of connectedNodeIds) {
        const connectedNode = node.graph?.getNodeById(nodeId);
        if (!connectedNode || !connectedNode.inputs) continue;

        for (let i = 1; i < connectedNode.inputs.length; i++) {
            const input = connectedNode.inputs[i];

            // Extract actual slot number from input name (e.g., "* 04" -> 4)
            const match = input.name?.match(/\* (\d+)/);
            if (!match) continue;

            const slotNum = parseInt(match[1]);
            const currentSlot = masterState.slots[slotNum] || {};

            // Track if this slot has a connection
            const hasConnection = !!input.link;

            // Get connected type if available
            let connectedType = "*";
            if (input.link) {
                const link = connectedNode.graph?.links[input.link];
                if (link) {
                    const sourceNode = connectedNode.graph?.getNodeById(link.origin_id);
                    if (sourceNode && sourceNode.outputs) {
                        const sourceOutput = sourceNode.outputs[link.origin_slot];
                        if (sourceOutput && sourceOutput.type && sourceOutput.type !== "*") {
                            connectedType = sourceOutput.type;
                        }
                    }
                }
            }

            // Determine if label is custom (not a default "* XX" pattern and not just the type name)
            const isDefaultLabel = input.label?.match(/^\* \d{2}$/);
            const isTypeLabel = input.label === input.type;
            const isCustomLabel = input.label && !isDefaultLabel && !isTypeLabel;

            // Build new slot state with priority:
            // 1. Keep existing custom label (highest priority)
            // 2. If this input has custom label and master doesn't, use it
            // 3. Update type if we have connected non-wildcard type
            // 4. Update generic label as fallback

            const newSlot = {
                type: currentSlot.type || "*",
                label: currentSlot.label || null,
                customLabel: currentSlot.customLabel || null,
                hasConnection: currentSlot.hasConnection || hasConnection
            };

            // Update custom label - only if this node has one and master doesn't
            if (isCustomLabel && !newSlot.customLabel) {
                newSlot.customLabel = input.label;
            }

            // Update type - prioritize non-wildcard types
            if (connectedType !== "*" && (newSlot.type === "*" || newSlot.type === connectedType)) {
                newSlot.type = connectedType;
            }

            // Update label based on priority:
            // 1. Custom label (highest priority)
            // 2. Type label (if type is known and not wildcard)
            // 3. Default pattern "* XX"
            if (newSlot.customLabel) {
                newSlot.label = newSlot.customLabel;
            } else if (newSlot.type !== "*") {
                newSlot.label = newSlot.type;
            } else {
                newSlot.label = `* ${String(slotNum).padStart(2, '0')}`;
            }

            // Update hasConnection flag
            if (hasConnection) {
                newSlot.hasConnection = true;
            }

            // Check if state actually changed
            if (JSON.stringify(currentSlot) !== JSON.stringify(newSlot)) {
                masterState.slots[slotNum] = newSlot;
                masterStateChanged = true;
            }
        }
    }

    // STEP 2: If master state changed, propagate to all nodes in profile
    if (masterStateChanged) {
        masterState.lastUpdate = Date.now();
        propagateMasterStateToNodes(profileId);
    }
}

// Helper: Update profile for all BUS-connected nodes
export function updateConnectedNodesProfile(node, newProfileId) {
    const connectedNodeIds = getBusConnectedNodes(node);

    for (const nodeId of connectedNodeIds) {
        const connectedNode = node.graph?.getNodeById(nodeId);
        if (!connectedNode) continue;

        if (connectedNode._anybus_profileId !== newProfileId) {
            const oldProfileId = connectedNode._anybus_profileId;

            // Remove from old profile registry
            if (oldProfileId) {
                getProfileEntry(oldProfileId).delete(connectedNode);
            }

            // Update profile ID
            connectedNode._anybus_profileId = newProfileId;

            // Add to new profile registry
            getProfileEntry(newProfileId).add(connectedNode);

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
