// State management for AnyBus_v2
// CENTRALIZED STATE - Sidebar is the single source of truth
// NOW USES AUTO-GENERATED UNIQUE PROFILE IDs

// Global profile registry to track all AnyBus nodes by unique profile ID
export const profileRegistry = new Map(); // profileId -> Set of nodes

// Global slot order registry by profile ID
export const profileSlotOrders = new Map(); // profileId -> { originalIndex -> newIndex }

// CENTRALIZED PROFILE STATE
// Each profile has a master configuration that gets propagated to all nodes
// Structure: { profileId -> { id, label, slots, numSlots, slotOrder, lastUpdate } }
export const profileMasterState = new Map();

// Profile metadata: Maps profile ID to user-friendly labels
// Structure: { profileId -> { label, createdAt } }
export const profileMetadata = new Map();

// State change listeners for sidebar updates
const stateChangeListeners = new Set();

// Counter for generating unique profile IDs
let profileIdCounter = 1;

/**
 * Generate a new unique profile ID
 */
export function generateProfileId() {
    return `profile_${Date.now()}_${profileIdCounter++}`;
}

/**
 * Get or create profile metadata
 */
export function getProfileMetadata(profileId) {
    if (!profileMetadata.has(profileId)) {
        profileMetadata.set(profileId, {
            label: `Flow ${profileMetadata.size + 1}`,
            createdAt: Date.now()
        });
    }
    return profileMetadata.get(profileId);
}

/**
 * Update profile label
 */
export function updateProfileLabel(profileId, newLabel) {
    const metadata = getProfileMetadata(profileId);
    metadata.label = newLabel;

    // Update all nodes in this profile to refresh their titles
    const nodes = profileRegistry.get(profileId);
    if (nodes) {
        for (const node of nodes) {
            node.title = `AnyBus : ${newLabel}`;
            node.setDirtyCanvas(true, true);
        }
    }

    notifyStateChange(profileId, getProfileMasterState(profileId));
}

// Helper: Get or create profile entry
export function getProfileEntry(profileId) {
    if (!profileRegistry.has(profileId)) {
        profileRegistry.set(profileId, new Set());
    }
    return profileRegistry.get(profileId);
}

// Helper: Get or create slot order for profile
export function getProfileSlotOrder(profileId) {
    if (!profileSlotOrders.has(profileId)) {
        profileSlotOrders.set(profileId, {});
    }
    return profileSlotOrders.get(profileId);
}

// CENTRALIZED STATE MANAGEMENT

/**
 * Get or create master state for a profile ID
 */
export function getProfileMasterState(profileId) {
    if (!profileMasterState.has(profileId)) {
        const metadata = getProfileMetadata(profileId);
        profileMasterState.set(profileId, {
            id: profileId,
            label: metadata.label,
            slots: {}, // index -> { label, type, customLabel, hasConnection }
            numSlots: 3,
            slotOrder: {},
            lastUpdate: Date.now()
        });
    }
    return profileMasterState.get(profileId);
}

/**
 * Get all profile IDs
 */
export function getAllProfileIds() {
    return Array.from(profileRegistry.keys());
}

/**
 * Delete a profile and all its data
 */
export function deleteProfile(profileId) {
    profileRegistry.delete(profileId);
    profileMasterState.delete(profileId);
    profileMetadata.delete(profileId);
    profileSlotOrders.delete(profileId);
    notifyStateChange(profileId, null);
}

/**
 * Update master state for a profile ID and notify listeners
 */
export function updateProfileMasterState(profileId, updates) {
    const state = getProfileMasterState(profileId);

    if (updates.slots !== undefined) {
        state.slots = { ...state.slots, ...updates.slots };
    }
    if (updates.numSlots !== undefined) {
        state.numSlots = updates.numSlots;
    }
    if (updates.slotOrder !== undefined) {
        state.slotOrder = updates.slotOrder;
    }
    if (updates.label !== undefined) {
        state.label = updates.label;
        updateProfileLabel(profileId, updates.label);
    }

    state.lastUpdate = Date.now();

    // Notify all listeners
    notifyStateChange(profileId, state);

    return state;
}

/**
 * Sync slot information from a node to master state
 */
export function syncNodeToMasterState(node, slotIndex = null) {
    if (!node || !node._anybus_profileId) return;

    const profileId = node._anybus_profileId;
    const masterState = getProfileMasterState(profileId);
    let hasChanges = false;

    if (slotIndex !== null) {
        // Update specific slot
        const input = node.inputs?.[slotIndex];
        if (input) {
            const currentSlot = masterState.slots[slotIndex] || {};
            const isDefaultLabel = input.label?.match(/^\* \d{2}$/);
            const isCustomLabel = input.label && input.label !== input.type && !isDefaultLabel;

            const newSlot = {
                label: input.label || `* ${String(slotIndex).padStart(2, '0')}`,
                type: input.type || "*",
                customLabel: isCustomLabel ? input.label : (currentSlot.customLabel || null),
                hasConnection: !!input.link
            };

            if (JSON.stringify(currentSlot) !== JSON.stringify(newSlot)) {
                masterState.slots[slotIndex] = newSlot;
                hasChanges = true;
            }
        }
    } else {
        // Update all slots
        if (node.inputs) {
            for (let i = 1; i < node.inputs.length; i++) {
                const input = node.inputs[i];
                const currentSlot = masterState.slots[i] || {};
                const isDefaultLabel = input.label?.match(/^\* \d{2}$/);
                const isCustomLabel = input.label && input.label !== input.type && !isDefaultLabel;

                const newSlot = {
                    label: input.label || `* ${String(i).padStart(2, '0')}`,
                    type: input.type || "*",
                    customLabel: isCustomLabel ? input.label : (currentSlot.customLabel || null),
                    hasConnection: !!input.link
                };

                if (JSON.stringify(currentSlot) !== JSON.stringify(newSlot)) {
                    masterState.slots[i] = newSlot;
                    hasChanges = true;
                }
            }
        }
    }

    if (hasChanges) {
        masterState.lastUpdate = Date.now();
        notifyStateChange(profileId, masterState);
    }

    return hasChanges;
}

/**
 * Propagate master state to all nodes in a profile ID
 */
export function propagateMasterStateToNodes(profileId) {
    const masterState = getProfileMasterState(profileId);
    const nodes = profileRegistry.get(profileId);

    if (!nodes || nodes.size === 0) return;

    const metadata = getProfileMetadata(profileId);

    for (const node of nodes) {
        if (!node || !node.graph) continue;

        // Update node title with profile label
        node.title = `AnyBus : ${metadata.label}`;

        // Update slots based on master state
        if (node.inputs) {
            for (let i = 1; i < node.inputs.length; i++) {
                const input = node.inputs[i];

                // Extract actual slot number from input name (e.g., "* 04" -> 4)
                const match = input.name?.match(/\* (\d+)/);
                if (!match) continue;

                const slotNum = parseInt(match[1]);
                const slotState = masterState.slots[slotNum];
                if (!slotState) continue;

                // Update type
                if (input.type !== slotState.type) {
                    input.type = slotState.type;
                }

                // Update label - prioritize custom labels
                let newLabel;
                if (slotState.customLabel) {
                    newLabel = slotState.customLabel;
                } else if (slotState.type !== "*") {
                    newLabel = slotState.type;
                } else {
                    newLabel = `* ${String(slotNum).padStart(2, '0')}`;
                }

                if (input.label !== newLabel) {
                    input.label = newLabel;
                }
            }
        }

        // Update outputs to match inputs
        if (node.outputs) {
            for (let i = 1; i < node.outputs.length; i++) {
                const output = node.outputs[i];

                // Extract actual slot number from output name
                const match = output.name?.match(/\* (\d+)/);
                if (!match) continue;

                const slotNum = parseInt(match[1]);

                // Find corresponding input with same slot number
                let correspondingInput = null;
                if (node.inputs) {
                    for (const input of node.inputs) {
                        const inputMatch = input.name?.match(/\* (\d+)/);
                        if (inputMatch && parseInt(inputMatch[1]) === slotNum) {
                            correspondingInput = input;
                            break;
                        }
                    }
                }

                if (correspondingInput) {
                    if (output.type !== correspondingInput.type) {
                        output.type = correspondingInput.type;
                    }
                    if (output.label !== correspondingInput.label) {
                        output.label = correspondingInput.label;
                    }
                }
            }
        }

        node.setDirtyCanvas(true, true);
    }
}/**
 * Register a listener for state changes
 */
export function addStateChangeListener(listener) {
    stateChangeListeners.add(listener);
    return () => stateChangeListeners.delete(listener);
}

/**
 * Notify all listeners of state changes
 */
function notifyStateChange(profile, state) {
    for (const listener of stateChangeListeners) {
        try {
            listener(profile, state);
        } catch (error) {
            console.error('[AnyBus] Error in state change listener:', error);
        }
    }
}

/**
 * Initialize centralized state from existing workflow
 * Scans all AnyBus nodes and builds master state from their current configuration
 * IMPORTANT: Assigns unique profile IDs to each BUS-connected group
 *
 * @param {Object} graph - The ComfyUI graph object
 * @param {Function} getBusConnectedNodesFn - Function to get BUS-connected nodes (from Bus.jsx)
 */
export function initializeCentralizedState(graph, getBusConnectedNodesFn = null) {
    if (!graph || !graph._nodes) return;

    console.log('[AnyBus] Initializing centralized state from workflow...');

    // First pass: Find all AnyBus nodes
    const anybusNodes = graph._nodes.filter(n => n.type === "MaraScottAnyBus_v2");

    if (getBusConnectedNodesFn) {
        // Find BUS-connected groups and assign unique profile IDs
        const processedNodes = new Set();
        const connectionGroups = []; // Array of Sets, each Set is a group of connected nodes

        for (const node of anybusNodes) {
            if (processedNodes.has(node.id)) continue;

            // Find all nodes connected to this one via BUS
            const connectedNodeIds = getBusConnectedNodesFn(node);
            const connectedNodes = Array.from(connectedNodeIds)
                .map(id => graph.getNodeById(id))
                .filter(n => n && n.type === "MaraScottAnyBus_v2");

            // Mark all as processed
            connectedNodes.forEach(n => processedNodes.add(n.id));

            // Add this group
            connectionGroups.push(new Set(connectedNodes));
        }

        console.log(`[AnyBus] Found ${connectionGroups.length} BUS-connected group(s)`);

        // Assign unique profile IDs to each connection group
        for (const nodeGroup of connectionGroups) {
            const nodes = Array.from(nodeGroup);
            if (nodes.length === 0) continue;

            // Check if any node already has a profile ID
            let profileId = null;
            for (const node of nodes) {
                if (node._anybus_profileId) {
                    profileId = node._anybus_profileId;
                    break;
                }
            }

            // If no existing ID, generate a new one
            if (!profileId) {
                profileId = generateProfileId();
                console.log(`[AnyBus] Generated new profile ID "${profileId}" for ${nodes.length} node(s)`);
            } else {
                console.log(`[AnyBus] Using existing profile ID "${profileId}" for ${nodes.length} node(s)`);
            }

            // Get or create profile metadata with default label
            const metadata = getProfileMetadata(profileId);

            // Update all nodes in the group to use the same profile ID
            for (const node of nodes) {
                const oldProfileId = node._anybus_profileId;

                if (oldProfileId !== profileId) {
                    // Remove from old profile
                    if (oldProfileId) {
                        getProfileEntry(oldProfileId).delete(node);
                    }

                    // Assign new profile ID
                    node._anybus_profileId = profileId;
                }

                // Register in profile
                getProfileEntry(profileId).add(node);

                // Update node title
                node.title = `AnyBus : ${metadata.label}`;
            }
        }
    }

    // Second pass: Group nodes by their (now unified) profile ID and initialize state
    const profileNodes = new Map();
    for (const node of anybusNodes) {
        const profileId = node._anybus_profileId;
        if (!profileId) {
            console.warn(`[AnyBus] Node ${node.id} has no profile ID, skipping`);
            continue;
        }

        if (!profileNodes.has(profileId)) {
            profileNodes.set(profileId, []);
        }
        profileNodes.get(profileId).push(node);

        // Ensure node is registered
        getProfileEntry(profileId).add(node);
    }

    // Initialize master state for each profile ID
    for (const [profileId, nodes] of profileNodes.entries()) {
        const masterState = getProfileMasterState(profileId);
        const metadata = getProfileMetadata(profileId);

        console.log(`[AnyBus] Initializing profile "${metadata.label}" (ID: ${profileId}) with ${nodes.length} node(s)`);

        // Scan all nodes to collect slot information
        for (const node of nodes) {
            if (!node.inputs) continue;

            for (let i = 1; i < node.inputs.length; i++) {
                const input = node.inputs[i];

                // Extract slot number from input name
                const match = input.name?.match(/\* (\d+)/);
                if (!match) continue;

                const slotNum = parseInt(match[1]);

                // Get or create slot state
                let slotState = masterState.slots[slotNum] || {
                    label: `* ${String(slotNum).padStart(2, '0')}`,
                    type: "*",
                    customLabel: null,
                    hasConnection: false
                };

                // Update connection status
                if (input.link) {
                    slotState.hasConnection = true;

                    // Get connected type
                    const link = graph.links?.[input.link];
                    if (link) {
                        const sourceNode = graph.getNodeById(link.origin_id);
                        if (sourceNode?.outputs) {
                            const sourceOutput = sourceNode.outputs[link.origin_slot];
                            if (sourceOutput?.type && sourceOutput.type !== "*") {
                                // Prioritize non-wildcard types
                                if (slotState.type === "*" || slotState.type === sourceOutput.type) {
                                    slotState.type = sourceOutput.type;
                                }
                            }
                        }
                    }
                }

                // Check if this input has a custom label
                const isDefaultLabel = input.label?.match(/^\* \d{2}$/);
                const isTypeLabel = input.label === input.type;
                const isCustomLabel = input.label && !isDefaultLabel && !isTypeLabel;

                if (isCustomLabel && !slotState.customLabel) {
                    slotState.customLabel = input.label;
                }

                // Update label based on priority
                if (slotState.customLabel) {
                    slotState.label = slotState.customLabel;
                } else if (slotState.type !== "*") {
                    slotState.label = slotState.type;
                } else if (input.label && input.label !== `* ${String(slotNum).padStart(2, '0')}`) {
                    slotState.label = input.label;
                }

                masterState.slots[slotNum] = slotState;
            }

            // Update num_slots from widget
            const numSlotsWidget = node.widgets?.find(w => w.name === "num_slots");
            if (numSlotsWidget && numSlotsWidget.value > (masterState.numSlots || 0)) {
                masterState.numSlots = numSlotsWidget.value;
            }
        }

        masterState.lastUpdate = Date.now();

        console.log(`[AnyBus] Profile "${metadata.label}" (ID: ${profileId}) initialized with ${Object.keys(masterState.slots).length} slot(s):`, masterState.slots);
    }

    // Propagate initialized state to all nodes
    for (const profileId of profileNodes.keys()) {
        propagateMasterStateToNodes(profileId);
    }

    console.log('[AnyBus] Centralized state initialization complete');
}
