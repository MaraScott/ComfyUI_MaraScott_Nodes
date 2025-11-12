// State management for AnyBus_v2
// CENTRALIZED STATE - Sidebar is the single source of truth

// Global profile registry to track all AnyBus nodes by profile
export const profileRegistry = new Map();

// Global slot order registry by profile
// Stores the slot mapping: { profile -> { originalIndex -> newIndex } }
export const profileSlotOrders = new Map();

// CENTRALIZED PROFILE STATE
// Each profile has a master configuration that gets propagated to all nodes
// Structure: { profile -> { slots: { index -> { label, type, customLabel } }, numSlots, slotOrder } }
export const profileMasterState = new Map();

// State change listeners for sidebar updates
const stateChangeListeners = new Set();

// Helper: Get or create profile entry
export function getProfileEntry(profile) {
    if (!profileRegistry.has(profile)) {
        profileRegistry.set(profile, new Set());
    }
    return profileRegistry.get(profile);
}

// Helper: Get or create slot order for profile
export function getProfileSlotOrder(profile) {
    if (!profileSlotOrders.has(profile)) {
        profileSlotOrders.set(profile, {});
    }
    return profileSlotOrders.get(profile);
}

// CENTRALIZED STATE MANAGEMENT

/**
 * Get or create master state for a profile
 */
export function getProfileMasterState(profile) {
    if (!profileMasterState.has(profile)) {
        profileMasterState.set(profile, {
            slots: {}, // index -> { label, type, customLabel, hasConnection }
            numSlots: 3,
            slotOrder: {},
            lastUpdate: Date.now()
        });
    }
    return profileMasterState.get(profile);
}

/**
 * Update master state for a profile and notify listeners
 */
export function updateProfileMasterState(profile, updates) {
    const state = getProfileMasterState(profile);

    if (updates.slots !== undefined) {
        state.slots = { ...state.slots, ...updates.slots };
    }
    if (updates.numSlots !== undefined) {
        state.numSlots = updates.numSlots;
    }
    if (updates.slotOrder !== undefined) {
        state.slotOrder = updates.slotOrder;
    }

    state.lastUpdate = Date.now();

    // Notify all listeners
    notifyStateChange(profile, state);

    return state;
}

/**
 * Sync slot information from a node to master state
 */
export function syncNodeToMasterState(node, slotIndex = null) {
    if (!node || !node._anybus_profile) return;

    const profile = node._anybus_profile;
    const masterState = getProfileMasterState(profile);
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
        notifyStateChange(profile, masterState);
    }

    return hasChanges;
}

/**
 * Propagate master state to all nodes in a profile
 */
export function propagateMasterStateToNodes(profile) {
    const masterState = getProfileMasterState(profile);
    const nodes = profileRegistry.get(profile);

    if (!nodes || nodes.size === 0) return;

    for (const node of nodes) {
        if (!node || !node.graph) continue;

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
