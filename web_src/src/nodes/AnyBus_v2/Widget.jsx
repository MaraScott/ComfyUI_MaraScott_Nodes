// Widget management for AnyBus_v2
import { getProfileEntry } from "./State.jsx";
import { updateNodeSlots } from "./Node.jsx";
import { updateConnectedNodesProfile, updateConnectedNodesSlots, syncConnectedNodesLabelsAndTypes } from "./Bus.jsx";

// Setup profile widget callback
export function setupProfileWidget(node) {
    const profileWidget = node.widgets?.find(w => w.name === "profile");
    if (!profileWidget) return;

    // Store original callback
    const originalCallback = profileWidget.callback;

    // Replace with our callback
    profileWidget.callback = function() {
        const oldProfile = node._anybus_profile;
        const newProfile = profileWidget.value;

        // Remove from old profile registry
        if (oldProfile && oldProfile !== newProfile) {
            getProfileEntry(oldProfile).delete(node);
        }

        // Store new profile on node
        node._anybus_profile = newProfile;

        // Add to new profile registry
        getProfileEntry(newProfile).add(node);

        // Update all BUS-connected nodes
        updateConnectedNodesProfile(node, newProfile);

        // Call original callback if it exists
        if (originalCallback) {
            originalCallback.apply(this, arguments);
        }
    };
}

// Setup num_slots widget callback
export function setupNumSlotsWidget(node) {
    const numSlotsWidget = node.widgets?.find(w => w.name === "num_slots");
    if (!numSlotsWidget) return;

    // Store original callback
    const originalCallback = numSlotsWidget.callback;

    // Replace with our callback
    numSlotsWidget.callback = function() {
        const newNumSlots = numSlotsWidget.value;

        // Update this node's slots
        updateNodeSlots(node, newNumSlots);

        // Update all BUS-connected nodes
        updateConnectedNodesSlots(node, newNumSlots, updateNodeSlots);

        // Call original callback if it exists
        if (originalCallback) {
            originalCallback.apply(this, arguments);
        }
    };
}

// Setup all widgets for a node
export function setupWidgets(node) {
    setupProfileWidget(node);
    setupNumSlotsWidget(node);
}
