import { app } from "../../../scripts/app.js";
import { ensureReactGlobals, mountJSX } from "./AnyBus_v3/React.jsx";

// ---------- Extension exports ----------
const nodeName = "Comfy.MaraScott.AnyBus_v3";
const nodeId = nodeName.replace(/\./g, '-');
const NODE_CLASS = "MaraScott::AnyBus_v3";
const NODE_DISPLAY_NAME = "🐰 AnyBus v3";

// Global profile registry to track all AnyBus nodes by profile
const profileRegistry = new Map();

// Global slot order registry by profile
// Stores the slot mapping: { profile -> { originalIndex -> newIndex } }
const profileSlotOrders = new Map();

// Helper: Get or create profile entry
function getProfileEntry(profile) {
    if (!profileRegistry.has(profile)) {
        profileRegistry.set(profile, new Set());
    }
    return profileRegistry.get(profile);
}

// Helper: Get or create slot order for profile
function getProfileSlotOrder(profile) {
    if (!profileSlotOrders.has(profile)) {
        profileSlotOrders.set(profile, {});
    }
    return profileSlotOrders.get(profile);
}

// Helper: Apply slot order to a node
function applySlotOrderToNode(node, slotOrder) {
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
function updateNodeSlots(node, numSlots) {
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

    // Add missing inputs
    while (node.inputs.length < expectedInputs) {
        const slotNum = node.inputs.length; // 0 = BUS, 1+ = slots
        if (slotNum === 0) {
            node.addInput("bus_input", "ANYBUS_v3");
        } else {
            const label = `* ${String(slotNum).padStart(2, '0')}`;
            node.addInput(`input_${String(slotNum).padStart(2, '0')}`, "*", { label });
        }
    }

    // Remove extra inputs (keep BUS + numSlots)
    while (node.inputs.length > expectedInputs) {
        node.removeInput(node.inputs.length - 1);
    }

    // Add missing outputs
    while (node.outputs.length < expectedOutputs) {
        const slotNum = node.outputs.length;
        if (slotNum === 0) {
            node.addOutput("bus_output", "ANYBUS_v3");
        } else {
            const label = `* ${String(slotNum).padStart(2, '0')}`;
            node.addOutput(label, "*");
        }
    }

    // Remove extra outputs
    while (node.outputs.length > expectedOutputs) {
        node.removeOutput(node.outputs.length - 1);
    }

    node.setDirtyCanvas(true, true);
}

// Helper: Synchronize labels and types across BUS-connected nodes
function syncConnectedNodesLabelsAndTypes(node) {
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

// Helper: Synchronize labels across nodes with same profile (legacy - now uses syncConnectedNodesLabelsAndTypes)
function syncProfileLabels(node) {
    syncConnectedNodesLabelsAndTypes(node);
}

// Helper: Handle input type changes
function handleInputTypeChange(node, slotIndex, newType) {
    if (!node.inputs || slotIndex >= node.inputs.length) return;

    const input = node.inputs[slotIndex];
    if (input.type !== newType) {
        input.type = newType;
        node.setDirtyCanvas(true, true);
    }

    // Sync to matching outputs
    if (node.outputs && slotIndex < node.outputs.length) {
        const output = node.outputs[slotIndex];
        if (output.type !== newType) {
            output.type = newType;
            node.setDirtyCanvas(true, true);
        }
    }
}

// Helper: Get all BUS-connected nodes recursively
function getBusConnectedNodes(node, visited = new Set()) {
    if (!node || visited.has(node.id)) return visited;
    visited.add(node.id);

    const nodes = [node];

    // Check all inputs for BUS connections
    if (node.inputs) {
        for (let i = 0; i < node.inputs.length; i++) {
            const input = node.inputs[i];
            if (input.type === "ANYBUS_v3" && input.link) {
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
            if (output.type === "ANYBUS_v3" && output.links) {
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

// Helper: Update profile for all BUS-connected nodes
function updateConnectedNodesProfile(node, newProfile) {
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
    syncProfileLabels(node);
}

// Helper: Update num_slots for all BUS-connected nodes
function updateConnectedNodesSlots(node, newNumSlots) {
    const connectedNodeIds = getBusConnectedNodes(node);

    for (const nodeId of connectedNodeIds) {
        const connectedNode = node.graph?.getNodeById(nodeId);
        if (!connectedNode) continue;

        const numSlotsWidget = connectedNode.widgets?.find(w => w.name === "num_slots");
        if (numSlotsWidget && numSlotsWidget.value !== newNumSlots) {
            // Update widget value
            numSlotsWidget.value = newNumSlots;

            // Update the node's inputs/outputs
            updateNodeSlots(connectedNode, newNumSlots);
        }
    }
}

const MaraScottAnyBusNodeExtension = () => {
    return {
        name: nodeName,
        aboutPageBadges: [
            { label: "Website - MaraScott", url: "https://www.marascott.ai/", icon: "pi pi-home" },
            { label: "Donate - MaraScott", url: "https://github.com/sponsors/MaraScott", icon: "pi pi-heart" },
            { label: "GitHub - MaraScott", url: "https://github.com/MaraScott/ComfyUI_MaraScott_Nodes", icon: "pi pi-github" }
        ],

        async addCustomNodeDefs(defs, app) {
            // Define the custom node entirely in JavaScript
            defs[NODE_CLASS] = {
                name: NODE_CLASS,
                display_name: NODE_DISPLAY_NAME,
                category: "MaraScott/Bus",
                input: {
                    required: {
                        num_slots: ["INT", {
                            default: 3,
                            min: 1,
                            max: 20,
                            step: 1,
                            display: "number",
                            tooltip: "Number of any-type input/output slots (excluding the BUS connection)"
                        }],
                        profile: ["STRING", {
                            default: "default",
                            multiline: false,
                            tooltip: "Profile name for bus linking. Same profile nodes connect via BUS. 'default' can connect to any profile."
                        }]
                    },
                    optional: {
                        bus_input: ["ANYBUS_v3", {
                            tooltip: "BUS input connection from another AnyBus node"
                        }],
                        // Add default 3 dynamic slots
                        input_01: ["*", { tooltip: "Slot 1" }],
                        input_02: ["*", { tooltip: "Slot 2" }],
                        input_03: ["*", { tooltip: "Slot 3" }],
                    }
                },
                output: ["ANYBUS_v3", "*", "*", "*"],
                output_name: ["bus_output", "* 01", "* 02", "* 03"],
                output_node: false,
                description: "Dynamic bus connection system with profile-based linking and type synchronization"
            };
        },

        async beforeRegisterNodeDef(nodeType, nodeData, app) {
            if (nodeData.name !== NODE_CLASS) return;

            // Store original methods
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            const onConnectionsChange = nodeType.prototype.onConnectionsChange;
            const onConnectInput = nodeType.prototype.onConnectInput;

            // Override onNodeCreated
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated?.apply(this, arguments);

                // Initialize slots based on num_slots widget
                const numSlotsWidget = this.widgets?.find(w => w.name === "num_slots");
                const profileWidget = this.widgets?.find(w => w.name === "profile");

                if (numSlotsWidget) {
                    // Initial setup
                    updateNodeSlots(this, numSlotsWidget.value);

                    // Watch for changes - preserve the original callback and context
                    const originalCallback = numSlotsWidget.callback;
                    const node = this;
                    numSlotsWidget.callback = function(value, ...args) {
                        // Call original callback with proper context and all arguments
                        const result = originalCallback?.apply(this, arguments);
                        // Update this node's slots
                        updateNodeSlots(node, value);
                        // Update all BUS-connected nodes' slots
                        updateConnectedNodesSlots(node, value);
                        return result;
                    };
                }

                if (profileWidget) {
                    // Register in profile registry
                    const profile = profileWidget.value || "default";
                    getProfileEntry(profile).add(this);

                    // Watch for profile changes - preserve the original callback and context
                    const originalCallback = profileWidget.callback;
                    const node = this;
                    profileWidget.callback = function(value, ...args) {
                        // Call original callback with proper context and all arguments
                        const result = originalCallback?.apply(this, arguments);

                        // Update all BUS-connected nodes with the new profile
                        updateConnectedNodesProfile(node, value);

                        return result;
                    };
                }

                // Store reference for cleanup
                this._anybus_profile = profileWidget?.value || "default";

                return r;
            };

            // Override onConnectionsChange
            nodeType.prototype.onConnectionsChange = function(side, slot, connect, link_info, output) {
                const r = onConnectionsChange?.apply(this, arguments);

                // side: 1 = input, 2 = output
                // connect: true = connecting, false = disconnecting

                if (side === 1 && slot > 0) { // Input connection (not BUS)
                    // Sync all BUS-connected nodes after any connection change
                    syncConnectedNodesLabelsAndTypes(this);
                }

                // Handle BUS connections (slot 0)
                if (slot === 0) {
                    if (connect && link_info) {
                        const sourceNode = this.graph?.getNodeById(link_info.origin_id);
                        const profileWidget = this.widgets?.find(w => w.name === "profile");

                        if (sourceNode && profileWidget) {
                            const sourceProfile = sourceNode.widgets?.find(w => w.name === "profile")?.value;
                            const currentProfile = profileWidget.value;

                            // If this node is "default", adopt the source profile
                            if (currentProfile === "default" && sourceProfile && sourceProfile !== "default") {
                                // Remove from default profile
                                getProfileEntry("default").delete(this);

                                // Update widget
                                profileWidget.value = sourceProfile;

                                // Add to new profile
                                getProfileEntry(sourceProfile).add(this);
                                this._anybus_profile = sourceProfile;
                            }
                        }
                    }

                    // Sync after BUS connection/disconnection
                    syncConnectedNodesLabelsAndTypes(this);
                }

                return r;
            };

            // Override onConnectInput to validate connections
            nodeType.prototype.onConnectInput = function(targetSlot, type, output, originNode, originSlot) {
                // Validate BUS connections (slot 0)
                if (targetSlot === 0 && type === "ANYBUS_v3") {
                    const profileWidget = this.widgets?.find(w => w.name === "profile");
                    const sourceProfileWidget = originNode?.widgets?.find(w => w.name === "profile");

                    if (profileWidget && sourceProfileWidget) {
                        const targetProfile = profileWidget.value;
                        const sourceProfile = sourceProfileWidget.value;

                        // Allow connection if:
                        // 1. Both have the same profile
                        // 2. Target is "default" (will adopt source profile)
                        if (targetProfile !== sourceProfile && targetProfile !== "default") {
                            console.warn(`[AnyBus] Cannot connect profile "${sourceProfile}" to "${targetProfile}"`);
                            return false;
                        }
                    }
                }

                // Validate type compatibility for regular inputs (slot > 0)
                if (targetSlot > 0 && this.inputs && this.inputs[targetSlot]) {
                    const input = this.inputs[targetSlot];
                    const outputType = output?.type || type;

                    // If input has a specific type (not wildcard)
                    if (input.type !== "*" && input.type !== outputType) {
                        // Check if any BUS-connected node has this slot connected with a different type
                        const connectedNodeIds = getBusConnectedNodes(this);

                        for (const nodeId of connectedNodeIds) {
                            const connectedNode = this.graph?.getNodeById(nodeId);
                            if (!connectedNode || !connectedNode.inputs || nodeId === this.id) continue;

                            const connectedInput = connectedNode.inputs[targetSlot];
                            if (connectedInput && connectedInput.link) {
                                const link = connectedNode.graph?.links[connectedInput.link];
                                if (link) {
                                    const sourceNode = connectedNode.graph?.getNodeById(link.origin_id);
                                    if (sourceNode && sourceNode.outputs) {
                                        const sourceOutput = sourceNode.outputs[link.origin_slot];
                                        if (sourceOutput && sourceOutput.type !== "*" && sourceOutput.type !== outputType) {
                                            console.warn(`[AnyBus] Cannot connect ${outputType} to slot ${targetSlot}. Slot is already connected to ${sourceOutput.type} in BUS network.`);
                                            return false;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                const r = onConnectInput?.apply(this, arguments);
                return r;
            };

            // Cleanup on removal
            const onRemoved = nodeType.prototype.onRemoved;
            nodeType.prototype.onRemoved = function() {
                // Remove from profile registry
                if (this._anybus_profile) {
                    getProfileEntry(this._anybus_profile).delete(this);
                }
                return onRemoved?.apply(this, arguments);
            };
        },

        async setup() {
            console.log(`[MaraScott] ${nodeName} initialized`);
        },
    };
};

const MaraScottAnyBusNodeSidebarTab = () => {
    return {
        id: nodeId,
        icon: "mdi mdi-transit-connection-variant",
        title: "Any Bus v3",
        tooltip: "Any Bus v3 Dashboard",
        type: "custom",
        render: async (el) => {
            await ensureReactGlobals();
            const React = globalThis.React;

            function FlowSidebar() {
                const [profiles, setProfiles] = React.useState([]);
                const [selectedProfile, setSelectedProfile] = React.useState(null);
                const [slotOrders, setSlotOrders] = React.useState({});
                const [lastUpdate, setLastUpdate] = React.useState(new Date());
                const [draggedSlot, setDraggedSlot] = React.useState(null);

                React.useEffect(() => {
                    const updateProfiles = () => {
                        const profileData = [];
                        for (const [profile, nodes] of profileRegistry.entries()) {
                            // Get first node to extract slot info
                            const firstNode = Array.from(nodes)[0];
                            const numSlotsWidget = firstNode?.widgets?.find(w => w.name === "num_slots");
                            const numSlots = numSlotsWidget?.value || 3;

                            // Extract slot information
                            const slots = [];
                            if (firstNode && firstNode.inputs) {
                                for (let i = 1; i <= numSlots && i < firstNode.inputs.length; i++) {
                                    const input = firstNode.inputs[i];
                                    const output = firstNode.outputs[i];
                                    slots.push({
                                        index: i,
                                        label: input?.label || `* ${String(i).padStart(2, '0')}`,
                                        type: input?.type || "*",
                                        hasConnection: input?.link ? true : false
                                    });
                                }
                            }

                            const nodeList = Array.from(nodes).map(n => {
                                let inputConnections = 0;
                                let outputConnections = 0;
                                if (n.inputs) {
                                    inputConnections = n.inputs.filter(i => i.link).length;
                                }
                                if (n.outputs) {
                                    for (const output of n.outputs) {
                                        if (output.links && output.links.length > 0) {
                                            outputConnections += output.links.length;
                                        }
                                    }
                                }

                                return {
                                    id: n.id,
                                    title: n.title || `Node ${n.id}`,
                                    numSlots: numSlots,
                                    inputConnections,
                                    outputConnections,
                                };
                            });

                            profileData.push({
                                name: profile,
                                nodeCount: nodes.size,
                                nodes: nodeList,
                                slots: slots
                            });
                        }
                        setProfiles(profileData);
                        setLastUpdate(new Date());
                    };

                    updateProfiles();
                    const interval = setInterval(updateProfiles, 500);
                    return () => clearInterval(interval);
                }, []);

                const handleReorderSlots = (profileName, fromIndex, toIndex) => {
                    if (fromIndex === toIndex) return;

                    // Update slot order for this profile
                    const profile = profiles.find(p => p.name === profileName);
                    if (!profile) return;

                    const newSlots = [...profile.slots];
                    const [movedSlot] = newSlots.splice(fromIndex, 1);
                    newSlots.splice(toIndex, 0, movedSlot);

                    // Create new slot order mapping
                    const newOrder = {};
                    newSlots.forEach((slot, newIdx) => {
                        newOrder[slot.index] = newIdx + 1; // +1 because slot 0 is BUS
                    });

                    // Store the order
                    profileSlotOrders.set(profileName, newOrder);

                    // Apply to all nodes in this profile
                    const profileNodes = profileRegistry.get(profileName);
                    if (profileNodes) {
                        for (const node of profileNodes) {
                            applySlotOrderToNode(node, newOrder);
                        }
                    }

                    // Force UI update
                    setLastUpdate(new Date());
                };

                const handleDragStart = (e, profileName, slotIndex) => {
                    setDraggedSlot({ profileName, slotIndex });
                    e.dataTransfer.effectAllowed = 'move';
                };

                const handleDragOver = (e) => {
                    e.preventDefault();
                    e.dataTransfer.dropEffect = 'move';
                };

                const handleDrop = (e, profileName, dropIndex) => {
                    e.preventDefault();
                    if (draggedSlot && draggedSlot.profileName === profileName) {
                        handleReorderSlots(profileName, draggedSlot.slotIndex, dropIndex);
                    }
                    setDraggedSlot(null);
                };

                const handleResetNodeLabels = (nodeId) => {
                    // Find the node in the graph
                    const node = app.graph?._nodes_by_id?.[nodeId];
                    if (!node || !node.inputs || !node.outputs) return;

                    // Reset labels for disconnected slots
                    for (let i = 1; i < node.inputs.length; i++) {
                        const input = node.inputs[i];
                        // Only reset if not connected
                        if (!input.link) {
                            const defaultLabel = `* ${String(i).padStart(2, '0')}`;
                            input.label = defaultLabel;
                            input.type = "*";

                            // Reset corresponding output
                            if (node.outputs[i]) {
                                node.outputs[i].label = defaultLabel;
                                node.outputs[i].type = "*";
                            }
                        }
                    }

                    node.setDirtyCanvas(true, true);

                    // Sync changes across BUS-connected nodes
                    syncConnectedNodesLabelsAndTypes(node);

                    // Force UI update
                    setLastUpdate(new Date());
                };

                return (
                    <div style={{ padding: '10px', fontFamily: 'system-ui', color: '#ccc', height: '100%', overflow: 'auto' }}>
                        <h2 style={{ marginTop: 0, color: '#fff' }}>Any Bus v3 Dashboard</h2>
                        <p style={{ fontSize: '0.9em', color: '#888' }}>
                            Live monitoring • Updated: {lastUpdate.toLocaleTimeString()}
                        </p>

                        <div style={{ marginTop: '20px' }}>
                            <h3 style={{ color: '#fff' }}>Profiles ({profiles.length})</h3>
                            {profiles.length === 0 ? (
                                <p style={{ color: '#888', fontStyle: 'italic' }}>No AnyBus nodes in workflow</p>
                            ) : (
                                profiles.map(profile => (
                                    <div key={profile.name} style={{
                                        border: '1px solid #444',
                                        borderRadius: '6px',
                                        padding: '12px',
                                        marginBottom: '12px',
                                        backgroundColor: '#1a1a1a',
                                        boxShadow: '0 2px 4px rgba(0,0,0,0.3)'
                                    }}>
                                        <div
                                            style={{
                                                fontWeight: 'bold',
                                                marginBottom: '8px',
                                                fontSize: '1.1em',
                                                color: '#4a9eff',
                                                cursor: 'pointer',
                                                display: 'flex',
                                                justifyContent: 'space-between',
                                                alignItems: 'center'
                                            }}
                                            onClick={() => setSelectedProfile(selectedProfile === profile.name ? null : profile.name)}
                                        >
                                            <span>
                                                📦 {profile.name}
                                                <span style={{
                                                    marginLeft: '10px',
                                                    color: '#888',
                                                    fontSize: '0.85em',
                                                    fontWeight: 'normal'
                                                }}>
                                                    ({profile.nodeCount} {profile.nodeCount === 1 ? 'node' : 'nodes'})
                                                </span>
                                            </span>
                                            <span style={{ fontSize: '0.8em' }}>
                                                {selectedProfile === profile.name ? '▼' : '▶'}
                                            </span>
                                        </div>

                                        {selectedProfile === profile.name && (
                                            <>
                                                {/* Slot Reordering Section */}
                                                <div style={{
                                                    borderTop: '1px solid #333',
                                                    paddingTop: '8px',
                                                    marginBottom: '12px'
                                                }}>
                                                    <div style={{ fontSize: '0.95em', fontWeight: 'bold', marginBottom: '8px', color: '#ddd' }}>
                                                        📋 Slot Order (Drag to reorder)
                                                    </div>
                                                    {profile.slots && profile.slots.length > 0 ? (
                                                        profile.slots.map((slot, idx) => (
                                                            <div
                                                                key={slot.index}
                                                                draggable
                                                                onDragStart={(e) => handleDragStart(e, profile.name, idx)}
                                                                onDragOver={handleDragOver}
                                                                onDrop={(e) => handleDrop(e, profile.name, idx)}
                                                                style={{
                                                                    padding: '8px 10px',
                                                                    marginBottom: '4px',
                                                                    backgroundColor: draggedSlot?.slotIndex === idx && draggedSlot?.profileName === profile.name ? '#2a4a6a' : '#252525',
                                                                    border: '1px solid #444',
                                                                    borderRadius: '4px',
                                                                    cursor: 'move',
                                                                    display: 'flex',
                                                                    justifyContent: 'space-between',
                                                                    alignItems: 'center',
                                                                    fontSize: '0.9em'
                                                                }}
                                                            >
                                                                <span style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                                                                    <span style={{ color: '#888' }}>☰</span>
                                                                    <span style={{ fontWeight: '500', color: '#ccc' }}>
                                                                        {slot.label}
                                                                    </span>
                                                                </span>
                                                                <span style={{
                                                                    fontSize: '0.85em',
                                                                    color: slot.hasConnection ? '#6a6' : '#888',
                                                                    fontFamily: 'monospace'
                                                                }}>
                                                                    {slot.type}
                                                                </span>
                                                            </div>
                                                        ))
                                                    ) : (
                                                        <p style={{ color: '#888', fontSize: '0.85em', fontStyle: 'italic' }}>No slots</p>
                                                    )}
                                                </div>

                                                {/* Nodes Section */}
                                                <div style={{
                                                    borderTop: '1px solid #333',
                                                    paddingTop: '8px'
                                                }}>
                                                    <div style={{ fontSize: '0.95em', fontWeight: 'bold', marginBottom: '8px', color: '#ddd' }}>
                                                        🔗 Nodes
                                                    </div>
                                                    {profile.nodes.map(node => (
                                                        <div key={node.id} style={{
                                                            fontSize: '0.9em',
                                                            color: '#ccc',
                                                            marginBottom: '6px',
                                                            padding: '6px 8px',
                                                            backgroundColor: '#151515',
                                                            borderRadius: '4px',
                                                            display: 'flex',
                                                            justifyContent: 'space-between',
                                                            alignItems: 'center'
                                                        }}>
                                                            <span style={{ fontWeight: '500', flex: 1 }}>
                                                                {node.title}
                                                            </span>
                                                            <div style={{
                                                                fontSize: '0.85em',
                                                                color: '#888',
                                                                display: 'flex',
                                                                gap: '12px',
                                                                alignItems: 'center'
                                                            }}>
                                                                <span title="Number of slots">
                                                                    🎰 {node.numSlots}
                                                                </span>
                                                                <span title="Input connections">
                                                                    📥 {node.inputConnections}
                                                                </span>
                                                                <span title="Output connections">
                                                                    📤 {node.outputConnections}
                                                                </span>
                                                                <button
                                                                    onClick={() => handleResetNodeLabels(node.id)}
                                                                    title="Reset disconnected slots to default labels"
                                                                    style={{
                                                                        background: 'none',
                                                                        border: '1px solid #555',
                                                                        borderRadius: '3px',
                                                                        color: '#aaa',
                                                                        cursor: 'pointer',
                                                                        padding: '2px 6px',
                                                                        fontSize: '0.9em',
                                                                        display: 'flex',
                                                                        alignItems: 'center',
                                                                        gap: '4px'
                                                                    }}
                                                                    onMouseOver={(e) => {
                                                                        e.currentTarget.style.background = '#333';
                                                                        e.currentTarget.style.borderColor = '#777';
                                                                    }}
                                                                    onMouseOut={(e) => {
                                                                        e.currentTarget.style.background = 'none';
                                                                        e.currentTarget.style.borderColor = '#555';
                                                                    }}
                                                                >
                                                                    🔄
                                                                </button>
                                                            </div>
                                                        </div>
                                                    ))}
                                                </div>
                                            </>
                                        )}
                                    </div>
                                ))
                            )}
                        </div>
                    </div>
                );
            }

            // Clear element and render
            el.innerHTML = '';
            mountJSX(el, <FlowSidebar />);
        }
    };
};

export { MaraScottAnyBusNodeExtension, MaraScottAnyBusNodeSidebarTab };
