import { app } from "../../../scripts/app.js";
import { ensureReactGlobals, mountJSX } from "./AnyBus_v2/React.jsx";
import { getProfileEntry, getProfileSlotOrder, profileSlotOrders, profileRegistry } from "./AnyBus_v2/State.jsx";
import { applySlotOrderToNode, updateNodeSlots, resetNodeDisconnectedSlots } from "./AnyBus_v2/Node.jsx";
import {
    getBusConnectedNodes,
    syncConnectedNodesLabelsAndTypes,
    updateConnectedNodesProfile,
    updateConnectedNodesSlots,
    resetProfileDisconnectedSlots
} from "./AnyBus_v2/Bus.jsx";
import { setupWidgets } from "./AnyBus_v2/Widget.jsx";

// ---------- Extension exports ----------
const nodeName = "Comfy.MaraScott.AnyBus_v2";
const nodeId = nodeName.replace(/\./g, '-');
const NODE_CLASS = "MaraScottAnyBus_v2";  // Matches Python registration
const NODE_DISPLAY_NAME = "AnyBus v2";


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

// ---------- Extension definition ----------
const MaraScottAnyBusNodeExtension = () => {
    return {
        name: nodeName,
        aboutPageBadges: [
            { label: "Website - MaraScott", url: "https://www.marascott.ai/", icon: "pi pi-home" },
            { label: "Donate - MaraScott", url: "https://github.com/sponsors/MaraScott", icon: "pi pi-heart" },
            { label: "GitHub - MaraScott", url: "https://github.com/MaraScott/ComfyUI_MaraScott_Nodes", icon: "pi pi-github" }
        ],

        async beforeRegisterNodeDef(nodeType, nodeData, app) {
            if (nodeData.name !== NODE_CLASS) return;

            // Store original methods
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            const onConnectionsChange = nodeType.prototype.onConnectionsChange;
            const onConnectInput = nodeType.prototype.onConnectInput;            // Override onNodeCreated
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated?.apply(this, arguments);

                // Initialize slots based on num_slots widget
                const numSlotsWidget = this.widgets?.find(w => w.name === "num_slots");
                const profileWidget = this.widgets?.find(w => w.name === "profile");
                const modeWidget = this.widgets?.find(w => w.name === "mode");
                const getsetWidget = this.widgets?.find(w => w.name === "getset_source");

                if (numSlotsWidget) {
                    // Initial setup
                    updateNodeSlots(this, numSlotsWidget.value);
                }

                if (profileWidget) {
                    // Register in profile registry
                    const profile = profileWidget.value || "default";
                    getProfileEntry(profile).add(this);

                    // Store reference for cleanup
                    this._anybus_profile = profile;
                }

                // Setup mode widget callback
                if (modeWidget && getsetWidget) {
                    const originalCallback = modeWidget.callback;
                    modeWidget.callback = (value) => {
                        if (originalCallback) originalCallback.call(modeWidget, value);

                        const mode = value;
                        const busInput = this.inputs?.[0];

                        if (mode === "getset") {
                            // Hide bus input when in getset mode
                            if (busInput && busInput.type === "ANYBUS_v2") {
                                busInput.type = -1; // Hide input
                            }
                            // Show getset_source widget
                            if (getsetWidget) {
                                getsetWidget.type = "combo";
                                getsetWidget.options = { values: () => this.getAvailableGetSetSources() };
                            }
                        } else {
                            // Show bus input when in bus mode
                            if (busInput && busInput.type === -1) {
                                busInput.type = "ANYBUS_v2";
                            }
                            // Hide getset_source widget
                            if (getsetWidget) {
                                getsetWidget.type = "converted-widget";
                            }
                        }

                        this.setSize(this.computeSize());
                        this.setDirtyCanvas(true, true);
                    };

                    // Trigger initial callback to set up UI
                    if (modeWidget.value) {
                        modeWidget.callback(modeWidget.value);
                    }
                }

                // Add method to get available Get/Set sources
                this.getAvailableGetSetSources = function() {
                    if (!this.graph) return [""];

                    const sources = [""];
                    const profileWidget = this.widgets?.find(w => w.name === "profile");
                    const currentProfile = profileWidget?.value || "default";

                    // Find all AnyBus_v2 nodes with the same profile
                    for (const node of this.graph._nodes) {
                        if (node.type === NODE_CLASS && node.id !== this.id) {
                            const nodeProfile = node.widgets?.find(w => w.name === "profile")?.value;
                            if (nodeProfile === currentProfile || nodeProfile === "default" || currentProfile === "default") {
                                // Use node title as identifier
                                const identifier = node.title || `Node ${node.id}`;
                                sources.push(identifier);
                            }
                        }
                    }

                    return sources;
                };

                // Setup widget callbacks using the Widget module
                setupWidgets(this);

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
                        const numSlotsWidget = this.widgets?.find(w => w.name === "num_slots");

                        if (sourceNode && profileWidget) {
                            const sourceProfile = sourceNode.widgets?.find(w => w.name === "profile")?.value;
                            const sourceNumSlots = sourceNode.widgets?.find(w => w.name === "num_slots")?.value;
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

                            // Sync number of slots with source node
                            if (numSlotsWidget && sourceNumSlots !== undefined && numSlotsWidget.value !== sourceNumSlots) {
                                numSlotsWidget.value = sourceNumSlots;
                                updateNodeSlots(this, sourceNumSlots);
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
                if (targetSlot === 0 && type === "ANYBUS_v2") {
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
        title: "Any Bus v2",
        tooltip: "Any Bus v2 Dashboard",
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
                        const processedNodes = new Set();

                        // Iterate through all profiles
                        for (const [profile, nodes] of profileRegistry.entries()) {
                            // For each node in this profile, find its BUS-connected group
                            for (const node of nodes) {
                                if (processedNodes.has(node.id)) continue;

                                // Get all BUS-connected nodes for this node
                                const connectedNodeIds = getBusConnectedNodes(node);

                                // Filter to only nodes with the same profile
                                const groupNodes = Array.from(connectedNodeIds)
                                    .map(id => app.graph?._nodes_by_id?.[id])
                                    .filter(n => n && n._anybus_profile === profile);

                                // Skip if no valid nodes found
                                if (groupNodes.length === 0) continue;

                                // Mark all these nodes as processed
                                groupNodes.forEach(n => processedNodes.add(n.id));

                                // Get first node to extract slot info
                                const firstNode = groupNodes[0];
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

                                const nodeList = groupNodes.map(n => {
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
                                        numSlots: numSlots,
                                        inputConnections,
                                        outputConnections,
                                    };
                                });

                                // Create a unique identifier for this group (profile + first node id)
                                const groupId = `${profile}_${firstNode.id}`;

                                profileData.push({
                                    name: profile,
                                    groupId: groupId,
                                    nodeCount: groupNodes.length,
                                    nodes: nodeList,
                                    slots: slots
                                });
                            }
                        }
                        setProfiles(profileData);
                        setLastUpdate(new Date());
                    };

                    updateProfiles();
                    const interval = setInterval(updateProfiles, 500);
                    return () => clearInterval(interval);
                }, []);

                const handleReorderSlots = (groupId, fromIndex, toIndex) => {
                    if (fromIndex === toIndex) return;

                    // Find the profile group by groupId
                    const profileGroup = profiles.find(p => p.groupId === groupId);
                    if (!profileGroup) return;

                    const newSlots = [...profileGroup.slots];
                    const [movedSlot] = newSlots.splice(fromIndex, 1);
                    newSlots.splice(toIndex, 0, movedSlot);

                    // Create new slot order mapping
                    const newOrder = {};
                    newSlots.forEach((slot, newIdx) => {
                        newOrder[slot.index] = newIdx + 1; // +1 because slot 0 is BUS
                    });

                    // Store the order using profile name (not groupId)
                    profileSlotOrders.set(profileGroup.name, newOrder);

                    // Apply to all nodes in this specific group
                    // Get the first node from this group to find all BUS-connected nodes
                    if (profileGroup.nodes.length > 0) {
                        const firstNodeId = profileGroup.nodes[0].id;
                        const firstNode = app.graph?._nodes_by_id?.[firstNodeId];

                        if (firstNode) {
                            const connectedNodeIds = getBusConnectedNodes(firstNode);
                            for (const nodeId of connectedNodeIds) {
                                const node = app.graph?._nodes_by_id?.[nodeId];
                                if (node && node._anybus_profile === profileGroup.name) {
                                    applySlotOrderToNode(node, newOrder);
                                }
                            }
                        }
                    }

                    // Force UI update
                    setLastUpdate(new Date());
                };

                const handleDragStart = (e, groupId, slotIndex) => {
                    setDraggedSlot({ groupId, slotIndex });
                    e.dataTransfer.effectAllowed = 'move';
                };

                const handleDragOver = (e) => {
                    e.preventDefault();
                    e.dataTransfer.dropEffect = 'move';
                };

                const handleDrop = (e, groupId, dropIndex) => {
                    e.preventDefault();
                    if (draggedSlot && draggedSlot.groupId === groupId) {
                        handleReorderSlots(groupId, draggedSlot.slotIndex, dropIndex);
                    }
                    setDraggedSlot(null);
                };

                const handleResetProfile = (groupId) => {
                    // Find the profile group
                    const profileGroup = profiles.find(p => p.groupId === groupId);
                    if (!profileGroup || profileGroup.nodes.length === 0) return;

                    // Get first node to access the BUS-connected group
                    const firstNodeId = profileGroup.nodes[0].id;
                    const firstNode = app.graph?._nodes_by_id?.[firstNodeId];

                    if (!firstNode) return;

                    // Reset only the BUS-connected nodes in this specific group
                    const connectedNodeIds = getBusConnectedNodes(firstNode);
                    let hasAnyChanges = false;

                    for (const nodeId of connectedNodeIds) {
                        const node = app.graph?._nodes_by_id?.[nodeId];
                        if (node && node._anybus_profile === profileGroup.name) {
                            const changed = resetNodeDisconnectedSlots(node);
                            if (changed) {
                                hasAnyChanges = true;
                            }
                        }
                    }

                    // Sync changes across the group if there were any changes
                    if (hasAnyChanges) {
                        syncConnectedNodesLabelsAndTypes(firstNode);
                        setLastUpdate(new Date());
                    }
                };

                const handleLabelChange = (groupId, slotIndex, newLabel) => {
                    // Find the profile group
                    const profileGroup = profiles.find(p => p.groupId === groupId);
                    if (!profileGroup || profileGroup.nodes.length === 0) return;

                    // Get first node to access the BUS-connected group
                    const firstNodeId = profileGroup.nodes[0].id;
                    const firstNode = app.graph?._nodes_by_id?.[firstNodeId];

                    if (!firstNode) return;

                    // Update label for all BUS-connected nodes in this group
                    const connectedNodeIds = getBusConnectedNodes(firstNode);

                    for (const nodeId of connectedNodeIds) {
                        const node = app.graph?._nodes_by_id?.[nodeId];
                        if (node && node._anybus_profile === profileGroup.name) {
                            // Update input label
                            if (node.inputs && node.inputs[slotIndex]) {
                                node.inputs[slotIndex].label = newLabel || node.inputs[slotIndex].type;
                            }
                            // Update output label
                            if (node.outputs && node.outputs[slotIndex]) {
                                node.outputs[slotIndex].label = newLabel || node.outputs[slotIndex].type;
                            }
                            node.setDirtyCanvas(true, true);
                        }
                    }

                    // Force UI update
                    setLastUpdate(new Date());
                };

                const handleProfileNameChange = (groupId, newProfileName) => {
                    if (!newProfileName || newProfileName.trim() === '') return;

                    // Find the profile group
                    const profileGroup = profiles.find(p => p.groupId === groupId);
                    if (!profileGroup || profileGroup.nodes.length === 0) return;

                    // Get first node to access the BUS-connected group
                    const firstNodeId = profileGroup.nodes[0].id;
                    const firstNode = app.graph?._nodes_by_id?.[firstNodeId];

                    if (!firstNode) return;

                    // Update profile name for all nodes in this group
                    updateConnectedNodesProfile(firstNode, newProfileName.trim());

                    // Force UI update
                    setLastUpdate(new Date());
                };

                const handleNumSlotsChange = (groupId, newNumSlots) => {
                    const numSlots = parseInt(newNumSlots);
                    if (isNaN(numSlots) || numSlots < 1 || numSlots > 20) return;

                    // Find the profile group
                    const profileGroup = profiles.find(p => p.groupId === groupId);
                    if (!profileGroup || profileGroup.nodes.length === 0) return;

                    // Get first node to access the BUS-connected group
                    const firstNodeId = profileGroup.nodes[0].id;
                    const firstNode = app.graph?._nodes_by_id?.[firstNodeId];

                    if (!firstNode) return;

                    // Update num_slots widget and slots for all nodes in this group
                    const connectedNodeIds = getBusConnectedNodes(firstNode);

                    for (const nodeId of connectedNodeIds) {
                        const node = app.graph?._nodes_by_id?.[nodeId];
                        if (node && node._anybus_profile === profileGroup.name) {
                            const numSlotsWidget = node.widgets?.find(w => w.name === "num_slots");
                            if (numSlotsWidget) {
                                numSlotsWidget.value = numSlots;
                                updateNodeSlots(node, numSlots);
                            }
                        }
                    }

                    // Force UI update
                    setLastUpdate(new Date());
                };                return (
                    <div style={{ padding: '10px', fontFamily: 'system-ui', color: '#ccc', height: '100%', overflow: 'auto' }}>
                        <h2 style={{ marginTop: 0, color: '#fff' }}>Any Bus v2 Dashboard</h2>
                        <p style={{ fontSize: '0.9em', color: '#888' }}>
                            Live monitoring • Updated: {lastUpdate.toLocaleTimeString()}
                        </p>

                        <div style={{ marginTop: '20px' }}>
                            <h3 style={{ color: '#fff' }}>Profiles ({profiles.length})</h3>
                            {profiles.length === 0 ? (
                                <p style={{ color: '#888', fontStyle: 'italic' }}>No AnyBus nodes in workflow</p>
                            ) : (
                                profiles.map(profile => (
                                    <div key={profile.groupId} style={{
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
                                            onClick={() => setSelectedProfile(selectedProfile === profile.groupId ? null : profile.groupId)}
                                        >
                                            <span>
                                                � Profile: <span style={{ color: '#6af' }}>{profile.name}</span>
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
                                                {selectedProfile === profile.groupId ? '▼' : '▶'}
                                            </span>
                                        </div>

                                        {selectedProfile === profile.groupId && (
                                            <>
                                                <div style={{
                                                    fontSize: '0.85em',
                                                    color: '#999',
                                                    fontStyle: 'italic',
                                                    marginBottom: '12px',
                                                    paddingTop: '8px',
                                                    borderTop: '1px solid #333'
                                                }}>
                                                    Current settings for this profile group. Connected nodes sync these settings.
                                                </div>

                                                {/* Profile Settings Section */}
                                                <div style={{
                                                    borderTop: '1px solid #333',
                                                    paddingTop: '8px',
                                                    marginBottom: '12px',
                                                    display: 'flex',
                                                    flexDirection: 'column',
                                                    gap: '8px'
                                                }}>
                                                    <div style={{ fontSize: '0.95em', fontWeight: 'bold', marginBottom: '4px', color: '#ddd' }}>
                                                        ⚙️ Profile Settings
                                                    </div>

                                                    {/* Profile Name */}
                                                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                                        <label style={{ color: '#aaa', fontSize: '0.9em', minWidth: '80px' }}>
                                                            Profile:
                                                        </label>
                                                        <input
                                                            type="text"
                                                            value={profile.name}
                                                            onChange={(e) => handleProfileNameChange(profile.groupId, e.target.value)}
                                                            style={{
                                                                background: '#1a1a1a',
                                                                border: '1px solid #555',
                                                                borderRadius: '3px',
                                                                color: '#ccc',
                                                                padding: '4px 8px',
                                                                fontSize: '0.9em',
                                                                flex: 1
                                                            }}
                                                            placeholder="Profile name"
                                                        />
                                                    </div>

                                                    {/* Number of Slots */}
                                                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                                        <label style={{ color: '#aaa', fontSize: '0.9em', minWidth: '80px' }}>
                                                            Slots:
                                                        </label>
                                                        <input
                                                            type="number"
                                                            min="1"
                                                            max="20"
                                                            value={profile.slots.length}
                                                            onChange={(e) => handleNumSlotsChange(profile.groupId, e.target.value)}
                                                            style={{
                                                                background: '#1a1a1a',
                                                                border: '1px solid #555',
                                                                borderRadius: '3px',
                                                                color: '#ccc',
                                                                padding: '4px 8px',
                                                                fontSize: '0.9em',
                                                                width: '80px'
                                                            }}
                                                        />
                                                        <span style={{ color: '#888', fontSize: '0.85em' }}>
                                                            (1-20)
                                                        </span>
                                                    </div>
                                                </div>

                                                {/* Slot Reordering Section */}
                                                <div style={{
                                                    borderTop: '1px solid #333',
                                                    paddingTop: '8px',
                                                    marginBottom: '12px'
                                                }}>
                                                    <div style={{
                                                        fontSize: '0.95em',
                                                        fontWeight: 'bold',
                                                        marginBottom: '8px',
                                                        color: '#ddd',
                                                        display: 'flex',
                                                        justifyContent: 'space-between',
                                                        alignItems: 'center'
                                                    }}>
                                                        <span>📋 Slot Order (Drag to reorder)</span>
                                                        <button
                                                            onClick={() => handleResetProfile(profile.groupId)}
                                                            title="Reset all disconnected slots to default"
                                                            style={{
                                                                background: 'none',
                                                                border: '1px solid #555',
                                                                borderRadius: '3px',
                                                                color: '#aaa',
                                                                cursor: 'pointer',
                                                                padding: '4px 8px',
                                                                fontSize: '0.9em',
                                                                display: 'flex',
                                                                alignItems: 'center',
                                                                gap: '4px',
                                                                fontWeight: 'normal'
                                                            }}
                                                            onMouseOver={(e) => {
                                                                e.currentTarget.style.background = '#333';
                                                                e.currentTarget.style.borderColor = '#777';
                                                                e.currentTarget.style.color = '#fff';
                                                            }}
                                                            onMouseOut={(e) => {
                                                                e.currentTarget.style.background = 'none';
                                                                e.currentTarget.style.borderColor = '#555';
                                                                e.currentTarget.style.color = '#aaa';
                                                            }}
                                                        >
                                                            🔄 Reset
                                                        </button>
                                                    </div>
                                                    {profile.slots && profile.slots.length > 0 ? (
                                                        profile.slots.map((slot, idx) => (
                                                            <div
                                                                key={slot.index}
                                                                draggable
                                                                onDragStart={(e) => handleDragStart(e, profile.groupId, idx)}
                                                                onDragOver={handleDragOver}
                                                                onDrop={(e) => handleDrop(e, profile.groupId, idx)}
                                                                style={{
                                                                    padding: '8px 10px',
                                                                    marginBottom: '4px',
                                                                    backgroundColor: draggedSlot?.slotIndex === idx && draggedSlot?.groupId === profile.groupId ? '#2a4a6a' : '#252525',
                                                                    border: '1px solid #444',
                                                                    borderRadius: '4px',
                                                                    cursor: 'move',
                                                                    display: 'flex',
                                                                    justifyContent: 'space-between',
                                                                    alignItems: 'center',
                                                                    fontSize: '0.9em',
                                                                    gap: '8px'
                                                                }}
                                                            >
                                                                <span style={{ display: 'flex', gap: '8px', alignItems: 'center', flex: 1 }}>
                                                                    <span style={{ color: '#888', cursor: 'grab' }}>☰</span>
                                                                    <input
                                                                        type="text"
                                                                        value={slot.label}
                                                                        onChange={(e) => handleLabelChange(profile.groupId, slot.index, e.target.value)}
                                                                        onClick={(e) => e.stopPropagation()}
                                                                        onMouseDown={(e) => e.stopPropagation()}
                                                                        placeholder={slot.type}
                                                                        style={{
                                                                            background: '#1a1a1a',
                                                                            border: '1px solid #555',
                                                                            borderRadius: '3px',
                                                                            color: '#ccc',
                                                                            padding: '4px 8px',
                                                                            fontSize: '0.9em',
                                                                            fontWeight: '500',
                                                                            flex: 1,
                                                                            minWidth: '100px',
                                                                            cursor: 'text'
                                                                        }}
                                                                        title="Edit label (type remains unchanged)"
                                                                    />
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
                                                            <span style={{ fontWeight: '500', flex: 1, fontFamily: 'monospace', color: '#6a9eff' }}>
                                                                #{node.id}
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
