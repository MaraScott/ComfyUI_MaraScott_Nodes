/*
 * Definitions for litegraph.js
 * Project: litegraph.js
 * Definitions by: NateScarlet <https://github.com/NateScarlet>
 * https://github.com/NateScarlet/litegraph.js/blob/master/src/litegraph.js
 * ComfyUI\web\lib\litegraph.core.js - Check for settings
 * ComfyUI\web\extensions\logging.js.example
 * ComfyUI\custom_nodes\rgthree-comfy\src_web\typings\litegraph.d.ts
 *
 */

import { app } from "../../scripts/app.js";

if (!window.marascott) {
    window.marascott = {}
}
if (!window.marascott.AnyBus_v2) {
    window.marascott.AnyBus_v2 = {
        init: false,
        sync: false,
        input: {
            label: "0",
            index: 0,
        },
        clean: false,
        nodeToSync: null,
        flows: {},
        nodes: {},
    }
}
window.marascott.AnyBus_v2.nodes = {}
window.marascott.AnyBus_v2.flows = {
    start: [],
    list: [],
    end: [],
}
class MaraScottAnyBusNodeWidget {
    static PROFILE = {
        name: 'Profile',
        default: 'default',
    }
    static INPUTS = {
        name: "Nb Inputs",
        default: 5,
        min: 3,
        max: 25,
    }
    static CLEAN = {
        default: false,
        name: 'Clean Inputs',
    }

    static init(node) {
        if (!node) {
            console.warn('AnyBus: Cannot initialize widget - node is null');
            return;
        }

        try {
            // Set up properties first
            this.setDefaults(node);
            
            // Initialize widgets in order
            this.setProfileInput(node);
            
            // Delay inputs initialization if graph isn't ready
            if (!node.graph) {
                setTimeout(() => {
                    if (node.graph) {
                        this.setInputsSelect(node);
                        this.setCleanSwitch(node);
                    }
                }, 100);
            } else {
                this.setInputsSelect(node);
                this.setCleanSwitch(node);
            }
        } catch (err) {
            console.error('AnyBus: Error initializing widgets:', err);
            this.setDefaults(node);
        }
    }

    static setDefaults(node) {
        try {
            node.properties = node.properties || {};
            node.properties[this.PROFILE.name] = this.PROFILE.default;
            node.properties[this.INPUTS.name] = this.INPUTS.default;
            node.properties[this.CLEAN.name] = this.CLEAN.default;
        } catch (err) {
            console.error('AnyBus: Failed to set defaults:', err);
        }
    }

    static getByName(node, name) {
        if (!node?.widgets) {
            return undefined;
        }
        return node.widgets.find((w) => w.name === name);
    }

    static setValueProfile(node, name, value) {
        if (!node) return;
        try {
            // Update node title with new profile name
            node.title = "AnyBus - " + value;
            node.setDirtyCanvas(true, true);
        } catch (err) {
            console.error('AnyBus: Error in setValueProfile:', err);
        }
    }

    static setValueInputs(node, name, value) {
        if (!node) return;
        
        try {
            const currentLength = node.inputs.length;
            const newValue = parseInt(value) + MaraScottAnyBus_v2.FIRST_INDEX;
            
            // Store current state for recovery
            const originalInputs = [...node.inputs];
            const originalOutputs = [...node.outputs];

            try {
                if (currentLength > newValue) {
                    // Remove excess inputs/outputs from highest index down
                    for (let i = currentLength - 1; i >= newValue; i--) {
                        if (node.graph) {
                            // Safely disconnect if graph is available
                            if (node.inputs[i]?.link !== null) {
                                node.disconnectInput(i);
                            }
                            if (node.outputs[i]?.links?.length > 0) {
                                node.disconnectOutput(i);
                            }
                        }
                        node.inputs.pop();
                        node.outputs.pop();
                    }
                } else if (currentLength < newValue) {
                    // Add new inputs/outputs
                    for (let i = currentLength; i < newValue; i++) {
                        const slotLabel = "* " + (i - MaraScottAnyBus_v2.FIRST_INDEX + 1).toString().padStart(2, '0');
                        
                        // Add new input
                        node.inputs.push({
                            name: slotLabel,
                            type: "*",
                            link: null,
                            slot_index: i
                        });
                        
                        // Add matching output
                        node.outputs.push({
                            name: slotLabel,
                            type: "*",
                            links: null,
                            slot_index: i
                        });
                    }
                }

                // Update visual state
                if (node.graph) {
                    node.setDirtyCanvas(true, true);
                    node.graph.computeExecutionOrder();
                }
            } catch (err) {
                console.error('AnyBus: Error modifying inputs/outputs:', err);
                // Restore original state
                node.inputs = originalInputs;
                node.outputs = originalOutputs;
                throw err;
            }
        } catch (err) {
            console.error('AnyBus: Error in setValueInputs:', err);
        }
    }

    static setValue(node, name, value) {
        if (!node || !name) {
            console.warn('AnyBus: Invalid node or name in setValue');
            return;
        }

        try {
            const nodeWidget = this.getByName(node, name);
            if (!nodeWidget) {
                console.warn(`AnyBus: Widget ${name} not found`);
                return;
            }

            nodeWidget.value = value;
            node.properties[name] = nodeWidget.value ?? node.properties[name];

            // Handle specific widget updates
            if (name === this.PROFILE.name) {
                this.setValueProfile(node, name, value);
            } else if (name === this.INPUTS.name) {
                this.setValueInputs(node, name, value);
            }

            node.setDirtyCanvas(true);
        } catch (err) {
            console.error('AnyBus: Error setting value:', err);
        }
    }

    static setProfileInput(node) {
        if (!node) return;

        try {
            const nodeWidget = this.getByName(node, this.PROFILE.name);
            if (nodeWidget === undefined) {
                node.addWidget(
                    "text",
                    this.PROFILE.name,
                    node.properties[this.PROFILE.name] ?? this.PROFILE.default,
                    (value) => {
                        try {
                            this.setValue(node, this.PROFILE.name, value);
                            window.marascott.AnyBus_v2.sync = MaraScottAnyBusNodeFlow.FULLSYNC;
                            MaraScottAnyBusNodeFlow.syncProfile(node, this.PROFILE.name, null);
                            node.setProperty('prevProfileName', node.properties[this.PROFILE.name]);
                        } catch (err) {
                            console.error('AnyBus: Error in profile callback:', err);
                        }
                    },
                    { title: "Profile name for this bus" }
                );
                this.setValue(node, this.PROFILE.name, this.PROFILE.default);
                node.setProperty('prevProfileName', node.properties[this.PROFILE.name]);
            }
        } catch (err) {
            console.error('AnyBus: Error setting profile input:', err);
            this.setDefaults(node);
        }
    }

    static setInputsSelect(node) {
        if (!node) return;

        try {
            const nodeWidget = this.getByName(node, this.INPUTS.name);
            if (nodeWidget === undefined) {
                const values = Array.from(
                    { length: this.INPUTS.max - this.INPUTS.min + 1 },
                    (_, i) => i + this.INPUTS.min
                );

                node.addWidget(
                    "combo",
                    this.INPUTS.name,
                    node.properties[this.INPUTS.name] ?? this.INPUTS.default,
                    (value) => {
                        try {
                            // Only modify inputs if graph is available
                            if (node.graph) {
                                this.setValue(node, this.INPUTS.name, value);
                                window.marascott.AnyBus_v2.sync = MaraScottAnyBusNodeFlow.FULLSYNC;
                                MaraScottAnyBusNodeFlow.syncProfile(node, this.INPUTS.name, null);
                            } else {
                                console.warn('AnyBus: Graph not ready, storing value for later');
                                node.properties[this.INPUTS.name] = value;
                            }
                        } catch (err) {
                            console.error('AnyBus: Error in inputs callback:', err);
                        }
                    },
                    { values, title: "Number of input/output pairs" }
                );

                // Set initial properties without modifying inputs if graph isn't ready
                node.properties[this.INPUTS.name] = this.INPUTS.default;
                if (node.graph) {
                    this.setValue(node, this.INPUTS.name, this.INPUTS.default);
                }
            }
        } catch (err) {
            console.error('AnyBus: Error setting inputs select:', err);
            this.setDefaults(node);
        }
    }

    static setCleanSwitch(node) {
        if (!node) return;

        try {
            const nodeWidget = this.getByName(node, this.CLEAN.name);
            if (nodeWidget === undefined) {
                node.addWidget(
                    "toggle",
                    this.CLEAN.name,
                    this.CLEAN.default,
                    (value) => {
                        try {
                            const flows = window.marascott.AnyBus_v2.flows.end;
                            for (const index in flows) {
                                const targetNode = node.graph.getNodeById(flows[index]);
                                if (targetNode) {
                                    MaraScottAnyBusNodeFlow.clean(targetNode);
                                }
                            }
                            this.setValue(node, this.CLEAN.name, this.CLEAN.default);
                        } catch (err) {
                            console.error('AnyBus: Error in clean callback:', err);
                        }
                    },
                    { title: "Clean and reset all connections" }
                );
                this.setValue(node, this.CLEAN.name, this.CLEAN.default);
            }
        } catch (err) {
            console.error('AnyBus: Error setting clean switch:', err);
            this.setDefaults(node);
        }
    }
}

class MaraScottAnyBus_v2 {

    static TYPE = "MaraScottAnyBus_v2"

    static BUS_SLOT = 0
    static BASIC_PIPE_SLOT = 0
    static REFINER_PIPE_SLOT = 0

    static FIRST_INDEX = 1

    static configure(node) {

        node.shape = LiteGraph.CARD_SHAPE // BOX_SHAPE | ROUND_SHAPE | CIRCLE_SHAPE | CARD_SHAPE
        node.color = LGraphCanvas.node_colors.green.color
        node.bgcolor = LGraphCanvas.node_colors.green.bgcolor
        node.groupcolor = LGraphCanvas.node_colors.green.groupcolor
        node.size[0] = 150 // width
        if (!node.properties || !(MaraScottAnyBusNodeWidget.PROFILE.name in node.properties)) {
            node.properties[MaraScottAnyBusNodeWidget.PROFILE.name] = MaraScottAnyBusNodeWidget.PROFILE.default;
        }
        node.title = "AnyBus - " + node.properties[MaraScottAnyBusNodeWidget.PROFILE.name]
    }

    static setWidgets(node) {
        MaraScottAnyBusNodeWidget.init(node)
    }

    static setInputValue(node) {

        let protected_slots = []

        let inputsLength = window.marascott.AnyBus_v2.nodeToSync.inputs.length
        if (node.inputs.length < inputsLength) inputsLength = node.inputs.length

        for (let slot = this.FIRST_INDEX; slot < inputsLength; slot++) {

            if (protected_slots.indexOf(slot) > -1) continue
            if (typeof node.inputs[slot] == 'undefined' || typeof window.marascott.AnyBus_v2.nodeToSync.inputs[slot] == 'undefined') {
                console.log('[MaraScott Nodes] Check your profile Names')
                continue;
            }

            const isNodeInputAny = node.inputs[slot].type == "*"
            const isNodeOutputDifferent = node.outputs[slot].type == window.marascott.AnyBus_v2.nodeToSync.outputs[slot].type
            const isNodeInputDifferent =
                !isNodeOutputDifferent // output different from new input
            const isOutputAny = node.outputs[slot].type == "*"
            const isOutputDifferent = node.outputs[slot].type != window.marascott.AnyBus_v2.nodeToSync.outputs[slot].type
            const isOutputLinked = node.outputs[slot].links != null && node.outputs[slot].links.length > 0

            if (isNodeInputDifferent) {
                const preSyncMode = window.marascott.AnyBus_v2.sync;
                window.marascott.AnyBus_v2.sync = this.NOSYNC;
                if (node.inputs[slot].link == null) {
                    node.disconnectInput(slot)
                    node.disconnectOutput(slot)
                } else {
                    protected_slots.push(node.id)
                }
                window.marascott.AnyBus_v2.sync = preSyncMode;
            }
            if (window.marascott.AnyBus_v2.nodeToSync != null && window.marascott.AnyBus_v2.nodeToSync.id != node.id) {
                if (node.inputs[slot].link == null) {
                    node.inputs[slot].name = window.marascott.AnyBus_v2.nodeToSync.inputs[slot].name.toLowerCase()
                    node.inputs[slot].type = window.marascott.AnyBus_v2.nodeToSync.inputs[slot].type
                    node.outputs[slot].name = node.inputs[slot].name
                    if (isOutputDifferent || !isOutputLinked) node.outputs[slot].type = node.inputs[slot].type
                    console.log(node.outputs[slot].name, node.inputs[slot].name)
                }
            }
        }

    }

    static getSyncType(node, slot, link_info_node, link_info_slot) {

        // on connect
        const isInBusLink = node.inputs[0].link != null
        const isOutBusLink = node.outputs[0]?.links?.length > 0

        const isInputLink = node.inputs[slot].link != null
        const isOutputLink = node.outputs[slot]?.links?.length > 0

        const isInputAny = node.inputs[slot].type == "*" && node.outputs[slot].type == "*"
        const isInputSameType = node.inputs[slot].type == link_info_node?.outputs[link_info_slot].type
        const isInputSameName = node.inputs[slot].name.toLowerCase() == link_info_node?.outputs[link_info_slot].name.toLowerCase()

        const isFull = (link_info_node != null ? isInputAny : isInputLink) && isInBusLink && isOutBusLink
        const isBackward = !isFull && !isOutBusLink && isInBusLink
        const isForward = !isBackward && isOutBusLink

        let syncType = MaraScottAnyBusNodeFlow.NOSYNC
        if (isForward || isBackward || isFull) syncType = MaraScottAnyBusNodeFlow.FULLSYNC

        return syncType

    }

    static getBusParentNodeWithInput(node, slot) {
        if (!node || !node.inputs || !Array.isArray(node.inputs)) {
            console.warn('AnyBus: Invalid node or inputs in getBusParentNodeWithInput');
            return null;
        }

        let parentNode = null;
        try {
            const busInput = node.inputs[MaraScottAnyBus_v2.BUS_SLOT];
            
            if (busInput && busInput.link != null) {
                // Get graph links in a safe way
                const nodeGraphLinks = node.graph?.links ? 
                    (Array.isArray(node.graph.links) ? node.graph.links : Object.values(node.graph.links)) 
                    : [];

                const parentLink = nodeGraphLinks.find(
                    (otherLink) => otherLink && otherLink.id === busInput.link
                );

                if (parentLink) {
                    parentNode = node.graph.getNodeById(parentLink.origin_id);

                    // Handle reroute and get/set nodes recursively
                    if (parentNode) {
                        if (MaraScottAnyBusNodeFlow.ALLOWED_REROUTE_TYPE.indexOf(parentNode.type) > -1) {
                            parentNode = this.getBusParentNodeWithInput(parentNode, slot);
                        }
                        if (MaraScottAnyBusNodeFlow.ALLOWED_GETSET_TYPE.indexOf(parentNode.type) > -1) {
                            parentNode = this.getBusParentNodeWithInput(parentNode, slot);
                        }
                        if (parentNode && (!parentNode.inputs[slot] || parentNode.inputs[slot].link == null)) {
                            parentNode = this.getBusParentNodeWithInput(parentNode, slot);
                        }
                    }
                }
            }

            // Update node inputs/outputs if we have a valid parent
            if (parentNode && node.inputs[slot] && 
                MaraScottAnyBusNodeFlow.ALLOWED_REROUTE_TYPE.indexOf(parentNode.type) === -1 && 
                MaraScottAnyBusNodeFlow.ALLOWED_GETSET_TYPE.indexOf(parentNode.type) === -1) {
                
                if (parentNode.inputs[slot]) {
                    node.inputs[slot].name = parentNode.inputs[slot].name;
                    node.inputs[slot].type = parentNode.inputs[slot].type;
                } else {
                    node.inputs[slot].name = "* " + window.marascott.AnyBus_v2.input.index.toString().padStart(2, '0');
                    node.inputs[slot].type = "*";
                }

                if (node.outputs[slot]) {
                    node.outputs[slot].name = node.inputs[slot].name;
                    node.outputs[slot].type = node.inputs[slot].type;
                }
            }

        } catch (err) {
            console.error('AnyBus: Error in getBusParentNodeWithInput:', err);
            return null;
        }

        return parentNode;
    }

    static disConnectBus(node, slot) {

        return MaraScottAnyBusNodeFlow.FULLSYNC

    }

    static disConnectInput(node, slot) {

        const syncProfile = this.getSyncType(node, slot, null, null)
        const previousBusNode = this.getBusParentNodeWithInput(node, slot)
        let busNodes = []
        const busNodePaths = MaraScottAnyBusNodeFlow.getFlows(node)

        let newName = "* " + window.marascott.AnyBus_v2.input.index.toString().padStart(2, '0')
        let newType = "*"
        let _node = null
        for (let i in busNodePaths) {
            busNodes = busNodePaths[i]
            busNodes.reverse()
            for (let y in busNodes) {
                _node = node.graph.getNodeById(busNodes[y])
                if (_node != null) {
                    if (typeof _node.inputs[slot] != 'undefined' && _node.inputs[slot].link != null && _node.inputs[slot].type != "*") {
                        newName = _node.inputs[slot].name
                        newType = _node.inputs[slot].type
                    } else if (typeof _node.inputs[slot] != 'undefined' && _node.inputs[slot].type == newType && _node.inputs[slot].name != newName) {
                        newName = _node.inputs[slot].name
                    }
                }
            }
        }
        // input
        node.inputs[slot].name = newName
        node.inputs[slot].type = newType
        node.outputs[slot].name = node.inputs[slot].name
        // node.outputs[slot].type = node.inputs[slot].type

        return syncProfile

    }

    static connectBus(node, slot, node_origin, origin_slot) {

        const syncProfile = MaraScottAnyBusNodeFlow.FULLSYNC
        const isBusInput = slot == MaraScottAnyBus_v2.BUS_SLOT
        const isOutputs = node_origin.outputs?.length > 0
        let isMaraScottBusNode = node_origin.type == MaraScottAnyBus_v2.TYPE
        if (!isMaraScottBusNode) {
            const origin_reroute_node = MaraScottAnyBusNodeFlow.getOriginRerouteBusType(node_origin)
            isMaraScottBusNode = origin_reroute_node?.type == MaraScottAnyBus_v2.TYPE
            if (isMaraScottBusNode) {
                node_origin = origin_reroute_node
            }
        }
        const isOriginProfileSame = node.properties[MaraScottAnyBusNodeWidget.PROFILE.name] == node_origin.properties[MaraScottAnyBusNodeWidget.PROFILE.name]
        const isTargetProfileDefault = node.properties[MaraScottAnyBusNodeWidget.PROFILE.name] == MaraScottAnyBusNodeWidget.PROFILE.default
        const isOriginSlotBus = origin_slot == MaraScottAnyBus_v2.BUS_SLOT
        if (isBusInput && isOriginSlotBus && isOutputs && isMaraScottBusNode && (isOriginProfileSame || isTargetProfileDefault)) {
            if (isTargetProfileDefault) {
                MaraScottAnyBusNodeWidget.setValue(node, MaraScottAnyBusNodeWidget.PROFILE.name, node_origin.properties[MaraScottAnyBusNodeWidget.PROFILE.name])
                node.setProperty('prevProfileName', node.properties[MaraScottAnyBusNodeWidget.PROFILE.name])
            }

            MaraScottAnyBusNodeWidget.setValue(node, MaraScottAnyBusNodeWidget.INPUTS.name, node_origin.properties[MaraScottAnyBusNodeWidget.INPUTS.name])
            for (let _slot = MaraScottAnyBus_v2.FIRST_INDEX; _slot < node_origin.outputs.length; _slot++) {
                if (_slot > node_origin.properties[MaraScottAnyBusNodeWidget.INPUTS.name]) {
                    node.disconnectInput(_slot)
                    node.disconnectOutput(_slot)
                } else {
                    if (node_origin.outputs[_slot].type != node.inputs[_slot].type) {
                        node.disconnectInput(_slot)
                        node.disconnectOutput(_slot)
                    }
                    node.inputs[_slot].name = node_origin.outputs[_slot].name.toLowerCase()
                    node.inputs[_slot].type = node_origin.outputs[_slot].type
                    node.outputs[_slot].name = node.inputs[_slot].name
                    node.outputs[_slot].type = node.inputs[_slot].type
                }
            }
        } else {
            node.disconnectInput(slot)
        }
        return syncProfile
    }

    static connectInput(node, slot, node_origin, origin_slot) {
        let syncProfile = MaraScottAnyBusNodeFlow.NOSYNC;
        try {
            // Validate nodes and slots
            if (!node || !node_origin) {
                console.warn('AnyBus: Invalid nodes in connectInput');
                return syncProfile;
            }

            const isOriginAnyBusBus = node_origin.type === MaraScottAnyBus_v2.TYPE;
            const isOriginSlotBus = origin_slot === MaraScottAnyBus_v2.BUS_SLOT;

            // Don't handle bus-to-bus connections here
            if (isOriginAnyBusBus && isOriginSlotBus) {
                node.disconnectInput(slot);
                return syncProfile;
            }

            // Get origin output and validate
            const originOutput = node_origin.outputs[origin_slot];
            if (!originOutput) {
                console.warn('AnyBus: Invalid origin output');
                return syncProfile;
            }

            // Update input/output labels and types
            const originLabel = originOutput.name?.toLowerCase() || '';
            const originType = originOutput.type || '*';
            
            // Ensure input exists and can be updated
            if (node.inputs[slot]) {
                // Force update the input
                node.inputs[slot].name = originLabel || `* ${slot.toString().padStart(2, '0')}`;
                node.inputs[slot].type = originType;

                // Update matching output
                if (node.outputs[slot]) {
                    node.outputs[slot].name = node.inputs[slot].name;
                    node.outputs[slot].type = originType;
                }

                // Get sync type and propagate changes
                syncProfile = this.getSyncType(node, slot, node_origin, origin_slot);
                this.propagateLabelsViaBus(node, slot, node.inputs[slot].name, originType);
            }

            // Force canvas refresh
            if (node.graph) {
                node.setDirtyCanvas(true, true);
            }
        } catch (err) {
            console.error('AnyBus: Error in connectInput:', err);
        }

        return syncProfile;
    }

    static propagateLabelsViaBus(sourceNode, slot, label, type) {
        if (!sourceNode?.graph || !sourceNode.outputs || !sourceNode.outputs[0]?.links) {
            return;
        }

        try {
            // Get all nodes connected to bus output
            const connectedBusNodes = [];
            const busLinks = sourceNode.outputs[0].links;
            
            for (const linkId of busLinks) {
                const link = sourceNode.graph.links[linkId];
                if (!link) continue;

                const targetNode = sourceNode.graph.getNodeById(link.target_id);
                if (targetNode?.type === MaraScottAnyBus_v2.TYPE) {
                    connectedBusNodes.push(targetNode);
                }
            }

            // Update labels on all connected bus nodes
            for (const targetNode of connectedBusNodes) {
                if (targetNode.inputs[slot] && targetNode.outputs[slot]) {
                    targetNode.inputs[slot].name = label;
                    targetNode.inputs[slot].type = type;
                    targetNode.outputs[slot].name = label;
                    targetNode.outputs[slot].type = type;
                    
                    // Recursive propagation
                    this.propagateLabelsViaBus(targetNode, slot, label, type);
                }
            }
        } catch (err) {
            console.error('AnyBus: Error propagating labels:', err);
        }
    }

    static clean(node) {
        if (!node || !node.graph) return;

        try {
            window.marascott.AnyBus_v2.clean = false;
            
            // Handle bus connection cleaning
            if (node.inputs[MaraScottAnyBus_v2.BUS_SLOT]?.link != null) {
                const nodeGraphLinks = node.graph.links;
                if (!nodeGraphLinks) return;

                const busLink = Array.isArray(nodeGraphLinks) ? 
                    nodeGraphLinks.find(link => link?.id === node.inputs[MaraScottAnyBus_v2.BUS_SLOT].link) :
                    nodeGraphLinks[node.inputs[MaraScottAnyBus_v2.BUS_SLOT].link];

                if (busLink) {
                    const originNode = node.graph.getNodeById(busLink.origin_id);
                    if (originNode) {
                        // Disconnect and reconnect bus
                        node.disconnectInput(MaraScottAnyBus_v2.BUS_SLOT);
                        originNode.connect(MaraScottAnyBus_v2.BUS_SLOT, node, MaraScottAnyBus_v2.BUS_SLOT);
                    }
                }
            } else {
                // Clean all regular inputs
                for (let slot = MaraScottAnyBus_v2.FIRST_INDEX; slot < node.inputs.length; slot++) {
                    const input = node.inputs[slot];
                    if (!input) continue;

                    if (input.link != null) {
                        // Get original connection info
                        const nodeGraphLinks = node.graph.links;
                        const link = Array.isArray(nodeGraphLinks) ?
                            nodeGraphLinks.find(l => l?.id === input.link) :
                            nodeGraphLinks[input.link];

                        if (link) {
                            const originNode = node.graph.getNodeById(link.origin_id);
                            if (originNode) {
                                // Disconnect and reconnect to preserve connection
                                node.disconnectInput(slot);
                                originNode.connect(link.origin_slot, node, slot);
                            }
                        }
                    } else {
                        // Reset unconnected input/output
                        const defaultLabel = "* " + (slot + 1 - MaraScottAnyBus_v2.FIRST_INDEX).toString().padStart(2, '0');
                        node.inputs[slot].name = defaultLabel;
                        node.inputs[slot].type = "*";
                        node.outputs[slot].name = defaultLabel;
                        node.outputs[slot].type = "*";
                    }
                }

                // Trigger sync and update
                window.marascott.AnyBus_v2.sync = MaraScottAnyBusNodeFlow.FULLSYNC;
                MaraScottAnyBusNodeFlow.syncProfile(node, MaraScottAnyBusNodeWidget.CLEAN.name, false);
                MaraScottAnyBusNodeFlow.syncProfile(node, null, true);
                window.marascott.AnyBus_v2.clean = true;
            }

            // Visual feedback
            const cleanedLabel = " ... cleaned";
            node.title = node.title + cleanedLabel;
            node.setDirtyCanvas(true, true);  // Force immediate refresh

            // Remove cleaned label after delay
            setTimeout(() => {
                node.title = node.title.replace(cleanedLabel, "");
                node.setDirtyCanvas(true, true);  // Refresh again after removing label
            }, 500);

        } catch (err) {
            console.error('AnyBus: Error in clean:', err);
        }
    }

}

class MaraScottAnyBusNodeFlow {

    static NOSYNC = 0
    static FULLSYNC = 1

    static ALLOWED_REROUTE_TYPE = [
        "Reroute (rgthree)", // SUPPORTED - RgThree Custom Node
        // "Reroute", // UNSUPPORTED - ComfyUI native - do not allow connection on Any Type if origin Type is not Any Type too
        // "ReroutePrimitive|pysssss", // UNSUPPORTED - Pysssss Custom Node - do not display the name of the origin slot
        // "0246.CastReroute", //  UNSUPPORTED - 0246 Custom Node
    ]
    static ALLOWED_GETSET_TYPE = [
        "SetNode", // SUPPORTED - ComfyUI-KJNodes Custom Node
        "GetNode", // SUPPORTED - ComfyUI-KJNodes Custom Node
    ]
    static ALLOWED_NODE_TYPE = [
        MaraScottAnyBus_v2.TYPE,
        ...this.ALLOWED_REROUTE_TYPE,
        ...this.ALLOWED_GETSET_TYPE,
    ]

    static getLastBuses(nodes) {
        // Find all parent nodes
        let parents = Object.values(nodes);
        const leafs = Object.keys(nodes);
        let leafSet = new Set(leafs);

        let lastLeaves = leafs.filter(leaf => {
            // Check if the leaf is not a parent to any other node
            return !parents.includes(parseInt(leaf));
        }).map(leaf => parseInt(leaf));

        return lastLeaves;
    }

    static getOriginRerouteBusType(node) {
        let originNode = null
        let _originNode = null
        let isMaraScottAnyBusNode = false

        if (node && node.inputs && node.inputs[0].link != null) {

            if (node.inputs[0].link == 'setNode') {

                _originNode = node.graph.getNodeById(node.inputs[0].origin_id)

            } else {

                const nodeGraphLinks = Array.isArray(node.graph.links)
                    ? node.graph.links
                    : Object.values(node.graph.links);

                const __originLink = nodeGraphLinks.find(
                    (otherLink) => otherLink?.id == node.inputs[0].link
                )
                _originNode = node.graph.getNodeById(__originLink.origin_id)

            }

            if (this.ALLOWED_REROUTE_TYPE.indexOf(_originNode.type) > -1 && _originNode?.__outputType == 'BUS') {
                _originNode = this.getOriginRerouteBusType(_originNode)
            }

            if (this.ALLOWED_GETSET_TYPE.indexOf(_originNode.type) > -1) {
                _originNode = this.getOriginRerouteBusType(_originNode)
            }

            if (_originNode?.type == MaraScottAnyBus_v2.TYPE) {
                originNode = _originNode
            }

        }

        return originNode

    }

    static getFlows(node) {

        let firstItem = null;
        for (let list of window.marascott.AnyBus_v2.flows.list) {
            if (list.includes(node.id)) {
                firstItem = list[0];
                break;
            }
        }

        return window.marascott.AnyBus_v2.flows.list.filter(list => list[0] === firstItem);

    }

    static setFlows(node) {

        let _nodes = []
        let _nodes_list = []
        for (let i in node.graph._nodes) {
            let _node = node.graph._nodes[i]
            if (this.ALLOWED_NODE_TYPE.includes(_node.type) && _nodes_list.indexOf(_node.id) == -1) {
                _nodes_list.push(_node.id);
                if (_node.type == 'GetNode') {
                    const _setnode = _node.findSetter(_node.graph)
                    if (_setnode) _node.inputs = [{ 'link': 'setNode', 'origin_id': _setnode.id }]
                }
                // if (_node.inputs[0].link != null) _nodes.push(_node);
            }
        }

        // bus network
        let _bus_node_link = null
        let _bus_nodes = []
        let _bus_nodes_connections = {}
        let node_paths_start = []
        let node_paths = []
        let node_paths_end = []

        for (let i in _nodes_list) {
            let _node = node.graph.getNodeById(_nodes_list[i])
            _bus_nodes.push(_node)
            window.marascott.AnyBus_v2.nodes[_node.id] = _node
        }
        for (let i in _bus_nodes) {
            _bus_node_link = _bus_nodes[i].inputs[0].link
            if (_bus_node_link == 'setNode') {
                if (_bus_nodes[i].inputs[0].origin_id) _bus_nodes_connections[_bus_nodes[i].id] = _bus_nodes[i].inputs[0].origin_id
            } else if (_bus_node_link != null) {

                const nodeGraphLinks = Array.isArray(node.graph.links)
                    ? node.graph.links
                    : Object.values(node.graph.links);

                _bus_node_link = nodeGraphLinks.find(
                    (otherLink) => otherLink?.id == _bus_node_link
                )
                if (_bus_node_link) _bus_nodes_connections[_bus_nodes[i].id] = _bus_node_link.origin_id
            } else {
                node_paths_start.push(_bus_nodes[i].id)
            }
        }

        node_paths_end = this.getLastBuses(_bus_nodes_connections)

        for (let id in node_paths_end) {
            let currentNode = node_paths_end[id]
            node_paths[id] = [currentNode]; // Initialize the path with the starting node
            while (_bus_nodes_connections[currentNode] !== undefined) {
                currentNode = _bus_nodes_connections[currentNode]; // Move to the parent node
                const _currentNode = node.graph.getNodeById(currentNode)
                if (_currentNode.type == MaraScottAnyBus_v2.TYPE) {
                    node_paths[id].push(currentNode); // Add the parent node to the path
                }
            }
            node_paths[id].reverse()
        }

        window.marascott.AnyBus_v2.flows.start = node_paths_start
        window.marascott.AnyBus_v2.flows.end = node_paths_end
        window.marascott.AnyBus_v2.flows.list = node_paths

    }

    static syncProfile(node, isChangeWidget, isChangeConnect) {

        if (!node.graph || window.marascott.AnyBus_v2.sync == MaraScottAnyBusNodeFlow.NOSYNC) return
        if (window.marascott.AnyBus_v2.sync == MaraScottAnyBusNodeFlow.FULLSYNC) {
            MaraScottAnyBusNodeFlow.setFlows(node);
            const busNodes = [].concat(...this.getFlows(node)).filter((value, index, self) => self.indexOf(value) === index)
            this.sync(node, busNodes, isChangeWidget, isChangeConnect)
        }

        window.marascott.AnyBus_v2.sync = MaraScottAnyBusNodeFlow.NOSYNC
    }

    static sync(node, busNodes, isChangeWidget, isChangeConnect) {

        window.marascott.AnyBus_v2.nodeToSync = node;

        let _node = null
        for (let i in busNodes) {
            _node = node.graph.getNodeById(busNodes[i])
            if (_node.id !== window.marascott.AnyBus_v2.nodeToSync.id && this.ALLOWED_REROUTE_TYPE.indexOf(_node.type) == -1 && this.ALLOWED_GETSET_TYPE.indexOf(_node.type) == -1) {
                if (isChangeWidget != null) {
                    MaraScottAnyBusNodeWidget.setValue(_node, isChangeWidget, window.marascott.AnyBus_v2.nodeToSync.properties[isChangeWidget])
                    if (isChangeWidget == MaraScottAnyBusNodeWidget.PROFILE.name) _node.setProperty('prevProfileName', window.marascott.AnyBus_v2.nodeToSync.properties[MaraScottAnyBusNodeWidget.PROFILE.name])
                }
                if (isChangeConnect !== null) {
                    MaraScottAnyBusNodeWidget.setValue(_node, MaraScottAnyBusNodeWidget.INPUTS.name, window.marascott.AnyBus_v2.nodeToSync.properties[MaraScottAnyBusNodeWidget.INPUTS.name])
                    MaraScottAnyBus_v2.setInputValue(_node)
                }
            }
        }

        window.marascott.AnyBus_v2.nodeToSync = null;

    }


    static clean(node) {
        if (!node?.graph) return;

        try {
            window.marascott.AnyBus_v2.clean = false;
            let didClean = false;

            // Handle bus connection cleaning
            if (node.inputs[MaraScottAnyBus_v2.BUS_SLOT]?.link != null) {
                this.cleanBusConnection(node);
                didClean = true;
            } else {
                // Clean all regular inputs
                for (let slot = MaraScottAnyBus_v2.FIRST_INDEX; slot < node.inputs.length; slot++) {
                    if (this.cleanInputSlot(node, slot)) {
                        didClean = true;
                    }
                }
            }

            if (didClean) {
                // Trigger sync and refresh
                window.marascott.AnyBus_v2.sync = MaraScottAnyBusNodeFlow.FULLSYNC;
                this.syncProfile(node, MaraScottAnyBusNodeWidget.CLEAN.name, false);
                this.syncProfile(node, null, true);
                window.marascott.AnyBus_v2.clean = true;

                // Visual feedback
                this.showCleanedFeedback(node);
            }

        } catch (err) {
            console.error('AnyBus: Error in clean:', err);
        }
    }

    static cleanBusConnection(node) {
        const nodeGraphLinks = node.graph.links;
        if (!nodeGraphLinks) return false;

        const busInput = node.inputs[MaraScottAnyBus_v2.BUS_SLOT];
        const busLink = Array.isArray(nodeGraphLinks) ?
            nodeGraphLinks.find(link => link?.id === busInput.link) :
            nodeGraphLinks[busInput.link];

        if (busLink) {
            const originNode = node.graph.getNodeById(busLink.origin_id);
            if (originNode) {
                // Disconnect and reconnect bus
                node.disconnectInput(MaraScottAnyBus_v2.BUS_SLOT);
                originNode.connect(MaraScottAnyBus_v2.BUS_SLOT, node, MaraScottAnyBus_v2.BUS_SLOT);
                return true;
            }
        }
        return false;
    }

    static cleanInputSlot(node, slot) {
        const input = node.inputs[slot];
        if (!input) return false;

        if (input.link != null) {
            // Get original connection info
            const nodeGraphLinks = node.graph.links;
            const link = Array.isArray(nodeGraphLinks) ?
                nodeGraphLinks.find(l => l?.id === input.link) :
                nodeGraphLinks[input.link];

            if (link) {
                const originNode = node.graph.getNodeById(link.origin_id);
                if (originNode) {
                    // Store original connection info
                    const originSlot = link.origin_slot;
                    const originType = originNode.outputs[originSlot].type;
                    const originLabel = originNode.outputs[originSlot].name;

                    // Disconnect and reconnect
                    node.disconnectInput(slot);
                    originNode.connect(originSlot, node, slot);

                    // Force update labels and types
                    if (node.inputs[slot] && node.outputs[slot]) {
                        node.inputs[slot].name = originLabel;
                        node.inputs[slot].type = originType;
                        node.outputs[slot].name = originLabel;
                        node.outputs[slot].type = originType;
                    }
                    return true;
                }
            }
        } else {
            // Reset unconnected input/output with proper label format
            const defaultLabel = "* " + (slot - MaraScottAnyBus_v2.FIRST_INDEX + 1).toString().padStart(2, '0');
            if (node.inputs[slot]) {
                node.inputs[slot].name = defaultLabel;
                node.inputs[slot].type = "*";
            }
            if (node.outputs[slot]) {
                node.outputs[slot].name = defaultLabel;
                node.outputs[slot].type = "*";
            }
            return true;
        }
        return false;
    }

    static showCleanedFeedback(node) {
        const cleanedLabel = " ... cleaned";
        node.title = node.title + cleanedLabel;
        node.setDirtyCanvas(true, true);  // Force immediate refresh

        // Remove cleaned label after delay
        setTimeout(() => {
            if (node?.title) {
                node.title = node.title.replace(cleanedLabel, "");
                node.setDirtyCanvas(true, true);
            }
        }, 500);
    }
}

class MaraScottAnyBusNodeLiteGraph {

    static onExecuted(nodeType) {
        const onExecuted = nodeType.prototype.onExecuted
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments)
            // console.log("[MaraScott - logging " + this.name + "]", "on Executed", { "id": this.id, "properties": this.properties });
        }

    }

    static onNodeCreated(nodeType) {

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            // console.log("[MaraScott - logging " + this.name + "]", 'onNodeCreated')
            MaraScottAnyBus_v2.configure(this)
            MaraScottAnyBus_v2.setWidgets(this)

            return r;
        }

    }

    static getExtraMenuOptions(nodeType) {
        const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (_, options) {

            // console.log("[MaraScott - logging " + this.name + "]", "on Extra Menu Options", { "id": this.id, "properties": this.properties });

        }
    }

    static onConnectionsChange(nodeType) {
        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function(slotType, slot, isChangeConnect, link_info, output) {
            const r = onConnectionsChange ? onConnectionsChange.apply(this, arguments) : undefined;

            if (!window.marascott.AnyBus_v2.init || !this?.graph) return r;

            window.marascott.AnyBus_v2.sync = MaraScottAnyBusNodeFlow.NOSYNC;
            window.marascott.AnyBus_v2.input.index = slot + 1 - MaraScottAnyBus_v2.FIRST_INDEX;

            try {
                //On Disconnect
                if (!isChangeConnect && slotType == 1 && typeof link_info != 'undefined') {
                    // Validate inputs array and slot before accessing
                    if (!this.inputs || !Array.isArray(this.inputs) || slot >= this.inputs.length) {
                        console.warn('AnyBus: Invalid inputs array or slot in onConnectionsChange');
                        return r;
                    }

                    if (slot < MaraScottAnyBus_v2.FIRST_INDEX) {
                        // bus
                        if (slot == 0 && this.inputs) {
                            window.marascott.AnyBus_v2.sync = MaraScottAnyBus_v2.disConnectBus(this);
                        }
                    } else {
                        window.marascott.AnyBus_v2.sync = MaraScottAnyBus_v2.disConnectInput(this, slot);
                    }
                }

                //On Connect
                if (isChangeConnect && slotType == 1 && typeof link_info != 'undefined') {
                    // Validate link_info and graph before proceeding
                    if (!link_info?.origin_id || !this.graph?._nodes) {
                        console.warn('AnyBus: Invalid link_info or graph in onConnectionsChange');
                        return r;
                    }

                    const link_info_node = this.graph._nodes.find(
                        (otherNode) => otherNode?.id === link_info.origin_id
                    );

                    if (!link_info_node) {
                        console.warn('AnyBus: Could not find origin node in onConnectionsChange');
                        return r;
                    }

                    if (slot < MaraScottAnyBus_v2.FIRST_INDEX) {
                        window.marascott.AnyBus_v2.sync = MaraScottAnyBus_v2.connectBus(this, slot, link_info_node, link_info.origin_slot);
                    } else {
                        window.marascott.AnyBus_v2.sync = MaraScottAnyBus_v2.connectInput(this, slot, link_info_node, link_info.origin_slot);
                    }
                }

                if (this.graph) {
                    MaraScottAnyBusNodeFlow.syncProfile(this, null, isChangeConnect);
                }
            } catch (err) {
                console.error('AnyBus: Error in onConnectionsChange:', err);
            }

            return r;
        };
    }

    static onRemoved(nodeType) {
        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            onRemoved?.apply(this, arguments);
            // console.log('onRemoved')
        };
    }


}

// Improved extension with proper hooks and initialization
const MaraScottAnyBusNodeExtension = {
    name: "Comfy.MaraScott.AnyBus_v2",
    
    // Extension state
    settings: [
        {
            id: "marascott.anybus.defaultProfile",
            name: "Default Profile",
            type: "text",
            defaultValue: "default",
            category: ["MaraScott", "AnyBus", "General"],
            tooltip: "Default profile name for new AnyBus nodes"
        },
        {
            id: "marascott.anybus.defaultInputs",
            name: "Default Input Count",
            type: "slider",
            defaultValue: 5,
            attrs: {
                min: 3,
                max: 25,
                step: 1
            },
            category: ["MaraScott", "AnyBus", "General"],
            tooltip: "Default number of inputs for new AnyBus nodes"
        }
    ],

    async init(app) {
        // Initialize extension state
        if (!window.marascott) {
            window.marascott = {};
        }
        window.marascott.AnyBus_v2 = {
            init: false,
            sync: false,
            input: { label: "0", index: 0 },
            clean: false,
            nodeToSync: null,
            flows: { start: [], list: [], end: [] },
            nodes: {}
        };
    },

    async setup(app) {
        // Add cleanup handler when window unloads
        window.addEventListener('unload', () => {
            window.marascott.AnyBus_v2 = null;
        });
    },

    // Called before any nodes are registered
    async beforeConfigureGraph(app) {
        window.marascott.AnyBus_v2.init = false;
    },

    // Called after all nodes are registered
    async afterConfigureGraph(app) {
        window.marascott.AnyBus_v2.init = true;
    },

    // Handle node definition registration
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === MaraScottAnyBus_v2.TYPE) {
            // Register node behaviors
            MaraScottAnyBusNodeLiteGraph.onNodeCreated(nodeType);
            MaraScottAnyBusNodeLiteGraph.onConnectionsChange(nodeType);
            MaraScottAnyBusNodeLiteGraph.onRemoved(nodeType);

            // Add right-click menu options
            const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
            nodeType.prototype.getExtraMenuOptions = function(_, options) {
                getExtraMenuOptions?.apply(this, arguments);
                
                options.push(null); // separator
                options.push({
                    content: "Clean All Connections",
                    callback: () => {
                        const widget = MaraScottAnyBusNodeWidget.getByName(this, MaraScottAnyBusNodeWidget.CLEAN.name);
                        if (widget) {
                            widget.value = true;
                            this.graph.setDirtyCanvas(true);
                        }
                    }
                });
            };
        }
    },

    // Handle each node instance creation
    async nodeCreated(node) {
        if (node.type === MaraScottAnyBus_v2.TYPE) {
            // Initialize with settings
            const defaultProfile = app.extensionManager.setting.get('marascott.anybus.defaultProfile');
            const defaultInputs = app.extensionManager.setting.get('marascott.anybus.defaultInputs');
            
            if (defaultProfile) {
                node.properties[MaraScottAnyBusNodeWidget.PROFILE.name] = defaultProfile;
            }
            if (defaultInputs) {
                node.properties[MaraScottAnyBusNodeWidget.INPUTS.name] = defaultInputs;
            }
            
            MaraScottAnyBus_v2.configure(node);
            MaraScottAnyBus_v2.setWidgets(node);
        }
    },

    // Handle workflow loading
    async loadedGraphNode(node, app) {
        if (node.type === MaraScottAnyBus_v2.TYPE) {
            node.setProperty('uuid', node.id);
            MaraScottAnyBusNodeFlow.setFlows(node);
        }
    }
};

// Register the extension
app.registerExtension(MaraScottAnyBusNodeExtension);
