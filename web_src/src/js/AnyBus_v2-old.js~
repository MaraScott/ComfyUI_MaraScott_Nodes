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
    static NAMESPACE = "MaraScott";
    static TYPE = "AnyBus_v2";
    static BUS_SLOT = 0;
    static FIRST_INDEX = 1;

    static ALLOWED_REROUTE_TYPE = [
        "Reroute (rgthree)", // SUPPORTED - RgThree Custom Node
    ];

    static ALLOWED_GETSET_TYPE = [
        "SetNode", // SUPPORTED - ComfyUI-KJNodes Custom Node
        "GetNode", // SUPPORTED - ComfyUI-KJNodes Custom Node
    ];

    static ALLOWED_NODE_TYPE = [
        "AnyBus_v2",
        ...MaraScottAnyBusNodeWidget.ALLOWED_REROUTE_TYPE,
        ...MaraScottAnyBusNodeWidget.ALLOWED_GETSET_TYPE,
    ];

    static PROFILE = {
        name: "profile",
        default: "default"
    };
    
    static INPUTS = {
        name: "inputs",
        min: 2,
        max: 24,
        default: 2
    };

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
        const oldValue = node.properties[name];
        node.properties[name] = value;

        if (name === this.PROFILE.name) {
            this.updateNodeTitle(node);
            node.properties['prevProfileName'] = value;
            
            // Only propagate profile if target nodes have matching or default profile
            const destNodes = this.findDestinationBusNodes(node, node.graph);
            for (const destNode of destNodes) {
                if (destNode.properties[this.PROFILE.name] === this.PROFILE.default ||
                    destNode.properties[this.PROFILE.name] === value) {
                    this.setValue(destNode, this.PROFILE.name, value);
                }
            }
        } 
        else if (name === this.INPUTS.name) {
            if (oldValue !== value) {
                this.updateNodeIO(node);
                // Propagate input count changes through flow
                const destNodes = this.findDestinationBusNodes(node, node.graph);
                for (const destNode of destNodes) {
                    this.setValue(destNode, this.INPUTS.name, value);
                }
            }
        }

        node.setDirtyCanvas(true);
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
        if (!node || !window.marascott.AnyBus_v2.nodeToSync) {
            console.warn('AnyBus: Invalid node or sync node');
            return;
        }
    
        let protected_slots = [];
    
        // Ensure both inputs and outputs arrays exist and are properly sized
        const targetLength = parseInt(node.properties["Nb Inputs"]) + this.FIRST_INDEX;
        
        // Initialize arrays if needed
        if (!Array.isArray(node.outputs)) node.outputs = [];
        if (!Array.isArray(node.inputs)) node.inputs = [];
    
        // Ensure arrays have correct length
        while (node.outputs.length < targetLength) {
            const idx = node.outputs.length;
            const label = "* " + (idx - this.FIRST_INDEX).toString().padStart(2, '0');
            node.outputs.push({
                name: label,
                type: "*",
                links: null
            });
        }
    
        let inputsLength = Math.min(
            window.marascott.AnyBus_v2.nodeToSync.inputs.length,
            targetLength
        );
    
        for (let slot = this.FIRST_INDEX; slot < inputsLength; slot++) {
            try {
                if (protected_slots.indexOf(slot) > -1) continue;
    
                // Validate slot existence
                if (!node.inputs[slot] || !node.outputs[slot] || 
                    !window.marascott.AnyBus_v2.nodeToSync.inputs[slot]) {
                    continue;
                }
    
                // Safe checks for type comparisons
                const isNodeInputAny = node.inputs[slot].type === "*";
                const isNodeOutputDifferent = node.outputs[slot].type === 
                    (window.marascott.AnyBus_v2.nodeToSync.outputs[slot]?.type || "*");
                const isNodeInputDifferent = !isNodeOutputDifferent;
                const isOutputAny = node.outputs[slot].type === "*";
                const isOutputDifferent = node.outputs[slot].type !== 
                    (window.marascott.AnyBus_v2.nodeToSync.outputs[slot]?.type || "*");
                const isOutputLinked = node.outputs[slot].links != null && 
                    node.outputs[slot].links.length > 0;
    
                if (isNodeInputDifferent) {
                    const preSyncMode = window.marascott.AnyBus_v2.sync;
                    window.marascott.AnyBus_v2.sync = this.NOSYNC;
                    if (node.inputs[slot].link == null) {
                        node.disconnectInput(slot);
                        node.disconnectOutput(slot);
                    } else {
                        protected_slots.push(node.id);
                    }
                    window.marascott.AnyBus_v2.sync = preSyncMode;
                }
    
                if (window.marascott.AnyBus_v2.nodeToSync?.id !== node.id) {
                    if (node.inputs[slot].link == null) {
                        const syncInput = window.marascott.AnyBus_v2.nodeToSync.inputs[slot];
                        if (syncInput) {
                            node.inputs[slot].name = syncInput.name.toLowerCase();
                            node.inputs[slot].type = syncInput.type;
                            node.outputs[slot].name = node.inputs[slot].name;
                            if (isOutputDifferent || !isOutputLinked) {
                                node.outputs[slot].type = node.inputs[slot].type;
                            }
                        }
                    }
                }
            } catch (err) {
                console.warn(`AnyBus: Error processing slot ${slot}:`, err);
            }
        }
    
        // Update canvas if needed
        if (node.graph) {
            node.setDirtyCanvas(true, true);
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
        // node.outputs[slot].type = node.inputs[slot].type // do not assign input type to keep old connections on other bus nodes

        return syncProfile

    }

    static connectBus(node, slot, node_origin, origin_slot) {
        const syncProfile = MaraScottAnyBusNodeFlow.FULLSYNC;
        const isBusInput = slot == MaraScottAnyBus_v2.BUS_SLOT;
        const isOutputs = node_origin.outputs?.length > 0;
        let isMaraScottBusNode = node_origin.type == MaraScottAnyBus_v2.TYPE;
        
        if (!isMaraScottBusNode) {
            const origin_reroute_node = MaraScottAnyBusNodeFlow.getOriginRerouteBusType(node_origin);
            isMaraScottBusNode = origin_reroute_node?.type == MaraScottAnyBus_v2.TYPE;
            if (isMaraScottBusNode) {
                node_origin = origin_reroute_node;
            }
        }
        
        const isOriginProfileSame = node.properties[MaraScottAnyBusNodeWidget.PROFILE.name] == node_origin.properties[MaraScottAnyBusNodeWidget.PROFILE.name];
        const isTargetProfileDefault = node.properties[MaraScottAnyBusNodeWidget.PROFILE.name] == MaraScottAnyBusNodeWidget.PROFILE.default;
        const isOriginSlotBus = origin_slot == MaraScottAnyBus_v2.BUS_SLOT;

        // Only allow connection if profiles match or target is default
        if (isBusInput && isOriginSlotBus && isOutputs && isMaraScottBusNode && (isOriginProfileSame || isTargetProfileDefault)) {
            if (isTargetProfileDefault) {
                MaraScottAnyBusNodeWidget.setValue(node, MaraScottAnyBusNodeWidget.PROFILE.name, node_origin.properties[MaraScottAnyBusNodeWidget.PROFILE.name]);
                node.setProperty('prevProfileName', node.properties[MaraScottAnyBusNodeWidget.PROFILE.name]);
            }

            // Sync input count and input/output configuration
            MaraScottAnyBusNodeWidget.setValue(node, MaraScottAnyBusNodeWidget.INPUTS.name, node_origin.properties[MaraScottAnyBusNodeWidget.INPUTS.name]);
            
            for (let _slot = MaraScottAnyBus_v2.FIRST_INDEX; _slot < node_origin.outputs.length; _slot++) {
                if (_slot > node_origin.properties[MaraScottAnyBusNodeWidget.INPUTS.name]) {
                    node.disconnectInput(_slot);
                    node.disconnectOutput(_slot);
                } else {
                    if (node_origin.outputs[_slot].type != node.inputs[_slot].type) {
                        node.disconnectInput(_slot);
                        node.disconnectOutput(_slot);
                    }
                    node.inputs[_slot].name = node_origin.outputs[_slot].name.toLowerCase();
                    node.inputs[_slot].type = node_origin.outputs[_slot].type;
                    node.outputs[_slot].name = node.inputs[_slot].name;
                    node.outputs[_slot].type = node.inputs[_slot].type;
                }
            }
        } else {
            node.disconnectInput(slot);
        }
        
        return syncProfile;
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

        for (let i in busNodes) {
            const targetNode = node.graph.getNodeById(busNodes[i]);
            if (!targetNode) continue;
            
            if (targetNode.id !== window.marascott.AnyBus_v2.nodeToSync.id && 
                !this.ALLOWED_REROUTE_TYPE.includes(targetNode.type) && 
                !this.ALLOWED_GETSET_TYPE.includes(targetNode.type)) {
                
                // Only sync profile if target is default or matches source
                if (isChangeWidget === MaraScottAnyBusNodeWidget.PROFILE.name) {
                    if (targetNode.properties[MaraScottAnyBusNodeWidget.PROFILE.name] === MaraScottAnyBusNodeWidget.PROFILE.default ||
                        targetNode.properties[MaraScottAnyBusNodeWidget.PROFILE.name] === window.marascott.AnyBus_v2.nodeToSync.properties[isChangeWidget]) {
                        MaraScottAnyBusNodeWidget.setValue(targetNode, isChangeWidget, window.marascott.AnyBus_v2.nodeToSync.properties[isChangeWidget]);
                        targetNode.setProperty('prevProfileName', window.marascott.AnyBus_v2.nodeToSync.properties[MaraScottAnyBusNodeWidget.PROFILE.name]);
                    }
                } 
                else if (isChangeWidget) {
                    MaraScottAnyBusNodeWidget.setValue(targetNode, isChangeWidget, window.marascott.AnyBus_v2.nodeToSync.properties[isChangeWidget]);
                }
                
                if (isChangeConnect !== null) {
                    MaraScottAnyBusNodeWidget.setValue(targetNode, MaraScottAnyBusNodeWidget.INPUTS.name, window.marascott.AnyBus_v2.nodeToSync.properties[MaraScottAnyBusNodeWidget.INPUTS.name]);
                    MaraScottAnyBus_v2.setInputValue(targetNode);
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

const MaraScottAnyBusNodeExtension = {
    name: "Comfy.MaraScott.AnyBus_v2",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeType.comfyClass === "MaraScottAnyBus_v2") {
            const onConnectionsChange = nodeType.prototype.onConnectionsChange;
            
            nodeType.prototype.onConnectionsChange = function(type, slotIndex, isConnected, link, ioSlot) {
                if (!window.marascott.AnyBus_v2.init || !this.graph) return;

                window.marascott.AnyBus_v2.sync = MaraScottAnyBusNodeFlow.NOSYNC;
                window.marascott.AnyBus_v2.input.index = slotIndex + 1 - MaraScottAnyBus_v2.FIRST_INDEX;

                // Handle new connections
                if (isConnected && link) {
                    const originNode = this.graph.getNodeById(link.origin_id);
                    if (originNode) {
                        if (slotIndex === MaraScottAnyBus_v2.BUS_SLOT) {
                            // Validate and handle BUS connections
                            const sourceBusNode = MaraScottAnyBusNodeWidget.findOriginBusNode(originNode, this.graph);
                            if (sourceBusNode) {
                                if (this.properties[MaraScottAnyBusNodeWidget.PROFILE.name] === MaraScottAnyBusNodeWidget.PROFILE.default ||
                                    this.properties[MaraScottAnyBusNodeWidget.PROFILE.name] === sourceBusNode.properties[MaraScottAnyBusNodeWidget.PROFILE.name]) {
                                    MaraScottAnyBus_v2.connectBus(this, slotIndex, sourceBusNode, link.origin_slot);
                                } else {
                                    this.disconnectInput(slotIndex);
                                }
                            }
                        } else {
                            // Handle non-BUS connections
                            MaraScottAnyBus_v2.connectInput(this, slotIndex, originNode, link.origin_slot);
                        }
                    }
                } else if (!isConnected) {
                    // Handle disconnections
                    if (slotIndex === MaraScottAnyBus_v2.BUS_SLOT) {
                        MaraScottAnyBus_v2.disConnectBus(this);
                    } else {
                        MaraScottAnyBus_v2.disConnectInput(this, slotIndex);
                    }
                }

                MaraScottAnyBusNodeFlow.syncProfile(this, null, isConnected);
                onConnectionsChange?.apply(this, arguments);
            };

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                MaraScottAnyBusNodeWidget.init(this);
                MaraScottAnyBusNodeWidget.setWidgets(this);
                
                onNodeCreated?.apply(this, arguments);
                this.serialize_widgets = true;
            };
        }
    }
};

app.registerExtension(MaraScottAnyBusNodeExtension);
