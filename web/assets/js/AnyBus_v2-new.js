import { app } from "../../scripts/app.js";

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
        MaraScottAnyBusNodeWidget.NAMESPACE + MaraScottAnyBusNodeWidget.TYPE,
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

    static init(node) {
        if (!node.properties) {
            node.properties = {};
        }
        node.properties[this.PROFILE.name] = node.properties[this.PROFILE.name] ?? this.PROFILE.default;
        node.properties[this.INPUTS.name] = node.properties[this.INPUTS.name] ?? this.INPUTS.default;
        node.properties['prevProfileName'] = node.properties[this.PROFILE.name];
        
        // Set node appearance
        node.shape = LiteGraph.CARD_SHAPE;
        node.color = LGraphCanvas.node_colors.green.color;
        node.bgcolor = LGraphCanvas.node_colors.green.bgcolor;
        node.groupcolor = LGraphCanvas.node_colors.green.groupcolor;
        node.size[0] = 150;
        
        this.updateNodeTitle(node);
    }

    static findOriginBusNode(node, graph) {
        if (!node || !graph) return null;

        // If it's an AnyBus node, we found what we're looking for
        if (node.type === MaraScottAnyBusNodeWidget.NAMESPACE + MaraScottAnyBusNodeWidget.TYPE) {
            return node;
        }

        // For reroute/getset nodes, check their inputs
        if (node.inputs && node.inputs.length > 0 && node.inputs[0].link != null) {
            const link = graph.links[node.inputs[0].link];
            if (link) {
                const prevNode = graph.getNodeById(link.origin_id);
                if (prevNode && this.ALLOWED_NODE_TYPE.includes(prevNode.type)) {
                    const originBusNode = this.findOriginBusNode(prevNode, graph);
                    if (originBusNode) {
                        return originBusNode;
                    }
                }
            }
        }
        return null;
    }

    static findFlowNodes(startNode, graph, includingStart = false) {
        const nodes = new Set();
        
        function traverse(node, forward = true) {
            if (!node) return;
            
            if (forward) {
                // Forward traversal through outputs
                if (node.outputs && node.outputs[0].links) {
                    for (const linkId of node.outputs[0].links) {
                        const link = graph.links[linkId];
                        if (link) {
                            const nextNode = graph.getNodeById(link.target_id);
                            if (nextNode && nextNode.type === MaraScottAnyBusNodeWidget.NAMESPACE + MaraScottAnyBusNodeWidget.TYPE && !nodes.has(nextNode)) {
                                nodes.add(nextNode);
                                traverse(nextNode, true);
                            }
                        }
                    }
                }
            } else {
                // Backward traversal through inputs
                if (node.inputs && node.inputs[0].link) {
                    const link = graph.links[node.inputs[0].link];
                    if (link) {
                        const prevNode = graph.getNodeById(link.origin_id);
                        if (prevNode) {
                            const originBusNode = this.findOriginBusNode(prevNode, graph);
                            if (originBusNode && !nodes.has(originBusNode)) {
                                nodes.add(originBusNode);
                                traverse(originBusNode, false);
                            }
                        }
                    }
                }
            }
        }

        if (includingStart) {
            nodes.add(startNode);
        }
        traverse(startNode, true);  // Forward traversal
        traverse(startNode, false); // Backward traversal
        
        return Array.from(nodes);
    }

    static findDestinationBusNodes(node, graph) {
        const destNodes = [];
        if (!node || !graph) return destNodes;

        // Look through all nodes to find those connected to this one
        for (const n of graph._nodes) {
            if (n.type === MaraScottAnyBusNodeWidget.NAMESPACE + MaraScottAnyBusNodeWidget.TYPE && n.inputs && n.inputs[0].link) {
                const link = graph.links[n.inputs[0].link];
                if (link) {
                    const sourceNode = graph.getNodeById(link.origin_id);
                    if (sourceNode) {
                        if (sourceNode === node) {
                            destNodes.push(n);
                        } else if (this.ALLOWED_NODE_TYPE.includes(sourceNode.type)) {
                            const originBusNode = this.findOriginBusNode(sourceNode, graph);
                            if (originBusNode === node) {
                                destNodes.push(n);
                            }
                        }
                    }
                }
            }
        }

        return destNodes;
    }

    static validateBusConnection(node, originNode, graph) {
        if (!node || !graph) return false;
        
        const sourceBusNode = this.findOriginBusNode(originNode, graph);
        if (!sourceBusNode) return false;
        
        // Allow connection if target is default profile
        if (node.properties[this.PROFILE.name] === this.PROFILE.default) {
            return true;
        }
        
        // Allow connection if profiles match
        return node.properties[this.PROFILE.name] === sourceBusNode.properties[this.PROFILE.name];
    }

    static syncNodeWithSource(node, sourceNode, options = { propagateForward: true, propagateBackward: true }) {
        if (!node || !sourceNode || !node.graph) return;
        
        const syncedNodes = new Set();
        
        function syncInputs(targetNode, sourceNode, isForwardSync = true) {
            if (syncedNodes.has(targetNode)) return;
            syncedNodes.add(targetNode);

            // Sync profile if target is default or matches source
            if (targetNode.properties[MaraScottAnyBusNodeWidget.PROFILE.name] === MaraScottAnyBusNodeWidget.PROFILE.default ||
                targetNode.properties[MaraScottAnyBusNodeWidget.PROFILE.name] === sourceNode.properties[MaraScottAnyBusNodeWidget.PROFILE.name]) {
                MaraScottAnyBusNodeWidget.setValue(targetNode, MaraScottAnyBusNodeWidget.PROFILE.name, sourceNode.properties[MaraScottAnyBusNodeWidget.PROFILE.name]);
                targetNode.setProperty('prevProfileName', sourceNode.properties[MaraScottAnyBusNodeWidget.PROFILE.name]);
            }
            
            // Always sync number of inputs to maintain flow consistency
            if (targetNode.properties[MaraScottAnyBusNodeWidget.INPUTS.name] !== sourceNode.properties[MaraScottAnyBusNodeWidget.INPUTS.name]) {
                MaraScottAnyBusNodeWidget.setValue(targetNode, MaraScottAnyBusNodeWidget.INPUTS.name, sourceNode.properties[MaraScottAnyBusNodeWidget.INPUTS.name]);
            }
            
            // Sync inputs and outputs
            for (let i = MaraScottAnyBusNodeWidget.FIRST_INDEX; i < sourceNode.outputs.length && i < targetNode.inputs.length; i++) {
                if (targetNode.inputs[i] && targetNode.outputs[i]) {
                    const targetInput = targetNode.inputs[i];
                    const sourceOutput = sourceNode.outputs[i];
                    
                    // If input has no connection, sync it
                    if (targetInput.link == null) {
                        targetInput.name = sourceOutput.name;
                        targetInput.type = sourceOutput.type;
                        targetNode.outputs[i].name = sourceOutput.name;
                        targetNode.outputs[i].type = sourceOutput.type;
                    }
                }
            }
            
            // Forward synchronization
            if (isForwardSync && options.propagateForward && targetNode.outputs && targetNode.outputs[0].links) {
                for (const linkId of targetNode.outputs[0].links) {
                    const link = targetNode.graph.links[linkId];
                    if (link) {
                        const nextNode = targetNode.graph.getNodeById(link.target_id);
                        if (nextNode && nextNode.type === "AnyBus_v2") {
                            syncInputs(nextNode, targetNode, true);
                        }
                    }
                }
            }
            
            // Backward synchronization
            if (!isForwardSync && options.propagateBackward && targetNode.inputs && targetNode.inputs[0].link) {
                const link = targetNode.graph.links[targetNode.inputs[0].link];
                if (link) {
                    const prevNode = targetNode.graph.getNodeById(link.origin_id);
                    if (prevNode) {
                        const originBusNode = MaraScottAnyBusNodeWidget.findOriginBusNode(prevNode, targetNode.graph);
                        if (originBusNode) {
                            // First sync profile and input count
                            if (originBusNode.properties[MaraScottAnyBusNodeWidget.PROFILE.name] === MaraScottAnyBusNodeWidget.PROFILE.default) {
                                MaraScottAnyBusNodeWidget.setValue(originBusNode, MaraScottAnyBusNodeWidget.PROFILE.name, targetNode.properties[MaraScottAnyBusNodeWidget.PROFILE.name]);
                                originBusNode.setProperty('prevProfileName', targetNode.properties[MaraScottAnyBusNodeWidget.PROFILE.name]);
                            }
                            
                            if (originBusNode.properties[MaraScottAnyBusNodeWidget.INPUTS.name] !== targetNode.properties[MaraScottAnyBusNodeWidget.INPUTS.name]) {
                                MaraScottAnyBusNodeWidget.setValue(originBusNode, MaraScottAnyBusNodeWidget.INPUTS.name, targetNode.properties[MaraScottAnyBusNodeWidget.INPUTS.name]);
                            }
                            
                            // Then sync unconnected inputs
                            for (let i = MaraScottAnyBusNodeWidget.FIRST_INDEX; i < originBusNode.inputs.length; i++) {
                                const originInput = originBusNode.inputs[i];
                                if (originInput && i < targetNode.inputs.length && originInput.link == null) {
                                    originInput.name = targetNode.inputs[i].name;
                                    originInput.type = targetNode.inputs[i].type;
                                    originBusNode.outputs[i].name = targetNode.inputs[i].name;
                                    originBusNode.outputs[i].type = targetNode.inputs[i].type;
                                }
                            }
                            
                            // Continue backward propagation
                            syncInputs(originBusNode, targetNode, false);
                        }
                    }
                }
            }
        }
        
        // Start synchronization
        syncInputs(node, sourceNode, true);
        
        // If this is an initial change, also do backward sync
        if (options.propagateBackward) {
            syncInputs(node, sourceNode, false);
        }
    }

    static getByName(node, name) {
        return node.widgets?.find(w => w.name === name);
    }

    static setValue(node, name, value) {
        const oldValue = node.properties[name];
        node.properties[name] = value;

        if (name === this.PROFILE.name) {
            this.updateNodeTitle(node);
            // Store previous profile for sync validation
            node.properties['prevProfileName'] = value;
            
            // Propagate profile change through bus connections
            const destNodes = this.findDestinationBusNodes(node, node.graph);
            for (const destNode of destNodes) {
                if (destNode.properties[this.PROFILE.name] === this.PROFILE.default) {
                    this.setValue(destNode, this.PROFILE.name, value);
                }
            }
        } 
        else if (name === this.INPUTS.name) {
            if (oldValue !== value) {
                this.updateNodeIO(node);
                // Propagate input count change
                const destNodes = this.findDestinationBusNodes(node, node.graph);
                for (const destNode of destNodes) {
                    this.setValue(destNode, this.INPUTS.name, value);
                }
            }
        }
    }

    static updateNodeTitle(node) {
        node.title = "AnyBus - " + node.properties[this.PROFILE.name];
    }

    static updateNodeIO(node) {
        if (!node.graph) {
            console.warn('AnyBus: Cannot update IO, node not attached to graph');
            return;
        }

        const numInputs = node.properties[this.INPUTS.name];
        const oldInputs = [...(node.inputs || [])];
        const oldOutputs = [...(node.outputs || [])];
        
        // Store existing connections
        const connections = oldInputs.map((input, i) => {
            if (input.link != null) {
                const link = node.graph.links[input.link];
                if (link) {
                    const originNode = node.graph.getNodeById(link.origin_id);
                    if (originNode && originNode.outputs[link.origin_slot]) {
                        return {
                            slot: i,
                            originNode: originNode,
                            originSlot: link.origin_slot,
                            type: originNode.outputs[link.origin_slot].type,
                            name: originNode.outputs[link.origin_slot].name
                        };
                    }
                }
            }
            return null;
        }).filter(x => x);

        // Clear existing inputs/outputs
        node.inputs = [];
        node.outputs = [];

        // Add bus input/output
        node.addInput("bus", "BUS");
        node.addOutput("bus", "BUS");

        // Add dynamic inputs/outputs
        for (let i = 1; i <= numInputs; i++) {
            const defaultName = `* ${i.toString().padStart(2, '0')}`;
            const connection = connections.find(c => c.slot === i);
            
            node.addInput(
                connection ? connection.name.toLowerCase() : defaultName.toLowerCase(),
                connection ? connection.type : "*"
            );
            
            node.addOutput(
                connection ? connection.name.toLowerCase() : defaultName.toLowerCase(),
                connection ? connection.type : "*"
            );
        }

        // Restore compatible connections
        if (connections.length > 0) {
            connections.forEach(conn => {
                if (conn.slot < node.inputs.length) {
                    node.connect(conn.slot, conn.originNode, conn.originSlot);
                }
            });
        }

        node.setDirtyCanvas(true);
    }

    static setWidgets(node) {
        // Profile text input
        if (!this.getByName(node, this.PROFILE.name)) {
            node.addWidget(
                "text",
                this.PROFILE.name,
                node.properties[this.PROFILE.name] ?? this.PROFILE.default,
                (value) => {
                    this.setValue(node, this.PROFILE.name, value);
                }
            );
        }

        // Input count selector
        if (!this.getByName(node, this.INPUTS.name)) {
            const values = Array.from(
                { length: this.INPUTS.max - this.INPUTS.min + 1 }, 
                (_, i) => i + this.INPUTS.min
            );
            node.addWidget(
                "combo",
                this.INPUTS.name,
                this.INPUTS.default,
                (value) => {
                    this.setValue(node, this.INPUTS.name, value);
                },
                { values: values }
            );
        }
    }
}

const MaraScottAnyBusNodeExtension = {
    name: "Comfy.MaraScott.AnyBus_v2",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeType.comfyClass === MaraScottAnyBusNodeWidget.NAMESPACE + MaraScottAnyBusNodeWidget.TYPE) {
            const onConnectionsChange = nodeType.prototype.onConnectionsChange;
            
            nodeType.prototype.onConnectionsChange = function(type, slotIndex, isConnected, link, ioSlot) {
                // Handle new connections
                if (isConnected && link) {
                    const originNode = this.graph.getNodeById(link.origin_id);
                    if (originNode) {
                        if (slotIndex === MaraScottAnyBusNodeWidget.BUS_SLOT) {
                            // Validate and handle BUS connections
                            if (!MaraScottAnyBusNodeWidget.validateBusConnection(this, originNode, this.graph)) {
                                this.graph.removeLink(link.id);
                                return;
                            }
                            
                            // Find actual source bus node
                            const sourceBusNode = MaraScottAnyBusNodeWidget.findOriginBusNode(originNode, this.graph);
                            if (sourceBusNode) {
                                // Sync with source node
                                MaraScottAnyBusNodeWidget.syncNodeWithSource(this, sourceBusNode);
                            }
                        } else {
                            // Handle non-BUS connections
                            const originOutput = originNode.outputs[link.origin_slot];
                            if (this.inputs[slotIndex] && this.outputs[slotIndex]) {
                                this.inputs[slotIndex].name = originOutput.name;
                                this.inputs[slotIndex].type = originOutput.type;
                                this.outputs[slotIndex].name = originOutput.name;
                                this.outputs[slotIndex].type = originOutput.type;
                            }
                        }
                    }
                } else if (!isConnected) {
                    // Handle disconnections
                    if (slotIndex > 0 && this.inputs[slotIndex] && this.outputs[slotIndex]) {
                        const defaultName = `* ${slotIndex.toString().padStart(2, '0')}`;
                        this.inputs[slotIndex].name = defaultName;
                        this.inputs[slotIndex].type = "*";
                        this.outputs[slotIndex].name = defaultName;
                        this.outputs[slotIndex].type = "*";
                    }
                }

                onConnectionsChange?.apply(this, arguments);
                this.setDirtyCanvas(true);
            };

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                MaraScottAnyBusNodeWidget.init(this);
                MaraScottAnyBusNodeWidget.setWidgets(this);
                
                setTimeout(() => {
                    MaraScottAnyBusNodeWidget.updateNodeIO(this);
                }, 0);
                
                onNodeCreated?.apply(this, arguments);
                this.serialize_widgets = true;
            }
        }
    },

    async setup() {
        console.log(MaraScottAnyBusNodeWidget.NAMESPACE + MaraScottAnyBusNodeWidget.TYPE + " extension setup complete");
    },
}

app.registerExtension(MaraScottAnyBusNodeExtension);
