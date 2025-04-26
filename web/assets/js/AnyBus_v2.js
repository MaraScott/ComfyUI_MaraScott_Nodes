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

    static init(node) {
        if (!node.properties) {
            node.properties = {};
        }
        node.properties[this.PROFILE.name] = node.properties[this.PROFILE.name] ?? this.PROFILE.default;
        node.properties[this.INPUTS.name] = node.properties[this.INPUTS.name] ?? this.INPUTS.default;
        
        // Set node appearance
        node.shape = LiteGraph.CARD_SHAPE;
        node.color = LGraphCanvas.node_colors.green.color;
        node.bgcolor = LGraphCanvas.node_colors.green.bgcolor;
        node.groupcolor = LGraphCanvas.node_colors.green.groupcolor;
        node.size[0] = 150;
        
        // Update node title
        this.updateNodeTitle(node);
    }

    static findOriginBusNode(node, graph) {
        if (!node || !graph) return null;

        // If it's an AnyBus node, we found what we're looking for
        if (node.type === MaraScottAnyBusNodeWidget.NAMESPACE + MaraScottAnyBusNodeWidget.TYPE) {
            return node;
        }

        // If it's not a reroute or getset type node, stop here
        if (!this.ALLOWED_NODE_TYPE.includes(node.type)) {
            return null;
        }

        // For reroute/getset nodes, check their inputs
        if (node.inputs && node.inputs.length > 0) {
            for (const input of node.inputs) {
                if (input.link != null) {
                    const link = graph.links[input.link];
                    if (link) {
                        const prevNode = graph.getNodeById(link.origin_id);
                        if (prevNode) {
                            const originBusNode = this.findOriginBusNode(prevNode, graph);
                            if (originBusNode) {
                                return originBusNode;
                            }
                        }
                    }
                }
            }
        }

        return null;
    }

    static validateBusConnection(node, originNode, graph) {
        // If it's a direct AnyBus connection, allow it
        if (originNode.type === MaraScottAnyBusNodeWidget.NAMESPACE + MaraScottAnyBusNodeWidget.TYPE) {
            return true;
        }

        // If it's an allowed reroute/getset type, check its origin
        if (this.ALLOWED_NODE_TYPE.includes(originNode.type)) {
            const originBusNode = this.findOriginBusNode(originNode, graph);
            return originBusNode !== null;
        }

        return false;
    }

    static getByName(node, name) {
        return node.widgets?.find(w => w.name === name);
    }

    static setValue(node, name, value) {
        try {
            if (name === this.PROFILE.name) {
                node.properties[name] = value;
                this.updateNodeTitle(node);
            } else if (name === this.INPUTS.name) {
                const numInputs = parseInt(value);
                if (numInputs >= this.INPUTS.min && numInputs <= this.INPUTS.max) {
                    node.properties[name] = numInputs;
                    if (node.graph) {
                        this.updateNodeIO(node);
                    }
                }
            }
            node.setDirtyCanvas(true);
        } catch (err) {
            console.error('AnyBus: Error setting value:', err);
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
        
        // Store existing connections and their types
        const connections = oldInputs.map((input, i) => {
            if (input.link != null) {
                const link = node.graph.links[input.link];
                if (link) {
                    const originNode = node.graph.getNodeById(link.origin_id);
                    if (originNode && originNode.outputs[link.origin_slot]) {
                        // For BUS connections, validate the connection
                        if (i === this.BUS_SLOT) {
                            if (!this.validateBusConnection(node, originNode, node.graph)) {
                                return null;
                            }
                        }
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

        // Clear existing inputs/outputs safely
        node.inputs = [];
        node.outputs = [];

        // Add bus input/output first (slot 0)
        node.addInput("bus", "BUS");
        node.addOutput("bus", "BUS");

        // Add dynamic inputs/outputs (slots 1+)
        for (let i = 1; i <= numInputs; i++) {
            const defaultName = `* ${i.toString().padStart(2, '0')}`;
            const connection = connections.find(c => c.slot === i);
            
            // For inputs: use connected type/name if available, otherwise default
            node.addInput(
                connection ? connection.name.toLowerCase() : defaultName.toLowerCase(),
                connection ? connection.type : "*"
            );
            
            // For outputs: always use connected name if available, but preserve type
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
        // Set up profile text input first
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

        // Set up input number selector
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
            // Store original onConnectionsChange
            const onConnectionsChange = nodeType.prototype.onConnectionsChange;
            
            // Override onConnectionsChange to handle connection updates
            nodeType.prototype.onConnectionsChange = function(type, slotIndex, isConnected, link, ioSlot) {
                // For new connections to the BUS slot
                if (isConnected && slotIndex === MaraScottAnyBusNodeWidget.BUS_SLOT && link) {
                    const originNode = this.graph.getNodeById(link.origin_id);
                    if (originNode && !MaraScottAnyBusNodeWidget.validateBusConnection(this, originNode, this.graph)) {
                        // If validation fails, prevent the connection
                        if (link) {
                            this.graph.removeLink(link.id);
                        }
                        return;
                    }
                }

                // Call original handler if it exists
                onConnectionsChange?.apply(this, arguments);

                // Handle non-bus connections (slot > 0)
                if (slotIndex > 0 && this.inputs && this.outputs) {
                    if (isConnected && link) {
                        const originNode = this.graph.getNodeById(link.origin_id);
                        if (originNode && originNode.outputs && originNode.outputs[link.origin_slot]) {
                            // Update input and output type/name to match connected node
                            const originOutput = originNode.outputs[link.origin_slot];
                            if (this.inputs[slotIndex] && this.outputs[slotIndex]) {
                                this.inputs[slotIndex].type = originOutput.type;
                                this.inputs[slotIndex].name = originOutput.name.toLowerCase();
                                this.outputs[slotIndex].name = originOutput.name.toLowerCase();
                                this.outputs[slotIndex].type = originOutput.type;
                            }
                        }
                    } else if (this.inputs[slotIndex] && this.outputs[slotIndex]) {
                        // Reset to default on disconnect, but preserve output type
                        const defaultName = `* ${slotIndex.toString().padStart(2, '0')}`;
                        this.inputs[slotIndex].type = "*";
                        this.inputs[slotIndex].name = defaultName.toLowerCase();
                        this.outputs[slotIndex].name = defaultName.toLowerCase();
                    }
                    this.setDirtyCanvas(true);
                }
            };

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                // Initialize everything
                MaraScottAnyBusNodeWidget.init(this);
                MaraScottAnyBusNodeWidget.setWidgets(this);
                
                // Set up initial IO after properties are initialized
                setTimeout(() => {
                    MaraScottAnyBusNodeWidget.updateNodeIO(this);
                }, 0);
                
                // Call original onNodeCreated if it exists
                onNodeCreated?.apply(this, arguments);
                
                this.serialize_widgets = true;
            }
        }
    },

    async setup() {
        console.log(MaraScottAnyBusNodeWidget.NAMESPACE + MaraScottAnyBusNodeWidget.TYPE + " extension setup complete");
    },
}

// Register the extension
app.registerExtension(MaraScottAnyBusNodeExtension);
