import { Widget } from "./widgets.js";
import { Flow } from "./flow.js";
import { CONSTANTS } from "./constants.js";

export class Bus {

    static TYPE = CONSTANTS.NODE_TYPE

    static BUS_SLOT = CONSTANTS.BUS_SLOT
    static BASIC_PIPE_SLOT = CONSTANTS.BASIC_PIPE_SLOT
    static REFINER_PIPE_SLOT = CONSTANTS.REFINER_PIPE_SLOT

    static FIRST_INDEX = CONSTANTS.FIRST_INDEX

    static configure(node) {

        node.shape = LiteGraph.CARD_SHAPE // BOX_SHAPE | ROUND_SHAPE | CIRCLE_SHAPE | CARD_SHAPE
        node.color = LGraphCanvas.node_colors.green.color
        node.bgcolor = LGraphCanvas.node_colors.green.bgcolor
        node.groupcolor = LGraphCanvas.node_colors.green.groupcolor
        node.size[0] = 150 // width
        if (!node.properties || !(Widget.PROFILE.name in node.properties)) {
            node.properties[Widget.PROFILE.name] = Widget.PROFILE.default;
        }
        node.title = "AnyBus - " + node.properties[Widget.PROFILE.name]
    }

    static setWidgets(node) {
        Widget.init(node)
    }

    static setInputValue(node) {
        // Initial validation
        if (!node) {
            console.warn('[MaraScott Nodes] Node is undefined');
            return;
        }

        // Validate node structure
        if (!node.inputs || !node.outputs) {
            console.warn('[MaraScott Nodes] Invalid node structure - missing inputs/outputs');
            return;
        }

        // Validate global state
        const globalState = window.marascott?.AnyBus_v2;
        if (!globalState) {
            console.warn('[MaraScott Nodes] Global state not initialized');
            return;
        }

        const syncNode = globalState.nodeToSync;
        if (!syncNode?.inputs) {
            console.warn('[MaraScott Nodes] NodeToSync not properly initialized');
            return;
        }

        try {
            const protected_slots = new Set();
            
            // Safely determine inputs length
            const inputsLength = Math.min(
                syncNode.inputs?.length || 0,
                node.inputs?.length || 0
            );

            for (let slot = this.FIRST_INDEX; slot < inputsLength; slot++) {
                // Skip if slot is protected
                if (protected_slots.has(slot)) {
                    continue;
                }

                // Validate slot existence
                const currentInput = node.inputs[slot];
                const syncInput = syncNode.inputs[slot];
                const currentOutput = node.outputs[slot];
                const syncOutput = syncNode.outputs[slot];

                if (!currentInput || !syncInput || !currentOutput || !syncOutput) {
                    console.warn(`[MaraScott Nodes] Invalid slot configuration at slot ${slot}`);
                    continue;
                }

                // Check types and connections
                const isNodeInputAny = currentInput.type === "*";
                const isNodeOutputDifferent = currentOutput.type === syncOutput.type;
                const isNodeInputDifferent = !isNodeOutputDifferent;
                const isOutputAny = currentOutput.type === "*";
                const isOutputDifferent = currentOutput.type !== syncOutput.type;
                const isOutputLinked = Array.isArray(currentOutput.links) && currentOutput.links.length > 0;

                // Handle input differences
                if (isNodeInputDifferent) {
                    const preSyncMode = globalState.sync;
                    globalState.sync = Flow.NOSYNC;

                    if (!currentInput.link) {
                        this.safeDisconnect(node, slot);
                    } else {
                        protected_slots.add(slot);
                    }

                    globalState.sync = preSyncMode;
                }

                // Update node if needed
                if (syncNode.id !== node.id && !currentInput.link) {
                    this.updateNodeSlot(node, slot, syncNode, isOutputDifferent, isOutputLinked);
                }
            }
        } catch (error) {
            console.error('[MaraScott Nodes] Error in setInputValue:', error);
        }
    }

    // Helper method for safe disconnection
    static safeDisconnect(node, slot) {
        try {
            if (node?.inputs?.[slot]) {
                node.disconnectInput(slot);
            }
            if (node?.outputs?.[slot]) {
                node.disconnectOutput(slot);
            }
        } catch (error) {
            console.error(`[MaraScott Nodes] Error disconnecting slot ${slot}:`, error);
        }
    }

    // Helper method for updating node slot
    static updateNodeSlot(node, slot, syncNode, isOutputDifferent, isOutputLinked) {
        try {
            const currentInput = node.inputs[slot];
            const syncInput = syncNode.inputs[slot];
            const currentOutput = node.outputs[slot];

            if (!currentInput || !syncInput || !currentOutput) {
                return;
            }

            currentInput.name = syncInput.name.toLowerCase();
            currentInput.type = syncInput.type;
            currentOutput.name = currentInput.name;

            if (isOutputDifferent || !isOutputLinked) {
                currentOutput.type = currentInput.type;
            }
        } catch (error) {
            console.error(`[MaraScott Nodes] Error updating slot ${slot}:`, error);
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

        let syncType = Flow.NOSYNC
        if (isForward || isBackward || isFull) syncType = Flow.FULLSYNC

        return syncType

    }

    static getBusParentNodeWithInput(node, slot) {

        let parentNode = null

        if (node.inputs[0].link != null) {

            const nodeGraphLinks = Array.isArray(node.graph.links)
                ? node.graph.links
                : Object.values(node.graph.links);

            const parentLink = nodeGraphLinks.find(
                (otherLink) => otherLink?.id == node.inputs[0].link
            );

            if (parentLink != undefined) parentNode = node.graph.getNodeById(parentLink.origin_id)

            if (parentNode != null && Flow.ALLOWED_REROUTE_TYPE.indexOf(parentNode.type) > -1) {
                parentNode = this.getBusParentNodeWithInput(parentNode, slot)
            }
            if (parentNode != null && Flow.ALLOWED_GETSET_TYPE.indexOf(parentNode.type) > -1) {
                parentNode = this.getBusParentNodeWithInput(parentNode, slot)
            }
            if (parentNode != null && parentNode.inputs[slot].link == null) {
                parentNode = this.getBusParentNodeWithInput(parentNode, slot)
            }

        }

        if (parentNode != null && typeof node.inputs[slot] != "undefined" && Flow.ALLOWED_REROUTE_TYPE.indexOf(parentNode.type) == -1 && Flow.ALLOWED_GETSET_TYPE.indexOf(parentNode.type) == -1) {
            if (parentNode != null) {
                node.inputs[slot].name = parentNode.inputs[slot].name
                node.inputs[slot].type = parentNode.inputs[slot].type
            } else {
                node.inputs[slot].name = "* " + window.marascott.AnyBus_v2.input.index.toString().padStart(2, '0')
                node.inputs[slot].type = "*"
            }

            node.outputs[slot].name = node.inputs[slot].name
            node.outputs[slot].type = node.inputs[slot].type
        }
        return parentNode
    }

    static disConnectBus(node, slot) {

        return Flow.FULLSYNC

    }

    static disConnectInput(node, slot) {
        // Early validation with detailed error messages
        if (!node) {
            console.warn('[MaraScott Nodes] Node is undefined');
            return Flow.NOSYNC;
        }

        if (!node.graph) {
            console.warn('[MaraScott Nodes] Node graph is undefined');
            return Flow.NOSYNC;
        }

        if (!node.inputs?.[slot]) {
            console.warn(`[MaraScott Nodes] Input slot ${slot} is undefined`);
            return Flow.NOSYNC;
        }

        if (!node.outputs?.[slot]) {
            console.warn(`[MaraScott Nodes] Output slot ${slot} is undefined`);
            return Flow.NOSYNC;
        }
    
        try {
            const syncProfile = this.getSyncType(node, slot, null, null);
            const previousBusNode = this.getBusParentNodeWithInput(node, slot);
            const busNodePaths = Flow.getFlows(node) || [];
            
            // Default values with safe access to global state
            const defaultIndex = window.marascott?.AnyBus_v2?.input?.index ?? 0;
            let newName = `* ${defaultIndex.toString().padStart(2, '0')}`;
            let newType = "*";
    
            // Process bus nodes if paths exist
            if (busNodePaths.length > 0) {
                for (const busNodes of busNodePaths) {
                    // Create a new array instead of modifying the original
                    const reversedNodes = Array.from(busNodes).reverse();
                    
                    for (const nodeId of reversedNodes) {
                        const currentNode = node.graph.getNodeById(nodeId);
                        
                        // Skip if node or input slot doesn't exist
                        if (!currentNode?.inputs?.[slot]) continue;
    
                        const currentInput = currentNode.inputs[slot];
                        
                        if (currentInput.link != null && currentInput.type !== "*") {
                            newName = currentInput.name;
                            newType = currentInput.type;
                        } else if (currentInput.type === newType && currentInput.name !== newName) {
                            newName = currentInput.name;
                        }
                    }
                }
            }
    
            // Safely update node properties
            if (node.inputs?.[slot]) {
                node.inputs[slot].name = newName;
                node.inputs[slot].type = newType;
            }
            
            if (node.outputs?.[slot]) {
                node.outputs[slot].name = newName;
                // Type is intentionally not updated for outputs
            }

            return syncProfile;
    
        } catch (error) {
            console.error('[MaraScott Nodes] Error in disConnectInput:', error);
            return Flow.NOSYNC;
        }
    }

    static connectBus(node, slot, node_origin, origin_slot) {

        const syncProfile = Flow.FULLSYNC
        const isBusInput = slot == this.BUS_SLOT
        const isOutputs = node_origin.outputs?.length > 0
        let isMaraScottBusNode = node_origin.type == this.TYPE
        if (!isMaraScottBusNode) {
            const origin_reroute_node = Flow.getOriginRerouteBusType(node_origin)
            isMaraScottBusNode = origin_reroute_node?.type == this.TYPE
            if (isMaraScottBusNode) {
                node_origin = origin_reroute_node
            }
        }
        const isOriginProfileSame = node.properties[Widget.PROFILE.name] == node_origin.properties[Widget.PROFILE.name]
        const isTargetProfileDefault = node.properties[Widget.PROFILE.name] == Widget.PROFILE.default
        const isOriginSlotBus = origin_slot == this.BUS_SLOT
        if (isBusInput && isOriginSlotBus && isOutputs && isMaraScottBusNode && (isOriginProfileSame || isTargetProfileDefault)) {
            if (isTargetProfileDefault) {
                Widget.setValue(node, Widget.PROFILE.name, node_origin.properties[Widget.PROFILE.name])
                node.setProperty('prevProfileName', node.properties[Widget.PROFILE.name])
            }

            Widget.setValue(node, Widget.INPUTS.name, node_origin.properties[Widget.INPUTS.name])
            for (let _slot = this.FIRST_INDEX; _slot < node_origin.outputs.length; _slot++) {
                if (_slot > node_origin.properties[Widget.INPUTS.name]) {
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

        let syncProfile = Flow.NOSYNC
        const isOriginAnyBusBus = node_origin.type == this.TYPE
        const isOriginSlotBus = origin_slot == this.BUS_SLOT
        if (!(isOriginAnyBusBus && isOriginSlotBus)) {

            let anyPrefix = "* " + slot.toString().padStart(2, '0')
            let origin_name = node_origin.outputs[origin_slot]?.name.toLowerCase()
            let newName = origin_name
            if (origin_name && origin_name.indexOf("* ") === -1) {
                newName = anyPrefix + " - " + origin_name
            } else if (origin_name && origin_name.indexOf("* ") === 0 && node_origin.outputs[origin_slot].type === "*") {
                origin_name = "any"
                newName = anyPrefix + " - " + origin_name
            } else if (origin_name && origin_name.indexOf("* ") === 0) {
                origin_name = origin_name.split(" - ").pop()
                newName = anyPrefix + " - " + origin_name
            }
            if (node_origin.outputs[origin_slot] && node.inputs[slot].name == anyPrefix && node.inputs[slot].type == "*") {

                syncProfile = this.getSyncType(node, slot, node_origin, origin_slot)

                node.inputs[slot].name = newName
                node.inputs[slot].type = node_origin.outputs[origin_slot].type
                node.outputs[slot].name = node.inputs[slot].name
                node.outputs[slot].type = node.inputs[slot].type

            } else if (node.inputs[slot].type == node_origin.outputs[origin_slot]?.type && node.inputs[slot].type != newName) {

                syncProfile = this.getSyncType(node, slot, node_origin, origin_slot)

                node.inputs[slot].name = newName
                node.outputs[slot].name = node.inputs[slot].name

            }
        } else {
            node.disconnectInput(slot)
        }

        return syncProfile
    }

}
