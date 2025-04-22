import { Widget } from "./widgets.js";
import { Bus } from "./bus.js";

export class Flow {

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
        Bus.TYPE,
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

            if (_originNode?.type == Bus.TYPE) {
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
                if (_currentNode.type == Bus.TYPE) {
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

        if (!node.graph || window.marascott.AnyBus_v2.sync == this.NOSYNC) return
        if (window.marascott.AnyBus_v2.sync == this.FULLSYNC) {
            this.setFlows(node);
            const busNodes = [].concat(...this.getFlows(node)).filter((value, index, self) => self.indexOf(value) === index)
            this.sync(node, busNodes, isChangeWidget, isChangeConnect)
        }

        window.marascott.AnyBus_v2.sync = this.NOSYNC
    }

    static sync(node, busNodes, isChangeWidget, isChangeConnect) {

        window.marascott.AnyBus_v2.nodeToSync = node;

        let _node = null
        for (let i in busNodes) {
            _node = node.graph.getNodeById(busNodes[i])
            if (_node.id !== window.marascott.AnyBus_v2.nodeToSync.id && this.ALLOWED_REROUTE_TYPE.indexOf(_node.type) == -1 && this.ALLOWED_GETSET_TYPE.indexOf(_node.type) == -1) {
                if (isChangeWidget != null) {
                    Widget.setValue(_node, isChangeWidget, window.marascott.AnyBus_v2.nodeToSync.properties[isChangeWidget])
                    if (isChangeWidget == Widget.PROFILE.name) _node.setProperty('prevProfileName', window.marascott.AnyBus_v2.nodeToSync.properties[Widget.PROFILE.name])
                }
                if (isChangeConnect !== null) {
                    Widget.setValue(_node, Widget.INPUTS.name, window.marascott.AnyBus_v2.nodeToSync.properties[Widget.INPUTS.name])
                    Bus.setInputValue(_node)
                }
            }
        }

        window.marascott.AnyBus_v2.nodeToSync = null;

    }


    static clean(node) {

        window.marascott.AnyBus_v2.clean = false
        let _node_origin_link = null
        let _node_origin = null
        if (node.inputs[Bus.BUS_SLOT].link != null) {

            const nodeGraphLinks = Array.isArray(node.graph.links)
                ? node.graph.links
                : Object.values(node.graph.links);

            _node_origin_link = nodeGraphLinks.find(
                (otherLink) => otherLink?.id == node.inputs[Bus.BUS_SLOT].link
            )
            _node_origin = node.graph.getNodeById(_node_origin_link.origin_id)
            // console.log('disconnect',node.id)
            node.disconnectInput(Bus.BUS_SLOT)
            // console.log('reconnect', _node_origin.id, '=>', node.id)
            _node_origin.connect(Bus.BUS_SLOT, node, Bus.BUS_SLOT)

        } else {

            // disconnect
            for (let slot = Bus.FIRST_INDEX; slot < node.inputs.length; slot++) {

                if (node.inputs[slot].link != null) {

                    const nodeGraphLinks = Array.isArray(node.graph.links)
                        ? node.graph.links
                        : Object.values(node.graph.links);

                    _node_origin_link = nodeGraphLinks.find(
                        (otherLink) => otherLink?.id == node.inputs[slot].link
                    )
                    _node_origin = node.graph.getNodeById(_node_origin_link.origin_id)
                    node.disconnectInput(slot)
                    // console.log('reconnect', _node_origin.id, '=>', node.id)
                    _node_origin.connect(_node_origin_link.origin_slot, node, slot)

                } else {

                    node.disconnectInput(slot)
                    node.inputs[slot].name = "* " + (slot + 1 - Bus.FIRST_INDEX).toString().padStart(2, '0')
                    node.inputs[slot].type = "*"
                    node.outputs[slot].name = node.inputs[slot].name
                    node.outputs[slot].type = node.inputs[slot].type

                }

            }
            window.marascott.AnyBus_v2.sync = this.FULLSYNC
            this.syncProfile(node, Widget.CLEAN.name, false)
            this.syncProfile(node, null, true)
            window.marascott.AnyBus_v2.clean = true
        }
        const cleanedLabel = " ... cleaned"
        node.title = node.title + cleanedLabel
        setTimeout(() => {
            // Remove " (cleaned)" from the title
            node.title = node.title.replace(cleanedLabel, "");
        }, 500);

    }

}