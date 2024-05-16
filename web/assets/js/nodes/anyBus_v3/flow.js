class flow {

	_ext = null

	NOSYNC = 0
	FULLSYNC = 1

	ALLOWED_REROUTE_TYPE = [
		"Reroute (rgthree)", // SUPPORTED - RgThree Custom Node
		// "Reroute", // UNSUPPORTED - ComfyUI native - do not allow connection on Any Type if origin Type is not Any Type too
		// "ReroutePrimitive|pysssss", // UNSUPPORTED - Pysssss Custom Node - do not display the name of the origin slot
		// "0246.CastReroute", //  UNSUPPORTED - 0246 Custom Node
	]
	ALLOWED_GETSET_TYPE = [
		"SetNode", // SUPPORTED - ComfyUI-KJNodes Custom Node
		"GetNode", // SUPPORTED - ComfyUI-KJNodes Custom Node
	]
	ALLOWED_NODE_TYPE = [
	]

	constructor(extension) {

		this.ext = extension

		this.ALLOWED_NODE_TYPE = [
			this.ext.TYPE,
			...this.ALLOWED_REROUTE_TYPE,
			...this.ALLOWED_GETSET_TYPE,
		]
	
	}

	getLastBuses(nodes) {
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

	getOriginRerouteBusType(node) {
		let originNode = null
		let _originNode = null
		let isMaraScottAnyBusNode_v3 = false

		if (node && node.inputs && node.inputs[0].link != null) {

			if(node.inputs[0].link == 'setNode') {

				_originNode = node.graph.getNodeById(node.inputs[0].origin_id)

			} else {

				const __originLink = node.graph.links.find(
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

			if (_originNode?.type == this.ext.TYPE) {
				originNode = _originNode
			}

		}

		return originNode

	}

	getFlows(node) {

		let firstItem = null;
		for (let list of window.marascott[this.ext.name].flows.list) {
			if (list.includes(node.id)) {
				firstItem = list[0];
				break;
			}
		}

		return window.marascott[this.ext.name].flows.list.filter(list => list[0] === firstItem);

	}

	setFlows(node) {

		let _nodes = []
		let _nodes_list = []
		for (let i in node.graph._nodes) {
			let _node = node.graph._nodes[i]
			if (this.ALLOWED_NODE_TYPE.includes(_node.type) && _nodes_list.indexOf(_node.id) == -1) {
				_nodes_list.push(_node.id);
				if(_node.type == 'GetNode') {
					const _setnode = _node.findSetter(_node.graph)
					if(_setnode) _node.inputs = [ {'link' : 'setNode', 'origin_id': _setnode.id}]
				}
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
			window.marascott[this.ext.name].nodes[_node.id] = _node
		}
		for (let i in _bus_nodes) {
			_bus_node_link = _bus_nodes[i].inputs[0].link
			if (_bus_node_link == 'setNode') {
				if (_bus_nodes[i].inputs[0].origin_id) _bus_nodes_connections[_bus_nodes[i].id] = _bus_nodes[i].inputs[0].origin_id
			} else if (_bus_node_link != null) {
				_bus_node_link = node.graph.links.find(
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
				if (_currentNode.type == this.ext.TYPE) {
					node_paths[id].push(currentNode); // Add the parent node to the path
				}
			}
			node_paths[id].reverse()
		}
		
		window.marascott[this.ext.name].flows.start = node_paths_start
		window.marascott[this.ext.name].flows.end = node_paths_end
		window.marascott[this.ext.name].flows.list = node_paths

	}

	syncProfile(node, isChangeWidget, isChangeConnect) {

		if (!node.graph || window.marascott[this.ext.name].sync == this.NOSYNC) return
		if (window.marascott[this.ext.name].sync == this.FULLSYNC) {
			this.setFlows(node);
			const busNodes = [].concat(...this.getFlows(node)).filter((value, index, self) => self.indexOf(value) === index)
			this.sync(node, busNodes, isChangeWidget, isChangeConnect)
		}

		window.marascott[this.ext.name].sync = this.NOSYNC
	}

	sync(node, busNodes, isChangeWidget, isChangeConnect) {

		window.marascott[this.ext.name].nodeToSync = node;

		let _node = null
		for (let i in busNodes) {
			_node = node.graph.getNodeById(busNodes[i])
			if (_node.id !== window.marascott[this.ext.name].nodeToSync.id && this.ALLOWED_REROUTE_TYPE.indexOf(_node.type) == -1 && this.ALLOWED_GETSET_TYPE.indexOf(_node.type) == -1) {
				if (isChangeWidget != null) {
					this.ext.widget.setValue(_node, isChangeWidget, window.marascott[this.ext.name].nodeToSync.properties[isChangeWidget])
					if (isChangeWidget == this.ext.widget.PROFILE.name) _node.setProperty('prevProfileName', window.marascott[this.ext.name].nodeToSync.properties[this.ext.widget.PROFILE.name])
				}
				if (isChangeConnect !== null) {
					this.ext.widget.setValue(_node, this.ext.widget.INPUTS.name, window.marascott[this.ext.name].nodeToSync.properties[this.ext.widget.INPUTS.name])
					this.ext.core.setInputValue(_node)
				}
			}
		}

		window.marascott[this.ext.name].nodeToSync = null;

	}


	clean(node) {

		window.marascott[this.ext.name].clean = false
		let _node_origin_link = null
		let _node_origin = null
		if (node.inputs[this.ext.core.BUS_SLOT].link != null) {
			_node_origin_link = node.graph.links.find(
				(otherLink) => otherLink?.id == node.inputs[this.ext.core.BUS_SLOT].link
			)
			_node_origin = node.graph.getNodeById(_node_origin_link.origin_id)
			// console.log('disconnect',node.id)
			node.disconnectInput(this.ext.core.BUS_SLOT)
			// console.log('reconnect', _node_origin.id, '=>', node.id)
			_node_origin.connect(this.ext.core.BUS_SLOT, node, this.ext.core.BUS_SLOT)

		} else {

			// disconnect
			for (let slot = this.ext.core.FIRST_INDEX; slot < node.inputs.length; slot++) {

				if (node.inputs[slot].link != null) {

					_node_origin_link = node.graph.links.find(
						(otherLink) => otherLink?.id == node.inputs[slot].link
					)
					_node_origin = node.graph.getNodeById(_node_origin_link.origin_id)
					node.disconnectInput(slot)
					// console.log('reconnect', _node_origin.id, '=>', node.id)
					_node_origin.connect(_node_origin_link.origin_slot, node, slot)

				} else {

					node.disconnectInput(slot)
					node.inputs[slot].name = "* " + (slot + 1 - this.ext.core.FIRST_INDEX).toString().padStart(2, '0')
					node.inputs[slot].type = "*"
					node.outputs[slot].name = node.inputs[slot].name
					node.outputs[slot].type = node.inputs[slot].type

				}

			}
			window.marascott[this.ext.name].sync = this.FULLSYNC
			this.syncProfile(node, this.ext.widget.CLEAN.name, false)
			this.syncProfile(node, null, true)
			window.marascott[this.ext.name].clean = true
		}
		const cleanedLabel = " ... cleaned"
		node.title = node.title + cleanedLabel
		setTimeout(() => {
			// Remove " (cleaned)" from the title
			node.title = node.title.replace(cleanedLabel, "");
		}, 500);

	}

    get ext(){
        return this._ext;
    }
    
    set ext(extension){
        this._ext = extension;
    }

}

export { flow }