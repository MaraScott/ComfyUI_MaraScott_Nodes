class core {
	
	BUS_SLOT = 0
	BASIC_PIPE_SLOT = 0
	REFINER_PIPE_SLOT = 0
	
	FIRST_INDEX = 1
	
	_ext = null

	constructor(extension) {
        this.ext = extension
	}

	configure(node) {

		node.shape = LiteGraph.CARD_SHAPE // BOX_SHAPE | ROUND_SHAPE | CIRCLE_SHAPE | CARD_SHAPE
		node.color = LGraphCanvas.node_colors.green.color
		node.bgcolor = LGraphCanvas.node_colors.green.bgcolor
		node.groupcolor = LGraphCanvas.node_colors.green.groupcolor
		node.groupcolor = LGraphCanvas.node_colors.green.groupcolor
		node.size[0] = 150 // width
		if (!node.properties || !(this.ext.widget.PROFILE.name in node.properties)) {
			node.properties[this.ext.widget.PROFILE.name] = this.ext.widget.PROFILE.default;
		}
		node.title = "AnyBus - " + node.properties[this.ext.widget.PROFILE.name]
	}

	setWidgets(node) {
		this.ext.widget.init(node)
	}

	setInputValue(node) {

		let protected_slots = []

		let inputsLength = window.marascott[this.ext.name].nodeToSync.inputs.length
		if(node.inputs.length < inputsLength) inputsLength = node.inputs.length

		for (let slot = this.FIRST_INDEX; slot < inputsLength; slot++) {

			if (protected_slots.indexOf(slot) > -1) continue
			if (typeof node.inputs[slot] == 'undefined' || typeof window.marascott[this.ext.name].nodeToSync.inputs[slot] == 'undefined') {
				console.log('[MaraScott Nodes] Check your profile Names')
				continue;
			}

			const isNodeInputAny = node.inputs[slot].type == "*"
			const isNodeOutputDifferent = node.outputs[slot].type == window.marascott[this.ext.name].nodeToSync.outputs[slot].type
			const isNodeInputDifferent = 
				!isNodeOutputDifferent // output different from new input
			const isOutputAny = node.outputs[slot].type == "*"
			const isOutputDifferent = node.outputs[slot].type != window.marascott[this.ext.name].nodeToSync.outputs[slot].type
			const isOutputLinked = node.outputs[slot].links != null &&node.outputs[slot].links.length > 0

			if (isNodeInputDifferent) {
				const preSyncMode = window.marascott[this.ext.name].sync;
				window.marascott[this.ext.name].sync = this.NOSYNC;
				if (node.inputs[slot].link == null) {
					node.disconnectInput(slot)
					node.disconnectOutput(slot)
				} else {
					protected_slots.push(node.id)
				}
				window.marascott[this.ext.name].sync = preSyncMode;
			}
			if (window.marascott[this.ext.name].nodeToSync.id != node.id) {
				if (node.inputs[slot].link == null) {
					node.inputs[slot].name = window.marascott[this.ext.name].nodeToSync.inputs[slot].name.toLowerCase()
					node.inputs[slot].type = window.marascott[this.ext.name].nodeToSync.inputs[slot].type
					node.outputs[slot].name = node.inputs[slot].name
					if(isOutputDifferent || !isOutputLinked) node.outputs[slot].type = node.inputs[slot].type
				}
			}
		}

	}

	getSyncType(node, slot, link_info_node, link_info_slot) {

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

		let syncType = this.ext.flow.NOSYNC
		if (isForward || isBackward || isFull) syncType = this.ext.flow.FULLSYNC

		return syncType

	}

	getBusParentNodeWithInput(node, slot) {

		let parentNode = null
		
		if (node.inputs[0].link != null) {

			const parentLink = node.graph.links.find(
				(otherLink) => otherLink?.id == node.inputs[0].link
			)
			if (parentLink != undefined) parentNode = node.graph.getNodeById(parentLink.origin_id)

			if (parentNode != null && this.ext.flow.ALLOWED_REROUTE_TYPE.indexOf(parentNode.type) > -1) {
				parentNode = this.getBusParentNodeWithInput(parentNode, slot)
			} 
			if (parentNode != null && this.ext.flow.ALLOWED_GETSET_TYPE.indexOf(parentNode.type) > -1) {
				parentNode = this.getBusParentNodeWithInput(parentNode, slot)
			} 
			if (parentNode != null && parentNode.inputs[slot].link == null) {
				parentNode = this.getBusParentNodeWithInput(parentNode, slot)
			}

		}

		if (parentNode != null && this.ext.flow.ALLOWED_REROUTE_TYPE.indexOf(parentNode.type) == -1 && this.ext.flow.ALLOWED_GETSET_TYPE.indexOf(parentNode.type) == -1) {
			if (parentNode != null) {
				node.inputs[slot].name = parentNode.inputs[slot].name
				node.inputs[slot].type = parentNode.inputs[slot].type
			} else {
				node.inputs[slot].name = "* " + window.marascott[this.ext.name].input.index.toString().padStart(2, '0')
				node.inputs[slot].type = "*"
			}

			node.outputs[slot].name = node.inputs[slot].name
			node.outputs[slot].type = node.inputs[slot].type
		}
		return parentNode
	}

	disConnectBus(node, slot) {
		
		return this.ext.flow.FULLSYNC

	}

	disConnectInput(node, slot) {

		const syncProfile = this.getSyncType(node, slot, null, null)
		const previousBusNode = this.getBusParentNodeWithInput(node, slot)
		let busNodes = []
		const busNodePaths = this.ext.flow.getFlows(node)

		let newName = "* " + window.marascott[this.ext.name].input.index.toString().padStart(2, '0')
		let newType = "*"
		let _node = null
		for (let i in busNodePaths) {
			busNodes = busNodePaths[i]
			busNodes.reverse()
			for (let y in busNodes) {
				_node = node.graph.getNodeById(busNodes[y])
				if (typeof _node.inputs[slot] != 'undefined' && _node.inputs[slot].link != null && _node.inputs[slot].type != "*") {
					newName = _node.inputs[slot].name
					newType = _node.inputs[slot].type
				} else if (typeof _node.inputs[slot] != 'undefined' && _node.inputs[slot].type == newType && _node.inputs[slot].name != newName) {
					newName = _node.inputs[slot].name
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

	connectBus(node, slot, node_origin, origin_slot) {

		const syncProfile = this.ext.flow.FULLSYNC
		const isBusInput = slot == this.BUS_SLOT
		const isOutputs = node_origin.outputs?.length > 0
		let isMaraScottBusNode = node_origin.type == this.ext.TYPE
		if (!isMaraScottBusNode) {
			const origin_reroute_node = this.ext.flow.getOriginRerouteBusType(node_origin)
			isMaraScottBusNode = origin_reroute_node?.type == this.ext.TYPE
			if (isMaraScottBusNode) {
				node_origin = origin_reroute_node
			}
		}
		const isOriginProfileSame = node.properties[this.ext.widget.PROFILE.name] == node_origin.properties[this.ext.widget.PROFILE.name]
		const isTargetProfileDefault = node.properties[this.ext.widget.PROFILE.name] == this.ext.widget.PROFILE.default
		const isOriginSlotBus = origin_slot == this.BUS_SLOT
		if (isBusInput && isOriginSlotBus && isOutputs && isMaraScottBusNode && (isOriginProfileSame || isTargetProfileDefault)) {
			if (isTargetProfileDefault) {
				this.ext.widget.setValue(node, this.ext.widget.PROFILE.name, node_origin.properties[this.ext.widget.PROFILE.name])
				node.setProperty('prevProfileName', node.properties[this.ext.widget.PROFILE.name])
			}

			this.ext.widget.setValue(node, this.ext.widget.INPUTS.name, node_origin.properties[this.ext.widget.INPUTS.name])
			for (let _slot = this.FIRST_INDEX; _slot < node_origin.outputs.length; _slot++) {
				if (_slot > node_origin.properties[this.ext.widget.INPUTS.name]) {
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

	connectInput(node, slot, node_origin, origin_slot) {

		let syncProfile = this.ext.flow.NOSYNC
		const isOriginAnyBusBus = node_origin.type == this.ext.TYPE
		const isOriginSlotBus = origin_slot == this.BUS_SLOT
		if(!(isOriginAnyBusBus && isOriginSlotBus)) {

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

    get ext(){
        return this._ext;
    }
    
    set ext(extension){
        this._ext = extension;
    }

}

export { core }