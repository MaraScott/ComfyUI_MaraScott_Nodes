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

if (!window.marasit) {
	window.marasit = {}
}
if (!window.marasit.anyBus) {
	window.marasit.anyBus = {
		init: false,
		sync: false,
		input: {
			label: "0",
			index: 0,
		},
		clean: false,
		nodeToSync: null,
		flows: [],
		nodes: [],
	}
}

class MarasitAnyBusNodeWidget {

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

		this.setProfileInput(node)
		this.setInputsSelect(node)
		// this.setCleanSwitch(node)

	}

	static getByName(node, name) {
		return node.widgets?.find((w) => w.name === name);
	}

	static setValueProfile(node, name, value) {
		node.title = "AnyBus - " + node.properties[name];
	}

	static setValueInputs(node, name, value) {
		let qty = 0
		let _value = value + MarasitAnyBusNode.FIRST_INDEX
		if (node.inputs.length > _value) {
			qty = node.inputs.length - _value
			for (let i = qty; i > 0; i--) {
				node.removeInput(node.inputs.length - 1)
				node.removeOutput(node.outputs.length - 1)
			}
		} else if (node.inputs.length < _value) {
			qty = _value - node.inputs.length
			for (let i = 0; i < qty; i++) {
				const name = "* " + node.inputs.length.toString().padStart(2, '0')
				const type = "*"
				node.addInput(name, type)
				node.addOutput(name, type)
			}
		}
	}

	static setValue(node, name, value) {

		const nodeWidget = this.getByName(node, name);
		nodeWidget.value = value
		node.setProperty(name, nodeWidget.value ?? node.properties[name])
		if (name == this.PROFILE.name) this.setValueProfile(node, name, value)
		if (name == this.INPUTS.name) this.setValueInputs(node, name, value)
		node.setDirtyCanvas(true)

	}

	static setProfileInput(node) {

		const nodeWidget = this.getByName(node, this.PROFILE.name);

		if (nodeWidget == undefined) {
			node.addWidget(
				"text",
				this.PROFILE.name,
				node.properties[this.PROFILE.name] ?? this.PROFILE.default,
				(value, LGraphCanvas, Node, Coordinate, PointerEvent) => {
					this.setValue(node, this.PROFILE.name, value)
					window.marasit.anyBus.sync = MarasitAnyBusNodeFlow.FULLSYNC;
					MarasitAnyBusNodeFlow.syncProfile(node, this.PROFILE.name, null)
					node.setProperty('prevProfileName', node.properties[this.PROFILE.name])

				},
				{}
			)
			this.setValue(node, this.PROFILE.name, this.PROFILE.default)
			node.setProperty('prevProfileName', node.properties[this.PROFILE.name])
		}

	}

	static setInputsSelect(node) {

		const nodeWidget = this.getByName(node, this.INPUTS.name);

		if (nodeWidget == undefined) {

			let values = []

			for (let i = this.INPUTS.min; i <= this.INPUTS.max; i++) {
				values.push(i);
			}

			node.addWidget(
				"combo",
				this.INPUTS.name,
				node.properties[this.INPUTS.name] ?? this.INPUTS.default,
				(value, LGraphCanvas, Node, Coordinate, PointerEvent) => {
					this.setValue(node, this.INPUTS.name, value)
					window.marasit.anyBus.sync = MarasitAnyBusNodeFlow.FULLSYNC;
					MarasitAnyBusNodeFlow.syncProfile(node, this.INPUTS.name, null)
				},
				{
					"values": values
				}
			)
			node.setProperty(this.INPUTS.name, this.INPUTS.default)
			this.setValue(node, this.INPUTS.name, this.INPUTS.default)
		}

	}

	static setCleanSwitch(node) {

		const nodeWidget = this.getByName(node, this.CLEAN.name);
		if (nodeWidget == undefined) {
			node.addWidget(
				"toggle",
				this.CLEAN.name,
				this.CLEAN.clean,
				(value, LGraphCanvas, Node, Coordinate, PointerEvent) => {
					MarasitAnyBusNodeFlow.clean(node)
					this.setValue(node, this.CLEAN.name, this.CLEAN.clean)
				},
				{}
			)
			this.setValue(node, this.CLEAN.name, this.CLEAN.clean)
		}

	}

}

class MarasitAnyBusNode {

	static BUS_SLOT = 0
	static BASIC_PIPE_SLOT = 0
	static REFINER_PIPE_SLOT = 0

	static FIRST_INDEX = 1

	static configure(node) {

		node.shape = LiteGraph.CARD_SHAPE // BOX_SHAPE | ROUND_SHAPE | CIRCLE_SHAPE | CARD_SHAPE
		node.color = LGraphCanvas.node_colors.green.color
		node.bgcolor = LGraphCanvas.node_colors.green.bgcolor
		node.groupcolor = LGraphCanvas.node_colors.green.groupcolor
		node.groupcolor = LGraphCanvas.node_colors.green.groupcolor
		node.size[0] = 150 // width
		if (!node.properties || !(MarasitAnyBusNodeWidget.PROFILE.name in node.properties)) {
			node.properties[MarasitAnyBusNodeWidget.PROFILE.name] = MarasitAnyBusNodeWidget.PROFILE.default;
		}
		node.title = "AnyBus - " + node.properties[MarasitAnyBusNodeWidget.PROFILE.name]
	}

	static setWidgets(node) {
		MarasitAnyBusNodeWidget.init(node)
	}

	static setInputValue(node) {

		let protected_slots = []

		for (let slot = this.FIRST_INDEX; slot < window.marasit.anyBus.nodeToSync.inputs.length; slot++) {

			if (protected_slots.indexOf(slot) > -1) continue

			if (typeof node.inputs[slot] == 'undefined' || typeof window.marasit.anyBus.nodeToSync.inputs[slot] == 'undefined') {
				console.log('[MarasIT Nodes] Check your profile Names')
				continue;
			}

			const isNodeInputDifferent = node.inputs[slot].type != "*" && node.inputs[slot].type != window.marasit.anyBus.nodeToSync.inputs[slot].type

			if (isNodeInputDifferent) {
				const preSyncMode = window.marasit.anyBus.sync;
				window.marasit.anyBus.sync = this.NOSYNC;
				if (node.inputs[slot].link == null) {
					node.disconnectInput(slot)
					node.disconnectOutput(slot)
				} else {
					protected_slots.push(node.id)
				}
				window.marasit.anyBus.sync = preSyncMode;
			}
			if (window.marasit.anyBus.nodeToSync.id != node.id) {
				if (node.inputs[slot].link == null) {
					node.inputs[slot].name = window.marasit.anyBus.nodeToSync.inputs[slot].name.toLowerCase()
					node.inputs[slot].type = window.marasit.anyBus.nodeToSync.inputs[slot].type
					node.outputs[slot].name = node.inputs[slot].name
					node.outputs[slot].type = node.inputs[slot].type
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

		let syncType = MarasitAnyBusNodeFlow.NOSYNC
		if (isForward || isBackward || isFull) syncType = MarasitAnyBusNodeFlow.FULLSYNC

		return syncType

	}

	static getBusParentNodeWithInput(node, slot) {
		let parentNode = null

		if (node.inputs[0].link != null) {

			const parentLink = node.graph.links.find(
				(otherLink) => otherLink?.id == node.inputs[0].link
			)
			parentNode = node.graph.getNodeById(parentLink.origin_id)

			if (parentNode.inputs[slot].link == null) {
				parentNode = this.getBusParentNodeWithInput(parentNode, slot)
			}

		}

		if (parentNode != null) {
			node.inputs[slot].name = parentNode.inputs[slot].name
			node.inputs[slot].type = parentNode.inputs[slot].type
		} else {
			node.inputs[slot].name = "* " + window.marasit.anyBus.input.index.toString().padStart(2, '0')
			node.inputs[slot].type = "*"
		}

		node.outputs[slot].name = node.inputs[slot].name
		node.outputs[slot].type = node.inputs[slot].type

		return parentNode
	}
	
	static disConnectBus(node, slot) {

		return MarasitAnyBusNodeFlow.FULLSYNC

	}

	static disConnectInput(node, slot, clean) {

		const syncProfile = this.getSyncType(node, slot, null, null)
		const previousBusNode = this.getBusParentNodeWithInput(node, slot)
		let busNodes = []
		const busNodePaths = MarasitAnyBusNodeFlow.getFlows(node)

		let newName = "* " + window.marasit.anyBus.input.index.toString().padStart(2, '0')
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
		node.outputs[slot].type = node.inputs[slot].type
		if(window.marasit.anyBus.clean) MarasitAnyBusNodeFlow.clean(node)

		return syncProfile

	}	

	static connectBus(node, slot, node_origin) {

		const syncProfile = MarasitAnyBusNodeFlow.FULLSYNC
		const isBusInput = slot == 0
		const isOutputs = node_origin.outputs?.length > 0
		let isMarasitBusNode = node_origin.type == "MarasitAnyBusNode"
		if (!isMarasitBusNode) {
			const origin_reroute_node = MarasitAnyBusNodeFlow.getOriginRerouteBusType(node_origin)
			isMarasitBusNode = origin_reroute_node.type == "MarasitAnyBusNode"
			if (isMarasitBusNode) {
				node_origin = origin_reroute_node
			}
		}
		const isOriginProfileSame = node.properties[MarasitAnyBusNodeWidget.PROFILE.name] == node_origin.properties[MarasitAnyBusNodeWidget.PROFILE.name]
		const isTargetProfileDefault = node.properties[MarasitAnyBusNodeWidget.PROFILE.name] == MarasitAnyBusNodeWidget.PROFILE.default
		if (isBusInput && isOutputs && isMarasitBusNode && (isOriginProfileSame || isTargetProfileDefault)) {
			if (isTargetProfileDefault) {
				MarasitAnyBusNodeWidget.setValue(node, MarasitAnyBusNodeWidget.PROFILE.name, node_origin.properties[MarasitAnyBusNodeWidget.PROFILE.name])
				node.setProperty('prevProfileName', node.properties[MarasitAnyBusNodeWidget.PROFILE.name])
			}
			MarasitAnyBusNodeWidget.setValue(node, MarasitAnyBusNodeWidget.INPUTS.name, node_origin.properties[MarasitAnyBusNodeWidget.INPUTS.name])
			for (let _slot = window.marasit.anyBus.input.index; _slot < node_origin.outputs.length; _slot++) {
				if (_slot > node_origin.properties[MarasitAnyBusNodeWidget.INPUTS.name]) {
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

		let syncProfile = MarasitAnyBusNodeFlow.NOSYNC
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
		return syncProfile
	}

}

class MarasitAnyBusNodeFlow {

	static NOSYNC = 0
	static FULLSYNC = 1

	static ALLOWED_REROUTE_TYPE = [
		"Reroute",
	]
	static ALLOWED_NODE_TYPE = [
		"MarasitAnyBusNode",
		...this.ALLOWED_REROUTE_TYPE,
	]

	static getLastBuses(nodes) {
		// Find all parent nodes
		let parents = Object.values(nodes);
		const leafs = Object.keys(nodes);
		let leafSet = new Set(leafs);

		let lastLeaves = leafs.filter(leaf => {
			// Check if the leaf is not a parent to any other node
			return !parents.includes(parseInt(leaf));
		});

		return lastLeaves;
	}

	static getOriginRerouteBusType(node) {
		let originNode = null
		let _originNode = null
		let isMarasitAnyBusNode = false

		if (node.inputs[0].link != null) {

			const __originLink = node.graph.links.find(
				(otherLink) => otherLink?.id == node.inputs[0].link
			)
			_originNode = node.graph.getNodeById(__originLink.origin_id)

			if (_originNode.type == 'Reroute' && _originNode?.__outputType == 'BUS') {
				_originNode = this.getOriginRerouteBusType(_originNode)
			}
			if (_originNode?.type == "MarasitAnyBusNode") {
				originNode = _originNode
			}

		}

		return originNode

	}

	static getFlows(node) {

		let bus_flows = {}
		let nodes_list = []
		let nodes_paths = {}
		for (let i in node.graph._nodes) {
			let _node = node.graph._nodes[i]
			let _previous_node = null
			if (this.ALLOWED_NODE_TYPE.includes(_node.type) && nodes_list.indexOf(_node.id) == -1) {
				nodes_list.push(_node.id);
				let _previousNode_id = null;
				if (_node.inputs[0].link != null) {
					const _previousNode_link = node.graph.links.find(
						(otherLink) => otherLink?.id == _node.inputs[0].link
					)
					if (_previousNode_link) _previousNode_id = _previousNode_link.origin_id
				}

				nodes_paths[_node.id] = _previousNode_id
			}
		}
		const lastBuseNodeIds = this.getLastBuses(nodes_paths)
		for (let i in lastBuseNodeIds) {
			let _node = node.graph.getNodeById(lastBuseNodeIds[i])
			if (_node.properties[MarasitAnyBusNodeWidget.PROFILE.name] == node.properties['prevProfileName']) {
				bus_flows[lastBuseNodeIds[i]] = this.setFlows(_node)
			}
		}
		let busFlowRef = []
		for (let i in bus_flows) {
			if (bus_flows[i].indexOf(node.id)) {
				busFlowRef = bus_flows[i]
			}
		}
		bus_flows = this.getLinkedFlows(bus_flows, busFlowRef)
		return bus_flows
	}

	static setFlows(node) {

		let _backward_node = []
		let _backward_node_list = []
		for (let i in node.graph._nodes) {
			let _node = node.graph._nodes[i]
			if (this.ALLOWED_NODE_TYPE.includes(_node.type) && _backward_node_list.indexOf(_node.id) == -1) {
				_backward_node_list.push(_node.id);
				if (_node.inputs[0].link != null) _backward_node.push(_node);
			}
		}

		// bus network
		let _backward_bus_node_link = null
		let backward_bus_nodes = []
		let backward_bus_node_connections = {}
		for (let i in _backward_node_list) {
			backward_bus_nodes.push(node.graph.getNodeById(_backward_node_list[i]))
		}
		for (let i in backward_bus_nodes) {
			_backward_bus_node_link = backward_bus_nodes[i].inputs[0].link
			if (_backward_bus_node_link != null) {
				_backward_bus_node_link = node.graph.links.find(
					(otherLink) => otherLink?.id == _backward_bus_node_link
				)
				if (_backward_bus_node_link) backward_bus_node_connections[backward_bus_nodes[i].id] = _backward_bus_node_link.origin_id
			}
		}

		let currentNode = node.id
		const backward_path = [currentNode]; // Initialize the path with the starting node
		while (backward_bus_node_connections[currentNode] !== undefined) {
			currentNode = backward_bus_node_connections[currentNode]; // Move to the parent node
			const _currentNode = node.graph.getNodeById(currentNode)
			if (_currentNode.type == "MarasitAnyBusNode") {
				backward_path.push(currentNode); // Add the parent node to the path
			}
		}

		return backward_path;

	}

	static getLinkedFlows(bus_flows, busFlowRef) {

		const _bus_flows = Object.entries(bus_flows);

		// Filter the entries based on the condition that at least one node in the value array is included in busFlowRef
		const _filtered_bus_flows = _bus_flows.filter(([id, bus_flow]) =>
			bus_flow.some(node => busFlowRef.includes(node))
		);

		// Reconstruct the object from the filtered entries
		const filtered_bus_flows = Object.fromEntries(_filtered_bus_flows);

		return filtered_bus_flows;
	}

	static syncProfile(node, isChangeWidget, isChangeConnect) {

		if (!node.graph || window.marasit.anyBus.sync == MarasitAnyBusNodeFlow.NOSYNC) return
		
		let busNodes = []
		const busNodePaths = this.getFlows(node)
		let startIndex = null;
		const syncProfile = window.marasit.anyBus.sync
		for (let i in busNodePaths) {
			startIndex = null;
			busNodes = busNodePaths[i]

			if (syncProfile == MarasitAnyBusNodeFlow.FULLSYNC) {

				busNodes?.reverse()
				startIndex = 0
			}

			if (startIndex != null) {
				this.sync(node, busNodes, isChangeWidget, isChangeConnect)
			}

		}

		window.marasit.anyBus.sync = MarasitAnyBusNodeFlow.NOSYNC
	}

	static sync(node, busNodes, isChangeWidget, isChangeConnect) {

		window.marasit.anyBus.nodeToSync = node;

		let _node = null
		for (let i in busNodes) {
			_node = node.graph.getNodeById(busNodes[i])
			if (_node.id !== window.marasit.anyBus.nodeToSync.id && this.ALLOWED_REROUTE_TYPE.indexOf(_node.type) == -1) {
				if (isChangeWidget != null) {
					MarasitAnyBusNodeWidget.setValue(_node, isChangeWidget, node.properties[isChangeWidget])
					if (isChangeWidget == MarasitAnyBusNodeWidget.PROFILE.name) _node.setProperty('prevProfileName', node.properties[MarasitAnyBusNodeWidget.PROFILE.name])
				}
				if (isChangeConnect !== null) MarasitAnyBusNode.setInputValue(_node)
			}
		}

	}


	static clean(node) {

		window.marasit.anyBus.clean = false
		let _node_origin_link = null
		let _node_origin = null
		if (node.inputs[MarasitAnyBusNode.BUS_SLOT].link != null) {
			_node_origin_link = node.graph.links.find(
				(otherLink) => otherLink?.id == node.inputs[MarasitAnyBusNode.BUS_SLOT].link
			)
			_node_origin = node.graph.getNodeById(_node_origin_link.origin_id)
			// console.log('disconnect',node.id)
			node.disconnectInput(MarasitAnyBusNode.BUS_SLOT)
			// console.log('reconnect', _node_origin.id, '=>', node.id)
			_node_origin.connect(MarasitAnyBusNode.BUS_SLOT, node, MarasitAnyBusNode.BUS_SLOT)

		} else {

			// disconnect
			for (let slot = MarasitAnyBusNode.FIRST_INDEX; slot < node.inputs.length; slot++) {

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
					node.inputs[slot].name = "* " + (slot + 1 - MarasitAnyBusNode.FIRST_INDEX).toString().padStart(2, '0')
					node.inputs[slot].type = "*"
					node.outputs[slot].name = node.inputs[slot].name
					node.outputs[slot].type = node.inputs[slot].type
				}
			}
			window.marasit.anyBus.sync = MarasitAnyBusNodeFlow.FULLSYNC
			MarasitAnyBusNodeFlow.syncProfile(node, MarasitAnyBusNodeWidget.CLEAN.name, false)
			MarasitAnyBusNodeFlow.syncProfile(node, null, true)
			window.marasit.anyBus.clean = true
		}
		const cleanedLabel = " ... cleaned"
		node.title = node.title + cleanedLabel
		setTimeout(() => {
			// Remove " (cleaned)" from the title
			node.title = node.title.replace(cleanedLabel, "");
		}, 500);

	}

}

class MarasitAnyBusNodeLiteGraph {

	static onExecuted(nodeType) {
		const onExecuted = nodeType.prototype.onExecuted
		nodeType.prototype.onExecuted = function (message) {
			onExecuted?.apply(this, arguments)
			console.log("[MarasIT - logging " + this.name + "]", "on Executed", { "id": this.id, "properties": this.properties });
		}

	}

	static onNodeCreated(nodeType) {

		const onNodeCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function () {
			const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

			console.log("[MarasIT - logging " + this.name + "]", 'onNodeCreated')
			MarasitAnyBusNode.configure(this)
			MarasitAnyBusNode.setWidgets(this)

			return r;
		}

	}

	static getExtraMenuOptions(nodeType) {
		const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
		nodeType.prototype.getExtraMenuOptions = function (_, options) {

			console.log("[MarasIT - logging " + this.name + "]", "on Extra Menu Options", { "id": this.id, "properties": this.properties });

		}
	}

	static onConnectionsChange(nodeType) {

		const onConnectionsChange = nodeType.prototype.onConnectionsChange
		nodeType.prototype.onConnectionsChange = function (
			slotType,	//1 = input, 2 = output
			slot,
			isChangeConnect,
			link_info,
			output
		) {
			const r = onConnectionsChange ? onConnectionsChange.apply(this, arguments) : undefined

			if (!window.marasit.anyBus.init) return r

			window.marasit.anyBus.sync = MarasitAnyBusNodeFlow.NOSYNC
			window.marasit.anyBus.input.index = slot + 1 - MarasitAnyBusNode.FIRST_INDEX

			//On Disconnect
			if (!isChangeConnect && slotType == 1 && typeof link_info != 'undefined') {
				// console.log('disconnect');

				if (slot < MarasitAnyBusNode.FIRST_INDEX) {
					// bus
					if (slot == 0 && this.inputs) {
						window.marasit.anyBus.sync = MarasitAnyBusNode.disConnectBus(this)
					}

				} else {

					window.marasit.anyBus.sync = MarasitAnyBusNode.disConnectInput(this, slot)
				}

			}
			if (!isChangeConnect && slotType == 2 && typeof link_info != 'undefined') {
				// do something
				// this.inputs[slot].name = "* ("+slot.toString().padStart(2, '0')+")"
			}
			//On Connect
			if (isChangeConnect && slotType == 1 && typeof link_info != 'undefined' && this.graph) {
				// console.log('connect');
				// do something
				let link_info_node = this.graph._nodes.find(
					(otherNode) => otherNode.id == link_info.origin_id
				)

				if (slot < window.marasit.anyBus.input.index) {
					// bus
					window.marasit.anyBus.sync = MarasitAnyBusNode.connectBus(this, slot, link_info_node)
				} else {

					window.marasit.anyBus.sync = MarasitAnyBusNode.connectInput(this, slot, link_info_node, link_info.origin_slot)


				}

			}

			if (isChangeConnect && slotType == 2 && typeof link_info != 'undefined') {
				// do something
				// this.inputs[slot].name = ":) ("+slot.toString().padStart(2, '0')+")"
			}

			MarasitAnyBusNodeFlow.syncProfile(this, null, isChangeConnect)

			return r;
		}

	}

	static onRemoved(nodeType) {
		const onRemoved = nodeType.prototype.onRemoved;
		nodeType.prototype.onRemoved = function () {
			onRemoved?.apply(this, arguments);
			console.log('onRemoved')
		};
	}


}

const MarasitAnyBusNodeExtension = {
	// Unique name for the extension
	name: "Comfy.MarasIT.AnyBusNode",
	init(app) {
		// Any initial setup to run as soon as the page loads
		console.log("[MarasIT - logging " + this.name + "]", "extension init");
	},
	setup(app) {
		// Any setup to run after the app is created
		console.log("[MarasIT - logging " + this.name + "]", "extension setup");
	},
	// !TODO should I find a way to define defs based on profile ?
	addCustomNodeDefs(defs, app) {
		// Add custom node definitions
		// These definitions will be configured and registered automatically
		// defs is a lookup core nodes, add yours into this
		const withNodesNames = false
		if(withNodesNames) {
			console.log("[MarasIT - logging " + this.name + "]", "add custom node definitions", "current nodes:", defs['MarasitAnyBusNode'], JSON.stringify(Object.keys(defs)));

		} else {
			console.log("[MarasIT - logging " + this.name + "]", "add custom node definitions", "current nodes:", defs['MarasitAnyBusNode']);
		}
	},
	getCustomWidgets(app) {
		// Return custom widget types
		// See ComfyWidgets for widget examples
		console.log("[MarasIT - logging " + this.name + "]", "provide custom widgets");
	},
	registerCustomNodes(app) {
		// Register any custom node implementations here allowing for more flexability than a custom node def
		console.log("[MarasIT - logging " + this.name + "]", "register custom nodes");
	},
	loadedGraphNode(node, app) {
		// Fires for each node when loading/dragging/etc a workflow json or png
		// If you break something in the backend and want to patch workflows in the frontend
		// This is the place to do this
		if (node.type == "MarasitAnyBusNode") {

			node.setProperty('uuid', node.id)
			console.log("[MarasIT - logging " + this.name + "]", "Loaded Graph", { "id": node.id, "properties": node.properties });

		}

		// This fires for every node on each load so only log once
		// delete MarasitAnyBusNode.loadedGraphNode;
	},
	// this is the python node created
	nodeCreated(node, app) {
		// Fires every time a node is constructed
		// You can modify widgets/add handlers/etc here
		console.log("[MarasIT - logging " + this.name + "]", "node created: ", { ...node });

		// This fires for every node so only log once
		// delete MarasitAnyBusNode.nodeCreated;
	},
	beforeRegisterNodeDef(nodeType, nodeData, app) {
		// Run custom logic before a node definition is registered with the graph

		if (nodeData.name === 'MarasitAnyBusNode') {
			// This fires for every node definition so only log once
			console.log("[MarasIT - logging " + this.name + "]", "before register node: ", nodeData, typeof MarasitAnyBusNodeLiteGraph, typeof MarasitAnyBusNodeLiteGraph.onNodeCreated);
			
			// MarasitAnyBusNodeLiteGraph.onExecuted(nodeType)
			MarasitAnyBusNodeLiteGraph.onNodeCreated(nodeType)
			// MarasitAnyBusNodeLiteGraph.getExtraMenuOptions(nodeType)
			MarasitAnyBusNodeLiteGraph.onConnectionsChange(nodeType)
			// delete MarasitAnyBusNode.beforeRegisterNodeDef;
			MarasitAnyBusNodeLiteGraph.onRemoved(nodeType)

		}
	},
	beforeConfigureGraph(app) {
		console.log("[MarasIT - logging " + this.name + "]", "extension beforeConfigureGraph");
		window.marasit.anyBus.init = false
	},
	afterConfigureGraph(app) {
		console.log("[MarasIT - logging " + this.name + "]", "extension afterConfigureGraph");
		window.marasit.anyBus.init = true
	},

};

app.registerExtension(MarasitAnyBusNodeExtension);
