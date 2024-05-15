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
if (!window.marascott.anyBus) {
	window.marascott.anyBus = {
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
window.marascott.anyBus.nodes = {}
window.marascott.anyBus.flows = {
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

		this.setProfileInput(node)
		this.setInputsSelect(node)
		// this.setCleanSwitch(node)

	}

	static getByName(node, name) {
		return node.widgets?.find((w) => w.name === name);
	}

	// static removeByName(node, name) {
	// 	if (node.widgets) node.widgets = node.widgets.filter((w) => w.name !== name);
	// }

	static setValueProfile(node, name, value) {
		node.title = "AnyBus - " + node.properties[name];
	}

	static setValueInputs(node, name, value) {
		let qty = 0
		let _value = value + MaraScottAnyBusNode.FIRST_INDEX
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
					window.marascott.anyBus.sync = MaraScottAnyBusNodeFlow.FULLSYNC;
					MaraScottAnyBusNodeFlow.syncProfile(node, this.PROFILE.name, null)
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
					window.marascott.anyBus.sync = MaraScottAnyBusNodeFlow.FULLSYNC;
					MaraScottAnyBusNodeFlow.syncProfile(node, this.INPUTS.name, null)
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
					for (const index in window.marascott.anyBus.flows.end) {
						const _node = node.graph.getNodeById(window.marascott.anyBus.flows.end[index])
						MaraScottAnyBusNodeFlow.clean(_node)
					}
					this.setValue(node, this.CLEAN.name, this.CLEAN.clean)
				},
				{}
			)
			this.setValue(node, this.CLEAN.name, this.CLEAN.clean)
		}

	}

}

class MaraScottAnyBusNode {

	static TYPE = "MaraScottAnyBusNode"

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

		let inputsLength = window.marascott.anyBus.nodeToSync.inputs.length
		if(node.inputs.length < inputsLength) inputsLength = node.inputs.length

		for (let slot = this.FIRST_INDEX; slot < inputsLength; slot++) {

			if (protected_slots.indexOf(slot) > -1) continue
			if (typeof node.inputs[slot] == 'undefined' || typeof window.marascott.anyBus.nodeToSync.inputs[slot] == 'undefined') {
				console.log('[MaraScott Nodes] Check your profile Names')
				continue;
			}

			const isNodeInputAny = node.inputs[slot].type == "*"
			const isNodeOutputDifferent = node.outputs[slot].type == window.marascott.anyBus.nodeToSync.outputs[slot].type
			const isNodeInputDifferent = 
				!isNodeOutputDifferent // output different from new input
			const isOutputAny = node.outputs[slot].type == "*"
			const isOutputDifferent = node.outputs[slot].type != window.marascott.anyBus.nodeToSync.outputs[slot].type
			const isOutputLinked = node.outputs[slot].links != null &&node.outputs[slot].links.length > 0

			if (isNodeInputDifferent) {
				const preSyncMode = window.marascott.anyBus.sync;
				window.marascott.anyBus.sync = this.NOSYNC;
				if (node.inputs[slot].link == null) {
					node.disconnectInput(slot)
					node.disconnectOutput(slot)
				} else {
					protected_slots.push(node.id)
				}
				window.marascott.anyBus.sync = preSyncMode;
			}
			if (window.marascott.anyBus.nodeToSync.id != node.id) {
				if (node.inputs[slot].link == null) {
					node.inputs[slot].name = window.marascott.anyBus.nodeToSync.inputs[slot].name.toLowerCase()
					node.inputs[slot].type = window.marascott.anyBus.nodeToSync.inputs[slot].type
					node.outputs[slot].name = node.inputs[slot].name
					if(isOutputDifferent || !isOutputLinked) node.outputs[slot].type = node.inputs[slot].type
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

		let parentNode = null
		
		if (node.inputs[0].link != null) {

			const parentLink = node.graph.links.find(
				(otherLink) => otherLink?.id == node.inputs[0].link
			)
			if (parentLink != undefined) parentNode = node.graph.getNodeById(parentLink.origin_id)

			if (parentNode != null && MaraScottAnyBusNodeFlow.ALLOWED_REROUTE_TYPE.indexOf(parentNode.type) > -1) {
				parentNode = this.getBusParentNodeWithInput(parentNode, slot)
			} 
			if (parentNode != null && MaraScottAnyBusNodeFlow.ALLOWED_GETSET_TYPE.indexOf(parentNode.type) > -1) {
				parentNode = this.getBusParentNodeWithInput(parentNode, slot)
			} 
			if (parentNode != null && parentNode.inputs[slot].link == null) {
				parentNode = this.getBusParentNodeWithInput(parentNode, slot)
			}

		}

		if (parentNode != null && MaraScottAnyBusNodeFlow.ALLOWED_REROUTE_TYPE.indexOf(parentNode.type) == -1 && MaraScottAnyBusNodeFlow.ALLOWED_GETSET_TYPE.indexOf(parentNode.type) == -1) {
			if (parentNode != null) {
				node.inputs[slot].name = parentNode.inputs[slot].name
				node.inputs[slot].type = parentNode.inputs[slot].type
			} else {
				node.inputs[slot].name = "* " + window.marascott.anyBus.input.index.toString().padStart(2, '0')
				node.inputs[slot].type = "*"
			}

			node.outputs[slot].name = node.inputs[slot].name
			node.outputs[slot].type = node.inputs[slot].type
		}
		return parentNode
	}

	static disConnectBus(node, slot) {
		
		return MaraScottAnyBusNodeFlow.FULLSYNC

	}

	static disConnectInput(node, slot) {

		const syncProfile = this.getSyncType(node, slot, null, null)
		const previousBusNode = this.getBusParentNodeWithInput(node, slot)
		let busNodes = []
		const busNodePaths = MaraScottAnyBusNodeFlow.getFlows(node)

		let newName = "* " + window.marascott.anyBus.input.index.toString().padStart(2, '0')
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

	static connectBus(node, slot, node_origin, origin_slot) {

		const syncProfile = MaraScottAnyBusNodeFlow.FULLSYNC
		const isBusInput = slot == MaraScottAnyBusNode.BUS_SLOT
		const isOutputs = node_origin.outputs?.length > 0
		let isMaraScottBusNode = node_origin.type == MaraScottAnyBusNode.TYPE
		if (!isMaraScottBusNode) {
			const origin_reroute_node = MaraScottAnyBusNodeFlow.getOriginRerouteBusType(node_origin)
			isMaraScottBusNode = origin_reroute_node?.type == MaraScottAnyBusNode.TYPE
			if (isMaraScottBusNode) {
				node_origin = origin_reroute_node
			}
		}
		const isOriginProfileSame = node.properties[MaraScottAnyBusNodeWidget.PROFILE.name] == node_origin.properties[MaraScottAnyBusNodeWidget.PROFILE.name]
		const isTargetProfileDefault = node.properties[MaraScottAnyBusNodeWidget.PROFILE.name] == MaraScottAnyBusNodeWidget.PROFILE.default
		const isOriginSlotBus = origin_slot == MaraScottAnyBusNode.BUS_SLOT
		if (isBusInput && isOriginSlotBus && isOutputs && isMaraScottBusNode && (isOriginProfileSame || isTargetProfileDefault)) {
			if (isTargetProfileDefault) {
				MaraScottAnyBusNodeWidget.setValue(node, MaraScottAnyBusNodeWidget.PROFILE.name, node_origin.properties[MaraScottAnyBusNodeWidget.PROFILE.name])
				node.setProperty('prevProfileName', node.properties[MaraScottAnyBusNodeWidget.PROFILE.name])
			}

			MaraScottAnyBusNodeWidget.setValue(node, MaraScottAnyBusNodeWidget.INPUTS.name, node_origin.properties[MaraScottAnyBusNodeWidget.INPUTS.name])
			for (let _slot = MaraScottAnyBusNode.FIRST_INDEX; _slot < node_origin.outputs.length; _slot++) {
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

		let syncProfile = MaraScottAnyBusNodeFlow.NOSYNC
		const isOriginAnyBusBus = node_origin.type == MaraScottAnyBusNode.TYPE
		const isOriginSlotBus = origin_slot == MaraScottAnyBusNode.BUS_SLOT
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
		MaraScottAnyBusNode.TYPE,
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

			if (_originNode?.type == MaraScottAnyBusNode.TYPE) {
				originNode = _originNode
			}

		}

		return originNode

	}

	static getFlows(node) {

		let firstItem = null;
		for (let list of window.marascott.anyBus.flows.list) {
			if (list.includes(node.id)) {
				firstItem = list[0];
				break;
			}
		}

		return window.marascott.anyBus.flows.list.filter(list => list[0] === firstItem);

	}

	static setFlows(node) {

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
			window.marascott.anyBus.nodes[_node.id] = _node
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
				if (_currentNode.type == MaraScottAnyBusNode.TYPE) {
					node_paths[id].push(currentNode); // Add the parent node to the path
				}
			}
			node_paths[id].reverse()
		}
		
		window.marascott.anyBus.flows.start = node_paths_start
		window.marascott.anyBus.flows.end = node_paths_end
		window.marascott.anyBus.flows.list = node_paths

	}

	static syncProfile(node, isChangeWidget, isChangeConnect) {

		if (!node.graph || window.marascott.anyBus.sync == MaraScottAnyBusNodeFlow.NOSYNC) return
		if (window.marascott.anyBus.sync == MaraScottAnyBusNodeFlow.FULLSYNC) {
			MaraScottAnyBusNodeFlow.setFlows(node);
			const busNodes = [].concat(...this.getFlows(node)).filter((value, index, self) => self.indexOf(value) === index)
			this.sync(node, busNodes, isChangeWidget, isChangeConnect)
		}

		window.marascott.anyBus.sync = MaraScottAnyBusNodeFlow.NOSYNC
	}

	static sync(node, busNodes, isChangeWidget, isChangeConnect) {

		window.marascott.anyBus.nodeToSync = node;

		let _node = null
		for (let i in busNodes) {
			_node = node.graph.getNodeById(busNodes[i])
			if (_node.id !== window.marascott.anyBus.nodeToSync.id && this.ALLOWED_REROUTE_TYPE.indexOf(_node.type) == -1 && this.ALLOWED_GETSET_TYPE.indexOf(_node.type) == -1) {
				if (isChangeWidget != null) {
					MaraScottAnyBusNodeWidget.setValue(_node, isChangeWidget, window.marascott.anyBus.nodeToSync.properties[isChangeWidget])
					if (isChangeWidget == MaraScottAnyBusNodeWidget.PROFILE.name) _node.setProperty('prevProfileName', window.marascott.anyBus.nodeToSync.properties[MaraScottAnyBusNodeWidget.PROFILE.name])
				}
				if (isChangeConnect !== null) {
					MaraScottAnyBusNodeWidget.setValue(_node, MaraScottAnyBusNodeWidget.INPUTS.name, window.marascott.anyBus.nodeToSync.properties[MaraScottAnyBusNodeWidget.INPUTS.name])
					MaraScottAnyBusNode.setInputValue(_node)
				}
			}
		}

		window.marascott.anyBus.nodeToSync = null;

	}


	static clean(node) {

		window.marascott.anyBus.clean = false
		let _node_origin_link = null
		let _node_origin = null
		if (node.inputs[MaraScottAnyBusNode.BUS_SLOT].link != null) {
			_node_origin_link = node.graph.links.find(
				(otherLink) => otherLink?.id == node.inputs[MaraScottAnyBusNode.BUS_SLOT].link
			)
			_node_origin = node.graph.getNodeById(_node_origin_link.origin_id)
			// console.log('disconnect',node.id)
			node.disconnectInput(MaraScottAnyBusNode.BUS_SLOT)
			// console.log('reconnect', _node_origin.id, '=>', node.id)
			_node_origin.connect(MaraScottAnyBusNode.BUS_SLOT, node, MaraScottAnyBusNode.BUS_SLOT)

		} else {

			// disconnect
			for (let slot = MaraScottAnyBusNode.FIRST_INDEX; slot < node.inputs.length; slot++) {

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
					node.inputs[slot].name = "* " + (slot + 1 - MaraScottAnyBusNode.FIRST_INDEX).toString().padStart(2, '0')
					node.inputs[slot].type = "*"
					node.outputs[slot].name = node.inputs[slot].name
					node.outputs[slot].type = node.inputs[slot].type

				}

			}
			window.marascott.anyBus.sync = MaraScottAnyBusNodeFlow.FULLSYNC
			MaraScottAnyBusNodeFlow.syncProfile(node, MaraScottAnyBusNodeWidget.CLEAN.name, false)
			MaraScottAnyBusNodeFlow.syncProfile(node, null, true)
			window.marascott.anyBus.clean = true
		}
		const cleanedLabel = " ... cleaned"
		node.title = node.title + cleanedLabel
		setTimeout(() => {
			// Remove " (cleaned)" from the title
			node.title = node.title.replace(cleanedLabel, "");
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
			MaraScottAnyBusNode.configure(this)
			MaraScottAnyBusNode.setWidgets(this)

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

		const onConnectionsChange = nodeType.prototype.onConnectionsChange
		nodeType.prototype.onConnectionsChange = function (
			slotType,	//1 = input, 2 = output
			slot,
			isChangeConnect,
			link_info,
			output
			) {

			const r = onConnectionsChange ? onConnectionsChange.apply(this, arguments) : undefined

			if (!window.marascott.anyBus.init) return r

			window.marascott.anyBus.sync = MaraScottAnyBusNodeFlow.NOSYNC
			window.marascott.anyBus.input.index = slot + 1 - MaraScottAnyBusNode.FIRST_INDEX

			//On Disconnect
			if (!isChangeConnect && slotType == 1 && typeof link_info != 'undefined') {
				// console.log('disconnect');

				if (slot < MaraScottAnyBusNode.FIRST_INDEX) {
					// bus
					if (slot == 0 && this.inputs) {
						window.marascott.anyBus.sync = MaraScottAnyBusNode.disConnectBus(this)
					}

				} else {

					window.marascott.anyBus.sync = MaraScottAnyBusNode.disConnectInput(this, slot)
					
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
					
				if (slot < MaraScottAnyBusNode.FIRST_INDEX) {
					// bus
					window.marascott.anyBus.sync = MaraScottAnyBusNode.connectBus(this, slot, link_info_node, link_info.origin_slot)

				} else {

					window.marascott.anyBus.sync = MaraScottAnyBusNode.connectInput(this, slot, link_info_node, link_info.origin_slot)


				}

			}

			if (isChangeConnect && slotType == 2 && typeof link_info != 'undefined') {
				// do something
				// this.inputs[slot].name = ":) ("+slot.toString().padStart(2, '0')+")"
			}

			MaraScottAnyBusNodeFlow.syncProfile(this, null, isChangeConnect)

			return r;
		}

	}

	static onRemoved(nodeType) {
		const onRemoved = nodeType.prototype.onRemoved;
		nodeType.prototype.onRemoved = function () {
			onRemoved?.apply(this, arguments);
			// console.log('onRemoved')
		};
	}


}

const MaraScottAnyBusNodeExtension = {
	// Unique name for the extension
	name: "Comfy.MaraScott.AnyBusNode",
	init(app) {
		// Any initial setup to run as soon as the page loads
		// console.log("[MaraScott - logging " + this.name + "]", "extension init");
	},
	setup(app) {
		// Any setup to run after the app is created
		// console.log("[MaraScott - logging " + this.name + "]", "extension setup");
	},
	// !TODO should I find a way to define defs based on profile ?
	addCustomNodeDefs(defs, app) {
		// Add custom node definitions
		// These definitions will be configured and registered automatically
		// defs is a lookup core nodes, add yours into this
		const withNodesNames = false
		if (withNodesNames) {
			// console.log("[MaraScott - logging " + this.name + "]", "add custom node definitions", "current nodes:", defs[MaraScottAnyBusNode.TYPE], JSON.stringify(Object.keys(defs)));

		} else {
			// console.log("[MaraScott - logging " + this.name + "]", "add custom node definitions", "current nodes:", defs[MaraScottAnyBusNode.TYPE]);
		}
	},
	getCustomWidgets(app) {
		// Return custom widget types
		// See ComfyWidgets for widget examples
		// console.log("[MaraScott - logging " + this.name + "]", "provide custom widgets");
	},
	registerCustomNodes(app) {
		// Register any custom node implementations here allowing for more flexability than a custom node def
		// console.log("[MaraScott - logging " + this.name + "]", "register custom nodes");
	},
	loadedGraphNode(node, app) {
		// Fires for each node when loading/dragging/etc a workflow json or png
		// If you break something in the backend and want to patch workflows in the frontend
		// This is the place to do this
		if (node.type == MaraScottAnyBusNode.TYPE) {

			node.setProperty('uuid', node.id)
			MaraScottAnyBusNodeFlow.setFlows(node);
			// console.log("[MaraScott - logging " + this.name + "]", "Loaded Graph", { "id": node.id, "properties": node.properties });

		}

		// This fires for every node on each load so only log once
		// delete MaraScottAnyBusNode.loadedGraphNode;
	},
	// this is the python node created
	nodeCreated(node, app) {
		// Fires every time a node is constructed
		// You can modify widgets/add handlers/etc here
		// console.log("[MaraScott - logging " + this.name + "]", "node created: ", { ...node });

		// This fires for every node so only log once
		// delete MaraScottAnyBusNode.nodeCreated;
	},
	beforeRegisterNodeDef(nodeType, nodeData, app) {
		// Run custom logic before a node definition is registered with the graph

		if (nodeData.name === MaraScottAnyBusNode.TYPE) {
			// This fires for every node definition so only log once
			// console.log("[MaraScott - logging " + this.name + "]", "before register node: ", nodeData, typeof MaraScottAnyBusNodeLiteGraph, typeof MaraScottAnyBusNodeLiteGraph.onNodeCreated);

			// MaraScottAnyBusNodeLiteGraph.onExecuted(nodeType)
			MaraScottAnyBusNodeLiteGraph.onNodeCreated(nodeType)
			// MaraScottAnyBusNodeLiteGraph.getExtraMenuOptions(nodeType)
			MaraScottAnyBusNodeLiteGraph.onConnectionsChange(nodeType)
			// delete MaraScottAnyBusNode.beforeRegisterNodeDef;
			MaraScottAnyBusNodeLiteGraph.onRemoved(nodeType)

		}
	},
	beforeConfigureGraph(app) {
		// console.log("[MaraScott - logging " + this.name + "]", "extension beforeConfigureGraph");
		window.marascott.anyBus.init = false
	},
	afterConfigureGraph(app) {
		// console.log("[MaraScott - logging " + this.name + "]", "extension afterConfigureGraph");
		window.marascott.anyBus.init = true
	},

};

app.registerExtension(MaraScottAnyBusNodeExtension);
