import { app } from "../../scripts/app.js";
import { api } from '../../scripts/api.js'
import * as shared from './helper.js'
import {
	infoLogger,
	warnLogger,
	successLogger,
	errorLogger,
} from './helper.js'

if (!window.marasit) {
	window.marasit = {}
}

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

class MarasitAnyBusNodeLiteGraph {

	NOSYNC = 0
	FULLSYNC = 1
	BACKWARDSYNC = 3
	FORWARDSYNC = 4

	DEFAULT_PROFILE = 'undefined'
	PROFILE_NAME = 'Profile'
	DEFAULT_QTY = 5
	MIN_QTY = 3
	MAX_QTY = 15
	QTY_NAME = "Nb Inputs"

	FIRST_ANY_INDEX = 1

	// BUS_SLOT = 0
	BASIC_PIPE_SLOT = 0
	REFINER_PIPE_SLOT = 0

	ALLOWED_REROUTE_TYPE = [
		"Reroute",
	]
	ALLOWED_NODE_TYPE = [
		"MarasitAnyBusNode",
		...this.ALLOWED_REROUTE_TYPE,
	]

	constructor() {

		this.syncProfile = this.NOSYNC
		this.busNodeForSync = null
		this.firstAnyIndex = this.FIRST_ANY_INDEX;
		this.AnyIndexLabel = 0
		this.isAnyBusNodeSetup = false

		return this

	}

	initNode(node) {

		// node.category = "marasit/utils"
		// node.isVirtualNode = true;
		node.shape = LiteGraph.CARD_SHAPE // BOX_SHAPE | ROUND_SHAPE | CIRCLE_SHAPE | CARD_SHAPE
		// same values as the comfy note
		node.color = LGraphCanvas.node_colors.green.color
		node.bgcolor = LGraphCanvas.node_colors.green.bgcolor
		node.groupcolor = LGraphCanvas.node_colors.green.groupcolor
		node.groupcolor = LGraphCanvas.node_colors.green.groupcolor
		node.size[0] = 150 // width
		if (!node.properties || !(this.PROFILE_NAME in node.properties)) {
			node.properties[this.PROFILE_NAME] = node.DEFAULT_PROFILE;
		}
		node.title = "AnyBus - " + node.properties[this.PROFILE_NAME]

	}

	getNodeWidgetByName(node, name) {
		return node.widgets?.find((w) => w.name === name);
	}
	
	setInputValue(node) {
	
		let protected_slots = []

		for (let slot = this.FIRST_ANY_INDEX; slot < MarasitAnyBusNode.LGraph.busNodeForSync.inputs.length; slot++) {
			if(protected_slots.indexOf(slot) > -1) continue
			const isNodeInputDifferent = node.inputs[slot].type != "*" && node.inputs[slot].type != MarasitAnyBusNode.LGraph.busNodeForSync.inputs[slot].type
			if(isNodeInputDifferent) {
				const preSyncMode = MarasitAnyBusNode.LGraph.syncProfile;
				MarasitAnyBusNode.LGraph.syncProfile = this.NOSYNC;
				if (node.inputs[slot].link == null) {
					node.disconnectInput(slot)
					node.disconnectOutput(slot)
				} else {
					protected_slots.push(node.id)
				}
				MarasitAnyBusNode.LGraph.syncProfile = this.preSyncMode;
			} 
			if (MarasitAnyBusNode.LGraph.busNodeForSync.id != node.id) {
				if (node.inputs[slot].link == null) {
					node.inputs[slot].name = MarasitAnyBusNode.LGraph.busNodeForSync.inputs[slot].name.toLowerCase()
					node.inputs[slot].type = MarasitAnyBusNode.LGraph.busNodeForSync.inputs[slot].type
					node.outputs[slot].name = node.inputs[slot].name
					node.outputs[slot].type = node.inputs[slot].type
				}
			}
		}

	}
	
	setWidgetValue(node, name, value) {
		const nodeWidget = this.getNodeWidgetByName(node, name);
		nodeWidget.value = value
		node.setProperty(name, nodeWidget.value ?? node.properties[name])
		if(name == this.PROFILE_NAME) {
			node.title = "AnyBus - " + node.properties[name];
		}
		if(name == this.QTY_NAME) {
			let qty = 0
			let _value = value + MarasitAnyBusNode.LGraph.firstAnyIndex
			if(node.inputs.length > _value) {
				qty = node.inputs.length - _value
				for (let i = qty; i > 0; i--) {
					node.removeInput(node.inputs.length-1)
					node.removeOutput(node.outputs.length-1)
				}
			} else if(node.inputs.length < _value) {
				qty = _value - node.inputs.length
				for (let i = 0; i < qty; i++) {
					const name = "* " + node.inputs.length.toString().padStart(2, '0')
					const type = "*"
					node.addInput(name, type)
					node.addOutput(name, type)
				}
			}

		}
		node.setDirtyCanvas(true)
	}	

	setBusWidgets(node) {

		let nodeWidget = this.getNodeWidgetByName(node, this.QTY_NAME);
		if (nodeWidget == undefined) {

			let values = []

			for (let i = this.MIN_QTY; i <= this.MAX_QTY; i++) {
				values.push(i);
			}

			node.addWidget(
				"combo",
				this.QTY_NAME,
				node.properties[this.QTY_NAME] ?? this.DEFAULT_QTY,
				(value, LGraphCanvas, Node, Coordinate, PointerEvent) => {
					this.setWidgetValue(node, this.QTY_NAME, value)
					MarasitAnyBusNode.LGraph.syncProfile = this.FULLSYNC;
					this.syncNodeProfile(node, this.QTY_NAME, null)
				},
				{
					"values": values
				}
			)
			node.setProperty(this.QTY_NAME, this.DEFAULT_QTY)
			this.setWidgetValue(node, this.QTY_NAME, this.DEFAULT_QTY)
		}

		nodeWidget = this.getNodeWidgetByName(node, this.PROFILE_NAME);
		if (nodeWidget == undefined) {
			node.addWidget(
				"text",
				this.PROFILE_NAME,
				node.properties[this.PROFILE_NAME] ?? this.DEFAULT_PROFILE,
				(value, LGraphCanvas, Node, Coordinate, PointerEvent) => {
					this.setWidgetValue(node, this.PROFILE_NAME, value)
					MarasitAnyBusNode.LGraph.syncProfile = this.FULLSYNC;
					this.syncNodeProfile(node, this.PROFILE_NAME, null)
				},
				{}
			)
			node.setProperty(this.PROFILE_NAME, this.DEFAULT_PROFILE)
			this.setWidgetValue(node, this.PROFILE_NAME, this.DEFAULT_PROFILE)
		}

	}

	onExecuted(nodeType) {
		const onExecuted = nodeType.prototype.onExecuted
		nodeType.prototype.onExecuted = function (message) {
			onExecuted?.apply(this, arguments)
			// console.log("[MarasIT - logging "+this.name+"]", "on Executed", {"id": this.id, "properties": this.properties});
		}

	}

	onNodeCreated(nodeType) {

		const onNodeCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function () {
			const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

			// console.log('onNodeCreated')
			MarasitAnyBusNode.LGraph.initNode(this)
			MarasitAnyBusNode.LGraph.setBusWidgets(this)

			return r;
		}

	}

	getExtraMenuOptions(nodeType) {
		const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
		nodeType.prototype.getExtraMenuOptions = function (_, options) {

			// console.log("[MarasIT - logging "+this.name+"]", "on Extra Menu Options", {"id": this.id, "properties": this.properties});

			// var options = []
			// {
			//       content: string;
			//       callback?: ContextMenuEventListener;
			//       /** Used as innerHTML for extra child element */
			//       title?: string;
			//       disabled?: boolean;
			//       has_submenu?: boolean;
			//       submenu?: {
			//           options: ContextMenuItem[];
			//       } & IContextMenuOptions;
			//       className?: string;
			//   }

			// Add callback
			const callback = () => {
				// do something
			};

			// options.unshift(
			// 	{
			// 		content: "Add Input",
			// 		callback: callback
			// 	},
			// );
			// return getExtraMenuOptions?.apply(this, arguments);
		}
	}

	setBusFlows(node) {

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
				if( _backward_bus_node_link) backward_bus_node_connections[backward_bus_nodes[i].id] = _backward_bus_node_link.origin_id
			}
		}

		let currentNode = node.id
		const backward_path = [currentNode]; // Initialize the path with the starting node
		while (backward_bus_node_connections[currentNode] !== undefined) {
			currentNode = backward_bus_node_connections[currentNode]; // Move to the parent node
			const _currentNode = node.graph.getNodeById(currentNode)
			if(_currentNode.type == "MarasitAnyBusNode") {
				backward_path.push(currentNode); // Add the parent node to the path
			}
		}

		return backward_path;

	}

	getFlowsLastBuses(nodes) {
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

	getBusFlows(node) {

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
					if(_previousNode_link) _previousNode_id = _previousNode_link.origin_id
				}

				nodes_paths[_node.id] = _previousNode_id
			}
		}
		const lastBuseNodeIds = this.getFlowsLastBuses(nodes_paths)
		for (let i in lastBuseNodeIds) {
			let _node = node.graph.getNodeById(lastBuseNodeIds[i])
			bus_flows[lastBuseNodeIds[i]] = this.setBusFlows(_node)
		}

		return bus_flows
	}

	getInputBusFlow(node, slot) {

		let _backward_node = []
		let _backward_node_list = []
		for (let i in node.graph._nodes) {
			let _node = node.graph._nodes[i]
			if (_node.type == "MarasitAnyBusNode" && _backward_node_list.indexOf(_node.id) == -1) {
				_backward_node_list.push(_node.id);
				if (_node.inputs[slot].link != null) _backward_node.push(_node);
			}
		}

		// bus network
		let _backward_bus_node_link = null
		let backward_bus_nodes = []
		let backward_bus_node_connections = {}
		for (let i in _backward_node_list) {
			backward_bus_nodes.push(node.graph._nodes.find(
				(otherNode) => otherNode.id == _backward_node_list[i]
			))
		}
		for (let i in backward_bus_nodes) {
			_backward_bus_node_link = backward_bus_nodes[i].inputs[0].link
			if (_backward_bus_node_link != null) {
				_backward_bus_node_link = node.graph.links.find(
					(otherLink) => otherLink?.id == _backward_bus_node_link
				)
				backward_bus_node_connections[backward_bus_nodes[i].id] = _backward_bus_node_link.origin_id
			}
		}

		let currentNode = node.id
		const backward_path = [currentNode]; // Initialize the path with the starting node
		while (backward_bus_node_connections[currentNode] !== undefined) {
			currentNode = backward_bus_node_connections[currentNode]; // Move to the parent node
			backward_path.push(currentNode); // Add the parent node to the path
		}

		return backward_path;

	}

	getOriginRerouteBusType(node) {
		let originNode = null
		let _originNode = null
		let isMarasitAnyBusNode = false

		if(node.inputs[0].link != null) {

			const __originLink = node.graph.links.find(
				(otherLink) => otherLink?.id == node.inputs[0].link
			)
			_originNode = node.graph.getNodeById(__originLink.origin_id)

			if (_originNode.type == 'Reroute' && _originNode?.__outputType == 'BUS') {
				_originNode = MarasitAnyBusNode.LGraph.getOriginRerouteBusType(_originNode) 
			}
			if (_originNode?.type == "MarasitAnyBusNode") {
				originNode = _originNode
			}

		}

		return originNode

	}

	getBusParentNodeWithInput(node, slot) {
		let parentNode = null

		if(node.inputs[0].link != null) {

			const parentLink = node.graph.links.find(
				(otherLink) => otherLink?.id == node.inputs[0].link
			)
			parentNode = node.graph.getNodeById(parentLink.origin_id)

			if(parentNode.inputs[slot].link == null) {
				parentNode = this.getBusParentNodeWithInput(parentNode, slot)
			}

		}

		if(parentNode != null) {
			node.inputs[slot].name = parentNode.inputs[slot].name
			node.inputs[slot].type = parentNode.inputs[slot].type
		} else {
			node.inputs[slot].name = "* " + MarasitAnyBusNode.LGraph.AnyIndexLabel.toString().padStart(2, '0')
			node.inputs[slot].type = "*"
		}

		node.outputs[slot].name = node.inputs[slot].name
		node.outputs[slot].type = node.inputs[slot].type

		return parentNode
	}

	syncBusNodes(node, busNodes, isChangeWidget, isChangeConnect) {

		MarasitAnyBusNode.LGraph.busNodeForSync = node;

		for(let i in busNodes) {
			let _node = node.graph.getNodeById(busNodes[i])
			if(_node.id !== node.id && this.ALLOWED_REROUTE_TYPE.indexOf(_node.type) == -1) {
				if(isChangeWidget != null) this.setWidgetValue(_node, isChangeWidget, node.properties[isChangeWidget])
				if(isChangeConnect !== null) this.setInputValue(_node)
			}
		}
	}

	syncNodeProfile(node, isChangeWidget, isChangeConnect) {

		if (!node.graph || MarasitAnyBusNode.LGraph.syncProfile == MarasitAnyBusNode.LGraph.NOSYNC) return
	
		const profile = node.properties[MarasitAnyBusNode.LGraph.PROFILE_NAME]
		let busNodes = []
		const busNodePaths = this.getBusFlows(node)
		for(let i in busNodePaths) {
			if(busNodePaths[i].indexOf(node.id) > -1) busNodes = busNodePaths[i] 
		}

		let startIndex = null;

		if(MarasitAnyBusNode.LGraph.syncProfile == MarasitAnyBusNode.LGraph.BACKWARDSYNC) {
			
			startIndex = busNodes.indexOf(node.id) + 1;
			busNodes = busNodes.slice(startIndex)
			
		}
		
		if(MarasitAnyBusNode.LGraph.syncProfile == MarasitAnyBusNode.LGraph.FORWARDSYNC) {
			
			busNodes?.reverse()
			startIndex = busNodes.indexOf(node.id) + 1;
			busNodes = busNodes.slice(startIndex)

		}

		if(MarasitAnyBusNode.LGraph.syncProfile == MarasitAnyBusNode.LGraph.FULLSYNC) {
					
			busNodes?.reverse()
			startIndex = 0

		}

		if(startIndex != null) this.syncBusNodes(node, busNodes, isChangeWidget, isChangeConnect)

		MarasitAnyBusNode.LGraph.syncProfile = MarasitAnyBusNode.LGraph.NOSYNC
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

		let syncType = MarasitAnyBusNode.LGraph.NOSYNC 
		if(isForward) syncType = MarasitAnyBusNode.LGraph.FORWARDSYNC
		if(isBackward) syncType = MarasitAnyBusNode.LGraph.BACKWARDSYNC
		if(isFull) syncType = MarasitAnyBusNode.LGraph.FULLSYNC

		return syncType

	}

	onConnectionsChange(nodeType) {

		const onConnectionsChange = nodeType.prototype.onConnectionsChange
		nodeType.prototype.onConnectionsChange = function (
			slotType,	//1 = input, 2 = output
			slot,
			isChangeConnect,
			link_info,
			output
		) {
			const r = onConnectionsChange ? onConnectionsChange.apply(this, arguments) : undefined

			if(!MarasitAnyBusNode.LGraph.isAnyBusNodeSetup) return r

			MarasitAnyBusNode.LGraph.syncProfile = MarasitAnyBusNode.LGraph.NOSYNC
			MarasitAnyBusNode.LGraph.AnyIndexLabel = slot + 1 - MarasitAnyBusNode.LGraph.FIRST_ANY_INDEX
			//On Disconnect
			if (!isChangeConnect && slotType == 1 && typeof link_info != 'undefined') {
				console.log('disconnect');

				if (slot < MarasitAnyBusNode.LGraph.firstAnyIndex) {
					// bus
					if (slot == 0 && this.inputs) {
						for (let _slot = MarasitAnyBusNode.LGraph.firstAnyIndex; _slot < this.inputs.length; _slot++) {
							if (this.inputs[_slot].link == null) {

								this.inputs[_slot].name = "* " + (_slot + 1 - MarasitAnyBusNode.LGraph.firstAnyIndex).toString().padStart(2, '0')
								this.inputs[_slot].type = "*"
								this.outputs[_slot].name = this.inputs[_slot].name
								this.outputs[_slot].type = this.inputs[_slot].type

							}
						}
						MarasitAnyBusNode.LGraph.syncProfile = MarasitAnyBusNode.LGraph.FORWARDSYNC
					}

				} else {

					MarasitAnyBusNode.LGraph.syncProfile = MarasitAnyBusNode.LGraph.getSyncType(this, slot, null, null)

					const previousBusNode = MarasitAnyBusNode.LGraph.getBusParentNodeWithInput(this, slot)

					let newName = "* " + MarasitAnyBusNode.LGraph.AnyIndexLabel.toString().padStart(2, '0')
					let newType = "*"
					if(previousBusNode != null && previousBusNode.outputs[slot]) {
						newName = previousBusNode.outputs[slot].name
						newType = previousBusNode.outputs[slot].type
					}

					// input
					this.inputs[slot].name = newName
					this.inputs[slot].type = newType
					this.outputs[slot].name = this.inputs[slot].name
					this.outputs[slot].type = this.inputs[slot].type

				}

			}
			if (!isChangeConnect && slotType == 2 && typeof link_info != 'undefined') {
				// do something
				// this.inputs[slot].name = "* ("+slot.toString().padStart(2, '0')+")"
			}
			//On Connect
			if (isChangeConnect && slotType == 1 && typeof link_info != 'undefined' && this.graph) {
				console.log('connect');
				// do something
				let link_info_node = this.graph._nodes.find(
					(otherNode) => otherNode.id == link_info.origin_id
				)
				
				if (slot < MarasitAnyBusNode.LGraph.firstAnyIndex) {
					// bus
					const isBusInput = slot == 0
					const isOutputs = link_info_node.outputs?.length > 0
					let isMarasitBusNode = link_info_node.type == "MarasitAnyBusNode"
					if(!isMarasitBusNode) {
						const link_info_node_origin = MarasitAnyBusNode.LGraph.getOriginRerouteBusType(link_info_node)
						isMarasitBusNode = link_info_node_origin.type == "MarasitAnyBusNode"
						if (isMarasitBusNode) {
							link_info_node = link_info_node_origin
						}
					}
					const isOriginProfileSame = this.properties[MarasitAnyBusNode.LGraph.PROFILE_NAME] == link_info_node.properties[MarasitAnyBusNode.LGraph.PROFILE_NAME]
					const isTargetProfileDefault = this.properties[MarasitAnyBusNode.LGraph.PROFILE_NAME] == MarasitAnyBusNode.LGraph.DEFAULT_PROFILE
					if (isBusInput && isOutputs && isMarasitBusNode && (isOriginProfileSame || isTargetProfileDefault)) {
						if(isTargetProfileDefault) MarasitAnyBusNode.LGraph.setWidgetValue(this, MarasitAnyBusNode.LGraph.PROFILE_NAME, link_info_node.properties[MarasitAnyBusNode.LGraph.PROFILE_NAME])
						MarasitAnyBusNode.LGraph.setWidgetValue(this, MarasitAnyBusNode.LGraph.QTY_NAME, link_info_node.properties[MarasitAnyBusNode.LGraph.QTY_NAME])
						for (let _slot = MarasitAnyBusNode.LGraph.firstAnyIndex; _slot < link_info_node.outputs.length; _slot++) {
							if(_slot > link_info_node.properties[MarasitAnyBusNode.LGraph.QTY_NAME]) {
								this.disconnectInput(_slot)
								this.disconnectOutput(_slot)
							} else {
								if (link_info_node.outputs[_slot].type != this.inputs[_slot].type) {
									this.disconnectInput(_slot)
									this.disconnectOutput(_slot)
								}
								this.inputs[_slot].name = link_info_node.outputs[_slot].name.toLowerCase()
								this.inputs[_slot].type = link_info_node.outputs[_slot].type
								this.outputs[_slot].name = this.inputs[_slot].name
								this.outputs[_slot].type = this.inputs[_slot].type
							}
						}
					} else {
						this.disconnectInput(slot)
					}
					MarasitAnyBusNode.LGraph.syncProfile = MarasitAnyBusNode.LGraph.FULLSYNC
				} else {

					const anyPrefix = "* " + MarasitAnyBusNode.LGraph.AnyIndexLabel.toString().padStart(2, '0')
					const origin_name = link_info_node.outputs[link_info.origin_slot]?.name.toLowerCase()
					let newName = origin_name
					if (origin_name && origin_name.indexOf(anyPrefix) === -1) {
						newName = anyPrefix + " - " + origin_name
					}
					if (link_info_node.outputs[link_info.origin_slot] && this.inputs[slot].name == anyPrefix && this.inputs[slot].type == "*") {
												
						MarasitAnyBusNode.LGraph.syncProfile = MarasitAnyBusNode.LGraph.getSyncType(this, slot, link_info_node, link_info.origin_slot)

						this.inputs[slot].name = newName
						this.inputs[slot].type = link_info_node.outputs[link_info.origin_slot].type
						this.outputs[slot].name = this.inputs[slot].name
						this.outputs[slot].type = this.inputs[slot].type

					} else if (this.inputs[slot].type == link_info_node.outputs[link_info.origin_slot]?.type && this.inputs[slot].type != newName) {

						MarasitAnyBusNode.LGraph.syncProfile = MarasitAnyBusNode.LGraph.getSyncType(this, slot, link_info_node, link_info.origin_slot)

						this.inputs[slot].name = newName
						this.outputs[slot].name = this.inputs[slot].name

					}


				}

			}

			if (isChangeConnect && slotType == 2 && typeof link_info != 'undefined') {
				// do something
				// this.inputs[slot].name = ":) ("+slot.toString().padStart(2, '0')+")"
			}

			MarasitAnyBusNode.LGraph.syncNodeProfile(this, null, isChangeConnect)

			return r;
		}

	}

	onRemoved(nodeType) {
		const onRemoved = nodeType.prototype.onRemoved;
		nodeType.prototype.onRemoved = function () {
			// console.log('onRemoved')
			onRemoved?.apply(this, arguments);
		};
	}


}

const MarasitAnyBusNode = {
	// Unique name for the extension
	name: "Comfy.MarasIT.AnyBusNode",
	LGraph: new MarasitAnyBusNodeLiteGraph(),
	init(app) {
		// Any initial setup to run as soon as the page loads
		// console.log("[MarasIT - logging "+this.name+"]", "extension init");
	},
	setup(app) {
		// Any setup to run after the app is created
		// console.log("[MarasIT - logging "+this.name+"]", "extension setup");
	},
	// !TODO should I find a way to define defs based on profile ?
	addCustomNodeDefs(defs, app) {
		// Add custom node definitions
		// These definitions will be configured and registered automatically
		// defs is a lookup core nodes, add yours into this
		// console.log("[MarasIT - logging "+this.name+"]", "add custom node definitions", "current nodes:", defs['MarasitAnyBusNode'],JSON.stringify(Object.keys(defs)));
	},
	getCustomWidgets(app) {
		// Return custom widget types
		// See ComfyWidgets for widget examples
		// console.log("[MarasIT - logging "+this.name+"]", "provide custom widgets");
	},
	registerCustomNodes(app) {
		// Register any custom node implementations here allowing for more flexability than a custom node def
		// console.log("[MarasIT - logging "+this.name+"]", "register custom nodes");
	},
	loadedGraphNode(node, app) {
		// Fires for each node when loading/dragging/etc a workflow json or png
		// If you break something in the backend and want to patch workflows in the frontend
		// This is the place to do this
		if (node.type == "MarasitAnyBusNode") {

			node.setProperty('uuid', node.id)

			// console.log("[MarasIT - logging "+this.name+"]", "Loaded Graph", {"id": node.id, "properties": node.properties});

		}

		// This fires for every node on each load so only log once
		// delete MarasitAnyBusNode.loadedGraphNode;
	},
	// this is the python node created
	nodeCreated(node, app) {
		// Fires every time a node is constructed
		// You can modify widgets/add handlers/etc here
		// console.log("[MarasIT - logging "+this.name+"]", "node created: ", {...node});

		// This fires for every node so only log once
		// delete MarasitAnyBusNode.nodeCreated;
	},
	beforeRegisterNodeDef(nodeType, nodeData, app) {
		// Run custom logic before a node definition is registered with the graph

		if (nodeData.name === 'MarasitAnyBusNode') {
			// console.log("[MarasIT - logging "+this.name+"]", "before register node: ", nodeData);
			// This fires for every node definition so only log once

			// MarasitAnyBusNode.LGraph.onExecuted(nodeType)
			MarasitAnyBusNode.LGraph.onNodeCreated(nodeType)
			// MarasitAnyBusNode.LGraph.getExtraMenuOptions(nodeType)
			MarasitAnyBusNode.LGraph.onConnectionsChange(nodeType)
			// delete MarasitAnyBusNode.beforeRegisterNodeDef;
			MarasitAnyBusNode.LGraph.onRemoved(nodeType)

		}
	},
	afterConfigureGraph(app) {
		MarasitAnyBusNode.LGraph.isAnyBusNodeSetup = true
		// console.log("[MarasIT - logging "+this.name+"]", "extension afterConfigureGraph");
	},

};

app.registerExtension(MarasitAnyBusNode);
