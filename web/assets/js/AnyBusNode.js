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

	constructor() {

		this.syncNodeProfile = false

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

	}

	setProfileWidget(node) {

		const widgetName = "Profile"
		const isProfileWidgetExists = !(node.widgets && node.widgets.length > 0 && node.widgets.every(widget => widget.name !== widgetName))
		if (!node.widgets || !isProfileWidgetExists) {
			node.addWidget(
				"text",
				widgetName,
				node.properties.profile ?? '',
				(s, t, u, v, x) => {
					// do something
				},
				{}
			)
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
		nodeType.prototype.onNodeCreated = async function () {
			const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

			// console.log('onNodeCreated')
			MarasitAnyBusNode.LGraph.initNode(this)
			// MarasitAnyBusNode.LGraph.setProfileWidget(this)

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

	syncNodeProfile() {

		for (let i in this.graph._nodes) {
			let profile_node = this.graph._nodes[i]
			if (profile_node.type == "MarasitAnyBusNode") {
				for (let _slot = 1; _slot < this.inputs.length; _slot++) {
					profile_node.inputs[_slot].name = this.inputs[_slot].name
					profile_node.inputs[_slot].type = this.inputs[_slot].type
					profile_node.outputs[_slot].name = profile_node.inputs[_slot].name
					profile_node.outputs[_slot].type = profile_node.inputs[_slot].type
				}
			}
		}
		this.syncNodeProfile = false;

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

			const busIndex = 0;
			const firstAnyIndex = 1;
			const AnyIndexLabel = slot + 1 - firstAnyIndex
			let syncNodeProfile = false
			//On Disconnect
			if (!isChangeConnect && slotType == 1 && typeof link_info != 'undefined') {

				if (slot < firstAnyIndex) {
					// bus
					if (slot == 0 && this.inputs) {
						for (let _slot = firstAnyIndex; _slot < this.inputs.length; _slot++) {
							if (this.inputs[_slot].link == null) {

								this.inputs[_slot].name = "* " + (_slot + 1 - firstAnyIndex).toString().padStart(2, '0')
								this.inputs[_slot].type = "*"
								this.outputs[_slot].name = this.inputs[_slot].name
								this.outputs[_slot].type = this.inputs[_slot].type

							}
						}
					}

				} else {

					this.inputs[slot].name = "* " + AnyIndexLabel.toString().padStart(2, '0')
					this.inputs[slot].type = "*"
					this.outputs[slot].name = this.inputs[slot].name
					this.outputs[slot].type = this.inputs[slot].type

					this.syncNodeProfile = true

				}

			}
			if (!isChangeConnect && slotType == 2 && typeof link_info != 'undefined') {
				// do something
				// this.inputs[slot].name = "* ("+slot.toString().padStart(2, '0')+")"
			}
			//On Connect
			if (isChangeConnect && slotType == 1 && typeof link_info != 'undefined' && this.graph) {
				// do something
				const link_info_node = this.graph._nodes.find(
					(otherNode) => otherNode.id == link_info.origin_id
				)

				if (slot < firstAnyIndex) {
					// bus
					if (slot == 0 && link_info_node.outputs && link_info_node.type == "MarasitAnyBusNode") {
						for (let _slot = firstAnyIndex; _slot < link_info_node.outputs.length; _slot++) {
							if (link_info_node.outputs[_slot].type != this.inputs[_slot].type) {
								this.disconnectInput(_slot)
								this.disconnectOutput(_slot)
							}
							this.inputs[_slot].name = link_info_node.outputs[_slot].name
							this.inputs[_slot].type = link_info_node.outputs[_slot].type
							this.outputs[_slot].name = this.inputs[_slot].name
							this.outputs[_slot].type = this.inputs[_slot].type

						}
					} else {
						this.disconnectInput(slot)
					}

				} else {

					const anyPrefix = "* " + AnyIndexLabel.toString().padStart(2, '0')
					const origin_name = link_info_node.outputs[link_info.origin_slot].name
					let newName = origin_name
					if (origin_name.indexOf(anyPrefix) === -1 ) {
						newName = anyPrefix + " - " + origin_name
					}
					this.inputs[slot].name = newName
					this.inputs[slot].type = link_info_node.outputs[link_info.origin_slot].type
					this.outputs[slot].name = this.inputs[slot].name
					this.outputs[slot].type = this.inputs[slot].type
					this.syncNodeProfile = true

				}

			}

			if (isChangeConnect && slotType == 2 && typeof link_info != 'undefined') {
				// do something
				// this.inputs[slot].name = ":) ("+slot.toString().padStart(2, '0')+")"
			}

			return r;
		}

	}

	onRemoved(nodeType) {
		const onRemoved = nodeType.prototype.onRemoved;
		nodeType.prototype.onRemoved = function () {
			this.syncNodeProfile = false;
			console.log(this)
			onRemoved?.apply(this, arguments);
		};		
	}


}

const MarasitAnyBusNode = {
	// Unique name for the extension
	name: "Comfy.MarasIT.AnyBusNode",
	LGraph: new MarasitAnyBusNodeLiteGraph(),
	async init(app) {
		// Any initial setup to run as soon as the page loads
		// console.log("[MarasIT - logging "+this.name+"]", "extension init");
	},
	// !TODO should I find a way to define defs based on profile ?
	addCustomNodeDefs(defs, app) {
		// Add custom node definitions
		// These definitions will be configured and registered automatically
		// defs is a lookup core nodes, add yours into this
		// console.log("[MarasIT - logging "+this.name+"]", "add custom node definitions", "current nodes:", defs['MarasitAnyBusNode'],JSON.stringify(Object.keys(defs)));
	},
	async getCustomWidgets(app) {
		// Return custom widget types
		// See ComfyWidgets for widget examples
		// console.log("[MarasIT - logging "+this.name+"]", "provide custom widgets");
	},
	async registerCustomNodes(app) {
		// Register any custom node implementations here allowing for more flexability than a custom node def
		// console.log("[MarasIT - logging "+this.name+"]", "register custom nodes");
	},
	async setup(app) {
		// Any setup to run after the app is created
		// console.log("[MarasIT - logging "+this.name+"]", "extension setup");
	},
	async loadedGraphNode(node, app) {
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
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
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
			if(MarasitAnyBusNode.LGraph.syncNodeProfile) {
				MarasitAnyBusNode.LGraph.syncNodeProfile()				
			}

		}
	},

};

app.registerExtension(MarasitAnyBusNode);
