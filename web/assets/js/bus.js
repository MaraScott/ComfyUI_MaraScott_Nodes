import { app } from "../../scripts/app.js";
import { api } from '../../scripts/api.js'
import * as shared from './helper.js'
import {
	infoLogger,
	warnLogger,
	successLogger,
	errorLogger,
} from './helper.js'

/*
 * Definitions for litegraph.js
 * Project: litegraph.js
 * Definitions by: NateScarlet <https://github.com/NateScarlet>
 * https://github.com/NateScarlet/litegraph.js/blob/master/src/litegraph.js
 * ComfyUI\web\lib\litegraph.core.js
 * ComfyUI\web\extensions\logging.js.example
 * ComfyUI\custom_nodes\rgthree-comfy\src_web\typings\litegraph.d.ts
 *
 */

class MarasitBusLGraphNode extends LiteGraph.LGraphNode {

	title = "Bus Node - default"

	category = "MarasIT/utils"

	isVirtualNode = true;

	// same values as the comfy note
	color = LGraphCanvas.node_colors.yellow.color
	bgcolor = LGraphCanvas.node_colors.yellow.bgcolor
	groupcolor = LGraphCanvas.node_colors.yellow.groupcolor

	_bus_type = "default"

	_inputs_name = []

	_entries = {
		"bus": "BUS",
		"model": "MODEL",
		"clip": "CLIP",
		"vae": "VAE",
		"positive": "CONDITIONING",
		"negative": "CONDITIONING",
		"latent": "LATENT",
		"image": "IMAGE",
		"mask": "MASK",
		"*": "*",
	}

	/**
	 * Manage initialization
	 */

	constructor() {

		super()

		this._init()

	}

	_init = function () {

		this.shape = LiteGraph.CARD_SHAPE // BOX_SHAPE | ROUND_SHAPE | CIRCLE_SHAPE | CARD_SHAPE

		if (!this.properties) {
			this.properties = {
				"uuid": shared.makeUUID(),
				"busType": this._bus_type,
				"previousTitle": "Bus Node - " + this._bus_type
			};
		}

		this._init_inputs();
		this._init_widgets();

	}

	_init_inputs = function () {
		// display initial inputs/outputs
		for (const name in this._entries) {
			this._set_Entries(name, this._entries[name])
		}
	}

	_set_Entries(_name, type) {
		this.addInput(_name, type)
		this.addOutput(_name, type)
	}

	_init_widgets = function () {
		// display name widget
		this.addWidget(
			"text",
			"Constant",
			this.properties.busType ?? '',
			(s, t, u, v, x) => {
				// node.validateName(node.graph);
				this.properties.busType = this.widgets[0].value ?? this._bus_type;
				this.title = "Bus Node - " + this.properties.busType;
				this.properties.previousTitle = this.title;
			},
			{}
		)
	}

	/**
	 * Manage connection
	 */

	onConnectionsChange = function (
		slotType,	//1 = input, 2 = output
		slot,
		isChangeConnect,
		link_info,
		output
	) {
		if (this.inputs[slot].name === 'bus' && isChangeConnect == 1 && this.graph && link_info) {

			const origin_node = this.graph._nodes.find((otherNode) => otherNode.id == link_info.origin_id);

			if (typeof origin_node.inputs != 'undefined') {

				// assign origin node input values in outputs
				for (const index in this.inputs) {
					if (this.inputs[index].name != 'bus' && this.inputs[index].link == null && origin_node.inputs[index].link != null) {
						// this.inputs[index] = origin_node.inputs[index];
					}
				}

				// console.log({
				// 	id: this.id,
				// 	slot: slot,
				// 	type: slotType == 1 ? 'input':'output',	//1 = input, 2 = output
				// 	inputs: this.inputs,
				// 	outputs: this.outputs,
				// 	link: link_info,
				// 	graph: this.graph
				// })

			}

		}

		// clean input bus
		if (this.inputs[0].name === 'bus' && this.inputs[0].link) {
			// const origin_node = this.graph._nodes.find((otherNode) => otherNode.id == this.inputs[0].link);
			// console.log('in', this.title, this.graph._nodes, this.inputs[0].link, origin_node)
			// 	const bus_inputs = this.graph.links[this.inputs[0].links[0]].data;
			// 	for (input in bus_inputs) {
			// 		if (this._inputs_name.includes(bus_inputs[input].name)) {
			// 			delete bus_inputs[input];
			// 		}
			// 	}
		}
		if (this.outputs[0].name === 'bus') {
			// console.log('out', this.title, this.outputs[0])
		}

	}

}

class MarasitBusNodeHelper {

	constructor() {

		this.uuid = shared.makeUUID()
		if (!window.MarasIT) {
			window.MarasIT = {}
		}

		return this

	}


	async setEntryList(node) {

		const route = '/marasit/bus';
		try {
			await api
				.fetchApi(route, {
					method: 'POST',
					body: JSON.stringify({
						inputs: node.inputs.map(input => input.name),
					}),
				})
				.then((response) => { 
					if (!response.ok) {
						throw new Error('Network response was not ok');
					}
					return response.json()
				})
				.then((data) => { 
					console.log(route, data.message)
				})
				.catch((error) => {
					console.error('Error:', error)
				})
		} catch (error) {
			console.error('Error:', error)
		}
	}

}

const MarasitBusNode = {
	// Unique name for the extension
	name: "Comfy.MarasIT.MarasitBusNode",
	helper: new MarasitBusNodeHelper(),
	async init(app) {
		// Any initial setup to run as soon as the page loads
		// console.log("[logging "+this.name+"]", "extension init");
	},
	async setup(app) {
		// Any setup to run after the app is created
		// console.log("[logging "+this.name+"]", "extension setup");
	},
	async addCustomNodeDefs(defs, app) {
		// Add custom node definitions
		// These definitions will be configured and registered automatically
		// defs is a lookup core nodes, add yours into this
		// console.log("[logging "+this.name+"]", "add custom node definitions", "current nodes:", Object.keys(defs));
	},
	async getCustomWidgets(app) {
		// Return custom widget types
		// See ComfyWidgets for widget examples
		// console.log("[logging "+this.name+"]", "provide custom widgets");
	},
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		// Run custom logic before a node definition is registered with the graph
		if (nodeData.name === 'MarasitBusNode') {
			// console.log("[logging "+this.name+"]", "before register node: ", nodeData, nodeType);
			// This fires for every node definition so only log once

			const onExecuted = nodeType.prototype.onExecuted
			nodeType.prototype.onExecuted = function (message) {
				onExecuted?.apply(this, arguments)
				// console.log({arguments: arguments, message: message})
			}

			/*
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

				if (!this.properties || !("uuid" in this.properties)) {
					this.addProperty("uuid", shared.makeUUID(), "string");
				}

				return r;
			};			
			*/

			const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
			nodeType.prototype.getExtraMenuOptions = function (_, options) {
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
				options.unshift(
					{
						content: "Add Input",
						callback: async () => {
							for (let _index in _.graph._nodes) {
								let _node = _.graph._nodes[_index]
								if (_node.type === "MarasitBusNode" && this.title === _node.title) {
									_node.index = _node.inputs.length + 1
									const name = "any_" + _node.index;
									const type = "*";
									_node.addInput(name, type);
									_node.addOutput(name, type);

									const inputLenth = _node.inputs.length - 1;
									const outputLenth = _node.outputs.length - 1;
									// const index = _node.widgets[_node.index].value;

									for (let i = inputLenth; i > _node.index + 1; i--) {
										swapInputs(_node, i, i - 1);
										swapOutputs(_node, i, i - 1);
									}

									// renameNodeInputs(_node, name);
									// renameNodeOutputs(_node, name);

									// _node.properties["values"].splice(_node.index+1, 0, [0, 0, 0, 0, 1]);
									// _node.widgets[_node.index].options.max = inputLenth;

									// _node.setDirtyCanvas(true);
									console.log('+ entry ' + name);
									await MarasitBusNode.helper.setEntryList(_node)

								}
							}
						}
					},
					{
						content: "Remove Last Input",
						callback: async () => {

							for (let _index in _.graph._nodes) {
								let _node = _.graph._nodes[_index]
								if (_node.type === "MarasitBusNode" && this.title === _node.title) {
									const inputLenth = _node.inputs.length - 1
									const outputLenth = _node.outputs.length - 1
									const name = _node.inputs[inputLenth].name

									_node.removeInput(inputLenth);
									_node.removeOutput(outputLenth);

									console.log('- entry ' + name);
									MarasitBusNode.helper.setEntryList(_node)

								}
							}
						}
					},
				);
				// return getExtraMenuOptions?.apply(this, arguments);
			}

			// delete MarasitBusNode.beforeRegisterNodeDef;
		}
	},
	async registerCustomNodes(app) {
		// Register any custom node implementations here allowing for more flexability than a custom node def
		// console.log("[logging "+this.name+"]", "register custom nodes");

		// LiteGraph.registerNodeType("MarasitBusNode-js", Object.assign(MarasitBusLGraphNode, {
		// 	// title_mode: LiteGraph.NO_TITLE,
		// 	title: "Bus Node (js)",
		// 	collapsable: false,
		// }))
	},
	loadedGraphNode(node, app) {
		// Fires for each node when loading/dragging/etc a workflow json or png
		// If you break something in the backend and want to patch workflows in the frontend
		// This is the place to do this
		// console.log("[logging "+this.name+"]", "loaded graph node: ", node);

		// This fires for every node on each load so only log once
		delete MarasitBusNode.loadedGraphNode;
	},
	nodeCreated(node, app) {
		// Fires every time a node is constructed
		// You can modify widgets/add handlers/etc here
		// console.log("[logging "+this.name+"]", "node created: ", node);

		// This fires for every node so only log once
		delete MarasitBusNode.nodeCreated;
	}
};

app.registerExtension(MarasitBusNode);
