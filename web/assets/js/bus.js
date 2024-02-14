import { app } from "../../scripts/app.js";
import { api } from '../../scripts/api.js'
import * as shared from './helper.js'
import {
	infoLogger,
	warnLogger,
	successLogger,
	errorLogger,
} from './helper.js'

if (!window.MarasIT) {
	window.MarasIT = {}
}

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

class MarasitBusNodeHelper {

	constructor() {

		return this

	}

	initNode(node) {

		node.category = "MarasIT/utils"
		// node.isVirtualNode = true;
		node.shape = LiteGraph.CARD_SHAPE // BOX_SHAPE | ROUND_SHAPE | CIRCLE_SHAPE | CARD_SHAPE
		// same values as the comfy note
		node.color = LGraphCanvas.node_colors.yellow.color
		node.bgcolor = LGraphCanvas.node_colors.yellow.bgcolor
		node.groupcolor = LGraphCanvas.node_colors.yellow.groupcolor
		if (!node.properties || !("profile" in node.properties)) {
			node.properties["profile"] = "default";
		}
		node.title = "Bus Node - " + node.properties.profile
		if (!node.properties || !("previousTitle" in node.properties)) {
			node.properties["previousTitle"] = node.title;
		}


	}

	getProfileEntries(node) {
		const entries = {
			"default" : {
				"bus": "BUS",
				"pipe": "BASIC_PIPE",
				"model": "MODEL",
				"clip": "CLIP",
				"vae": "VAE",
				"positive": "CONDITIONING",
				"negative": "CONDITIONING",
				"latent": "LATENT",
				"image": "IMAGE",
				"mask": "MASK",
				"*": "*",
			},
			"basic_pipe" : {
				"bus": "BUS",
				"pipe": "BASIC_PIPE",
				"model": "MODEL",
				"clip": "CLIP",
				"vae": "VAE",
				"positive": "CONDITIONING",
				"negative": "CONDITIONING",
			},
		}

		return entries[node.properties.profile]
	
	}

	setProfileEntries(node) {
		// display initial inputs/outputs
		const entries = MarasitBusNode.helper.getProfileEntries(node)
		for (const name in entries) {
			node.addInput(name, entries[name])
			node.addOutput(name, entries[name])
		}

	}

	setProfileWidget(node) {

		node.addWidget(
			"text",
			"Constant",
			node.properties.profile ?? '',
			(s, t, u, v, x) => {
				node.setProperty('profile', node.widgets[0].value ?? node.properties.profile)
				node.title = "Bus Node - " + node.properties.profile;
				node.setProperty('previousTitle', node.title)
			},
			{}
		)

	}

	setPipeWidget(node) {

		const isPipeWidth = () => {
			console.log('pipe')
			// app.canvas.setDirty(true)
		}
		const setPipeType = () => {
			console.log('in/out')
			// app.canvas.setDirty(true)
		}

		/**
		 * Defines a widget inside the node, it will be rendered on top of the node, you can control lots of properties
		 *
		 * @method addWidget
		 * @param {String} type the widget type (could be "number","string","combo"
		 * @param {String} name the text to show on the widget
		 * @param {String} value the default value
		 * @param {Function|String} callback function to call when it changes (optionally, it can be the name of the property to modify)
		 * @param {Object} options the object that contains special properties of this widget 
		 * @return {Object} the created widget object
		 */
		node.addWidget(
			'toggle',
			"Pipe",
			false,
			isPipeWidth,
			{"on": "Active", "off": "Inactive"}
		)
		node.addWidget(
			'toggle',
			"type",
			false,
			setPipeType,
			{"on": "Input", "off": "output"}
		)

	}


	async setEntryList(node) {

		const route = '/marasit/bus';
		try {
			await api
				.fetchApi(route, {
					method: 'POST',
					body: JSON.stringify({
						id: node._id,
						profile: node.properties.profile,
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

			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = async function () {
				const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

				MarasitBusNode.helper.initNode(this)
				MarasitBusNode.helper.setProfileWidget(this)
				MarasitBusNode.helper.setProfileEntries(this)
				// MarasitBusNode.helper.setPipeWidget(this)
				console.log(this.properties.profile, this._id)
				await MarasitBusNode.helper.setEntryList(this)

				return r;
			};

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
