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

	async getProfileEntries(node) {

		const AVAILABLE_INPUT_TYPES = {
			"bus": "BUS",
			"pipe (basic)": "BASIC_PIPE",
			"pipe (detailer)": "DETAILER_PIPE",
			"model": "MODEL",
			"clip": "CLIP",
			"vae": "VAE",
			"positive (text)": "STRING",
			"positive": "CONDITIONING",
			"negative (text)": "STRING",
			"negative": "CONDITIONING",
			"latent": "LATENT",
			"image": "IMAGE",
			"mask": "MASK",
			"*": "*",
		}
		const default_entries = {
			"default" : {
				"bus": "BUS",
				"pipe (basic)": "BASIC_PIPE",
				"model": "MODEL",
				"clip": "CLIP",
				"vae": "VAE",
				"positive (text)": "STRING",
				"positive": "CONDITIONING",
				"negative (text)": "STRING",
				"negative": "CONDITIONING",
				"latent": "LATENT",
				"image": "IMAGE",
				"mask": "MASK",
				"*": "*",
			},
			"basic_pipe" : {
				"bus": "BUS",
				"pipe (basic)": "BASIC_PIPE",
				"model": "MODEL",
				"clip": "CLIP",
				"vae": "VAE",
				"positive": "CONDITIONING",
				"negative": "CONDITIONING",
			},
		}

		const profile = node.properties.profile;
		let entries = default_entries[node.properties.profile]
		const url = `/extensions/MarasIT/profiles/profile_${profile}.json`;
		try {
			const response = await fetch(url);
			if (!response.ok) {
				console.log(`Failed to load profile entries from ${url}, switching back to default profile setup`);
			}
			entries = await response.json();
		} catch (error) {
			console.error('Error loading profile entries:', error);
		}

		return entries;
	
	}

	async setProfileEntries(node) {
		// display initial inputs/outputs
		const entries = await MarasitBusNode.helper.getProfileEntries(node)
		for (const name in entries) {
			if(node.findInputSlot(name) == -1) {
				node.addInput(name, entries[name])
				node.addOutput(name, entries[name])
			}
		}

	}

	setProfileWidget(node) {

		const widgetName = "Profile"
		const isProfileWidgetExists = !(node.widgets && node.widgets.length > 0 && node.widgets.every(widget => widget.name !== widgetName))
		if(!node.widgets || !isProfileWidgetExists) {
			node.addWidget(
				"text",
				widgetName,
				node.properties.profile ?? '',
				(s, t, u, v, x) => {
					node.setProperty('profile', node.widgets[0].value ?? node.properties.profile)
					node.title = "Bus Node - " + node.properties.profile;
					node.setProperty('previousTitle', node.title)
				},
				{}
			)
		}

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
		let inputs = [];
		try {
			if(node.inputs && node.inputs.length > 0 && typeof node.properties.uuid != 'undefined') {
				inputs = node.inputs.reduce((acc, input) => {				
					// Add the input name and type to the accumulator object
					acc[input.name] = input.type;
				
					return acc;
				}, {});
			}
			const params = {
				session_id: 'unique',
				node_id: node.id,
				profile: node.properties.profile,
				inputs: inputs,
			};
			console.log(params)
			await api
				.fetchApi(route, {
					method: 'POST',
					body: JSON.stringify(params),
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

	onExecuted(nodeType) {
		const onExecuted = nodeType.prototype.onExecuted
		nodeType.prototype.onExecuted = function (message) {
			onExecuted?.apply(this, arguments)
			console.log("[logging "+this.name+"]", "on Executed", {"id": this.id, "properties": this.properties});
		}

	}

	onNodeCreated(nodeType) {

		const onNodeCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = async function () {
			const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

			
			MarasitBusNode.helper.initNode(this)
			MarasitBusNode.helper.setProfileWidget(this)
			await MarasitBusNode.helper.setProfileEntries(this)
			await MarasitBusNode.helper.setEntryList(this)

			return r;
		}

	}

	async addInputMenuItem(_this, _, options) {
		for (let _index in _.graph._nodes) {
			let _node = _.graph._nodes[_index]
			if (_node.type === "MarasitBusNode" && _this.title === _node.title) {
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

	async removeLastInputMenuItem(_this, _, options) {
		for (let _index in _.graph._nodes) {
			let _node = _.graph._nodes[_index]
			if (_node.type === "MarasitBusNode" && _this.title === _node.title) {
				const inputLenth = _node.inputs.length - 1
				const outputLenth = _node.outputs.length - 1
				const name = _node.inputs[inputLenth].name

				_node.removeInput(inputLenth);
				_node.removeOutput(outputLenth);

				console.log('- entry ' + name);
				await MarasitBusNode.helper.setEntryList(_node)

			}
		}
	}

	getExtraMenuOptions(nodeType) {
		const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
		nodeType.prototype.getExtraMenuOptions = function (_, options) {

			console.log("[logging "+this.name+"]", "on Extra Menu Options", {"id": this.id, "properties": this.properties});

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

			// Add input callback
			const addInputCallback = async () => {
				await MarasitBusNode.helper.addInputMenuItem(this, _, options);
			};
			// Remove input callback
			const removeLastInputCallback = async () => {
				await MarasitBusNode.helper.removeLastInputMenuItem(this, _, options);
			};

			options.unshift(
				{
					content: "Add Input",
					callback: addInputCallback
				},
				{
					content: "Remove Last Input",
					callback: removeLastInputCallback
				},
			);
			// return getExtraMenuOptions?.apply(this, arguments);
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
	 // !TODO should I find a way to define defs based on profile ?
	addCustomNodeDefs(defs, app) {
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
	async registerCustomNodes(app) {
		// Register any custom node implementations here allowing for more flexability than a custom node def
		console.log("[logging "+this.name+"]", "register custom nodes");
	},
	async setup(app) {
		// Any setup to run after the app is created
		// console.log("[logging "+this.name+"]", "extension setup");
	},
	async loadedGraphNode(node, app) {
		// Fires for each node when loading/dragging/etc a workflow json or png
		// If you break something in the backend and want to patch workflows in the frontend
		// This is the place to do this
		if(node.type == "MarasitBusNode") {

			node.setProperty('uuid', node.id)

			console.log("[logging "+this.name+"]", "Loaded Graph", {"id": node.id, "properties": node.properties});
			MarasitBusNode.helper.initNode(node)
			MarasitBusNode.helper.setProfileWidget(node)
			await MarasitBusNode.helper.setProfileEntries(node)
			// MarasitBusNode.helper.setPipeWidget(node)
			await MarasitBusNode.helper.setEntryList(node)

		}

		// This fires for every node on each load so only log once
		// delete MarasitBusNode.loadedGraphNode;
	},
	// this is the python node created
	nodeCreated(node, app) {
		// Fires every time a node is constructed
		// You can modify widgets/add handlers/etc here
		// console.log("[logging "+this.name+"]", "node created: ", {...node});

		// This fires for every node so only log once
		// delete MarasitBusNode.nodeCreated;
	},
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		// Run custom logic before a node definition is registered with the graph
		
		if (nodeData.name === 'MarasitBusNode') {
			console.log("[logging "+this.name+"]", "before register node: ", nodeData);
			// This fires for every node definition so only log once

			MarasitBusNode.helper.onExecuted(nodeType)
			MarasitBusNode.helper.onNodeCreated(nodeType)
			MarasitBusNode.helper.getExtraMenuOptions(nodeType)



			// delete MarasitBusNode.beforeRegisterNodeDef;
		}
	}
};

app.registerExtension(MarasitBusNode);
