import { app } from "../../scripts/app.js";
import * as shared from './helper.js'
import {
	infoLogger,
	warnLogger,
	successLogger,
	errorLogger,
	getUnique,
	removeItemAll,	
} from './helper.js'

// Definitions for litegraph.js
// Project: litegraph.js
// Definitions by: NateScarlet <https://github.com/NateScarlet>
// https://github.com/NateScarlet/litegraph.js/blob/master/src/litegraph.js

class MarasitBusNode extends LiteGraph.LGraphNode {

	title = "Bus Node - Default"
	category = "MarasIT/utils"

	// same values as the comfy note
	color = LGraphCanvas.node_colors.yellow.color
	bgcolor = LGraphCanvas.node_colors.yellow.bgcolor
	groupcolor = LGraphCanvas.node_colors.yellow.groupcolor

	bus_type = "default"

	overiding_bus_inputs = []

	inOutPuts = {
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

	constructor() {

		super()
		this.uuid = shared.makeUUID()

		infoLogger(`Constructing Bus Node instance`)

		// - litegraph settings
		this.collapsable = true
		this.isVirtualNode = true
		this.shape = LiteGraph.BOX_SHAPE
		this.serialize_widgets = true


		if (!this.properties) {
			this.properties = {
				"previousTitle": "Bus Node - " + this.bus_type
			};
		}

		// display initial inputs/outputs
		for (const name in this.inOutPuts) {
			this.setInOutPut(name, this.inOutPuts[name])
		}

		// display name widget
		this.addWidget(
			"text",
			"Constant",
			'',
			(s, t, u, v, x) => {
				// node.validateName(node.graph);
				this.title = "Bus Node - " + (this.widgets[0].value ?? this.bus_type);
				this.update();
				this.properties.previousTitle = this.widgets[0].value;
			},
			{}
		)

		// This node is purely frontend and does not impact the resulting prompt so should not be serialized
		this.isVirtualNode = true;
		console.log("Maras IT Bus Node loaded");
	}

	update = function () {

		// clean input bus
		if (this.inputs[0].name === 'bus' && this.outputs[0].name === 'bus' && this.inputs[0].links && this.outputs[0].links && this.graph.links.length > 0) {
			const bus_inputs = this.graph.links[this.inputs[0].links[0]].data;
			for (input in bus_inputs) {
				if (this.overiding_bus_inputs.includes(bus_inputs[input].name)) {
					delete bus_inputs[input];
				}
			}
			console.log(this.inputs[0], this.outputs[0])
			// this.setOutputData(this.outputs[0].name, {
			// 	...bus_inputs,
			// 	...this.inputs
			// });
			// console.log(this.outputs[0]);
			// for (output in this.outputs[0]) {
			// 	if(this.outputs[output].name !== 'bus') {
			// 		this.setOutputData(this.outputs[output].name, output);
			// 	}
			// }
		}




	}

	setInOutPut(_name, type) {
		this.addInput(_name, type)
		this.addOutput(_name, type)
	}

	// validateName = function(graph) {
	// let widgetValue = node.widgets[0].value;

	// if (widgetValue !== '') {
	// 	let tries = 0;
	// 	const existingValues = new Set();

	// 	graph._nodes.forEach(otherNode => {
	// 		if (otherNode !== this && otherNode.type === 'MarasitBusNode') {
	// 			existingValues.add(otherNode.widgets[0].value);
	// 		}
	// 	});

	// 	while (existingValues.has(widgetValue)) {
	// 		widgetValue = node.widgets[0].value + "_" + tries;
	// 		tries++;
	// 	}

	// 	node.widgets[0].value = widgetValue;
	// 	this.update();
	// }
	// }

	clone = function () {
		const cloned = MarasitBusNode.prototype.clone.apply(this);
		cloned.size = cloned.computeSize();
		return cloned;
	};

	onAdded = function (graph) {
		// this.validateName(graph);
	}

	onConnectionsChange = function (
		slotType,	//1 = input, 2 = output
		slot,
		isChangeConnect,
		link_info,
		output
	) {

		//On Disconnect
		if (!isChangeConnect) this.disconnect(slotType, slot)
		//On Connect
		if (isChangeConnect) this.connect(slotType, slot, this, link_info)

		//Update either way
		this.update();

	}

	disconnectInput = function (slot) {
		delete this.inputs[slot];
		// this.overiding_bus_inputs = removeItemAll(this.inputs[slot].name, this.overiding_bus_inputs)
	}
	disconnectOutput = function (slot) {
	}
	disconnect = function (slotType, slot) {
		if (slotType == 1) {
			this.disconnectInput(slot)
		}
		// if (slotType == 2) {
		// 	this.disconnectOutput(slot)
		// }
	}

	connectInput = function (slot, node, link_info) {
		const connecting_node = node.graph._nodes.find((otherNode) => otherNode.id == link_info.origin_id);
			
		// if origina node to connect has outputdata
		if (connecting_node && connecting_node.outputs && connecting_node.outputs[link_info.origin_slot]) {
			const link_node_data = connecting_node.outputs[link_info.origin_slot];
			this.inputs[slot] = link_node_data
			this.outputs[slot] = link_node_data
			if(this.inputs[slot].name !== 'bus' && this.overiding_bus_inputs.indexOf(this.inputs[slot].name) !== -1) {
				this.overiding_bus_inputs = this.overiding_bus_inputs.push(this.inputs[slot].name)
			}

			// if (app.ui.settings.getSettingValue("KJNodes.nodeAutoColor")){
			// 	setColorAndBgColor.call(this, type);	
			// }

		} else {
			// alert("Error: Set node input undefined. Most likely you're missing custom nodes");
		}
	}
	connectOutput = function (slot, node, link_info) {
	}
	connect = function (slotType, slot, node, link_info) {
		// input
		if (link_info && node.graph && slotType == 1) {
			this.connectInput(slot, node, link_info)
		}
		// output
		// if (link_info && node.graph && slotType == 2) {
		// 	this.connectOutput(slot, node, link_info)
		// }

	}

	onCreate() {
		errorLogger('MarasITBusNode onCreate')
	}

	onNodeCreated() {
		infoLogger('Node created', this.uuid)
	}

	onRemoved() {
		infoLogger('Node removed', this.uuid)
	}

	// getExtraMenuOptions() {
	// 	var options = []
	// 	// {
	// 	//       content: string;
	// 	//       callback?: ContextMenuEventListener;
	// 	//       /** Used as innerHTML for extra child element */
	// 	//       title?: string;
	// 	//       disabled?: boolean;
	// 	//       has_submenu?: boolean;
	// 	//       submenu?: {
	// 	//           options: ContextMenuItem[];
	// 	//       } & IContextMenuOptions;
	// 	//       className?: string;
	// 	//   }
	// 	options.push({
	// 		content: `Set to ${this.edit_mode_widget.value === 'html' ? 'markdown' : 'html'
	// 			}`,
	// 		callback: () => {
	// 			this.edit_mode_widget.value =
	// 				this.edit_mode_widget.value === 'html' ? 'markdown' : 'html'
	// 			this.updateHTML(this.html_widget.value)
	// 		},
	// 	})

	// 	return options
	// }

}

app.registerExtension({
	name: "Comfy.MarasIT.MarasitBusNode",
	registerCustomNodes() {
		LiteGraph.registerNodeType("MarasitBusNode", MarasitBusNode)
		// MarasitBusNode.title_mode = LiteGraph.NO_TITLE
	},
});
