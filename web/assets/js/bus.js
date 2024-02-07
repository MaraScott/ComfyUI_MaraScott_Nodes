import { app } from "../../../scripts/app.js";
//based on diffus3's SetGet: https://github.com/diffus3/ComfyUI-extensions

class MarasitBusNode {

	
	category = "MarasIT/utils"
	
	bus_type = "default"
	bus_data = {}
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

		if (!this.properties) {
			this.properties = {
				"previousTitle": "Bus Node - " = this.bus_type
			};
		}

		const node = this;

		for(_name, type in this.inOutPuts) {
			this.setInOutPut(_name, type)
		}

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
		
		this.inputBus = {};

		this.updateBusOutput = function() {
			// Pass the merged busData as output 'bus'
			// if (this.outputs.length > 0 && this.outputs[0].name === 'bus') {
			// 	this.setOutputData(0, this.busData);
			// }

			// // Populate new input bus into appropriate outputs (model to model, clip to clip, etc.)
			// for (let i = 1; i < this.outputs.length; i++) {
			// 	const outputName = this.outputs[i].name;
			// 	if (this.busData[outputName]) {
			// 		this.setOutputData(i, this.busData[outputName]);
			// 	}
			// }
		};

		this.clone = function () {
			const cloned = MarasitBusNode.prototype.clone.apply(this);
			cloned.size = cloned.computeSize();
			return cloned;
		};

		this.onAdded = function(graph) {
			// this.validateName(graph);
		}


		this.update = function() {
			if (!node.graph) {
				return;
			}
		}


		// This node is purely frontend and does not impact the resulting prompt so should not be serialized
		this.isVirtualNode = true;
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

	onDisconnect = function(slot) {
		if (slotType == 1) {
			// if(this.inputs[slot].name === ''){
			// 	this.inputs[slot].type = '*';
			// 	this.inputs[slot].name = '*';
			// }
			// const inputName = this.inputs[slot].name;
			// delete this.busData[inputName]; // Remove disconnected input data from busData
			
		}
		if (slotType == 2) {
			// this.outputs[slot].type = '*';
			// this.outputs[slot].name = '*';						
		}	
	}

	onConnect = function(slotType, slot, node, link_info) {
		if (link_info && node.graph && slotType == 1) {
			const fromNode = node.graph._nodes.find((otherNode) => otherNode.id == link_info.origin_id);
			
			if (fromNode && fromNode.outputs && fromNode.outputs[link_info.origin_slot]) {
				if (this.inputs[slot].name !== 'bus') { // Exclude the 'bus' input from merging
					const outputData = fromNode.outputs[link_info.origin_slot].data;
					this.busData[inputName] = outputData; // Merge data into busData
				}

			// 	const type = fromNode.outputs[link_info.origin_slot].type;
			
			// 	if (this.title === "Set"){
			// 		this.title = "Set_" + type;	
			// 	}
			// 	if (this.widgets[0].value === '*'){
			// 		this.widgets[0].value = type	
			// 	}
				
			// 	this.validateName(node.graph);
			// 	this.inputs[0].type = type;
			// 	this.inputs[0].name = type;
				
				// if (app.ui.settings.getSettingValue("KJNodes.nodeAutoColor")){
				// 	setColorAndBgColor.call(this, type);	
				// }
			} else {
				alert("Error: Set node input undefined. Most likely you're missing custom nodes");
			}
		}
		if (link_info && node.graph && slotType == 2) {
			const fromNode = node.graph._nodes.find((otherNode) => otherNode.id == link_info.origin_id);
			
			if (fromNode && fromNode.inputs && fromNode.inputs[link_info.origin_slot]) {
				// const type = fromNode.inputs[link_info.origin_slot].type;
				
				// this.outputs[0].type = type;
				// this.outputs[0].name = type;
			} else {
				alert("Error: Get Set node output undefined. Most likely you're missing custom nodes");
			}
		}

	}

	onConnectionsChange = function(
		slotType,	//1 = input, 2 = output
		slot,
		isChangeConnect,
		link_info,
		output
	) {

		console.log({
			slotType: slotType, 
			slot: slot, 
			isChangeConnect: isChangeConnect, 
			link_info: link_info, 
			output: output
		})
		//On Disconnect
		if (!isChangeConnect) this.onDisconnect(slotType, slot)
		//On Connect
		if (isChangeConnect) this.onConnect(slotType, slot, node, link_info)

		this.updateBusOutput(); // Update the output bus with the merged data					

		//Update either way
		this.update();
	}

	onRemoved() {
	}
}

app.registerExtension({
	name: "Comfy.MarasIT.MarasitBusNode",
	registerCustomNodes() {
		LiteGraph.registerNodeType(
			"MarasitBusNode",
			Object.assign(MarasitBusNode, {
				title: "Bus Node - Default",
			})
		);

		MarasitBusNode.category = "MarasIT/utils";
	},
});
