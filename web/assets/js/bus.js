import { app } from "/scripts/app.js";

app.registerExtension({
	name: "Comfy.MarasIT.BusNode",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === "MarasITBusNode") {
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

				// this.setProperty("width", 512)

				// this.getExtraMenuOptions = function(_, options) {
				// 	options.unshift(
						// {
						// 	content: `insert input above ${this.widgets[this.index].value} /\\`,
						// 	callback: () => {
						// 		this.addInput("any", "ANY")
								
						// 		const inputLenth = this.inputs.length-1
						// 		const index = this.widgets[this.index].value

						// 		for (let i = inputLenth; i > index; i--) {
						// 			swapInputs(this, i, i-1)
						// 		}
						// 		renameNodeInputs(this, "any")

						// 		this.properties["values"].splice(index, 0, [0, 0, 0, 0, 1])
						// 		this.widgets[this.index].options.max = inputLenth

						// 		this.setDirtyCanvas(true);

						// 	},
						// },
						// {
						// 	content: `insert input below ${this.widgets[this.index].value} \\/`,
						// 	callback: () => {
						// 		this.addInput("any", "ANY")
								
						// 		const inputLenth = this.inputs.length-1
						// 		const index = this.widgets[this.index].value

						// 		for (let i = inputLenth; i > index+1; i--) {
						// 			swapInputs(this, i, i-1)
						// 		}
						// 		renameNodeInputs(this, "any")

						// 		this.properties["values"].splice(index+1, 0, [0, 0, 0, 0, 1])
						// 		this.widgets[this.index].options.max = inputLenth

						// 		this.setDirtyCanvas(true);
						// 	},
						// },
						// {
						// 	content: `swap with input above ${this.widgets[this.index].value} /\\`,
						// 	callback: () => {
						// 		const index = this.widgets[this.index].value
						// 		if (index !== 0) {
						// 			swapInputs(this, index, index-1)

						// 			renameNodeInputs(this, "any")

						// 			this.properties["values"].splice(index-1,0,this.properties["values"].splice(index,1)[0]);
						// 			this.widgets[this.index].value = index-1

						// 			this.setDirtyCanvas(true);
						// 		}
						// 	},
						// },
						// {
						// 	content: `swap with input below ${this.widgets[this.index].value} \\/`,
						// 	callback: () => {
						// 		const index = this.widgets[this.index].value
						// 		if (index !== this.inputs.length-1) {
						// 			swapInputs(this, index, index+1)

						// 			renameNodeInputs(this, "any")
									
						// 			this.properties["values"].splice(index+1,0,this.properties["values"].splice(index,1)[0]);
						// 			this.widgets[this.index].value = index+1

						// 			this.setDirtyCanvas(true);
						// 		}
						// 	},
						// },
						// {
						// 	content: `remove currently selected input ${this.widgets[this.index].value}`,
						// 	callback: () => {
						// 		const index = this.widgets[this.index].value
						// 		removeNodeInputs(this, [index])
						// 		renameNodeInputs(this, "any")
						// 	},
						// },
						// {
						// 	content: "remove all unconnected inputs",
						// 	callback: () => {
						// 		let indexesToRemove = []

						// 		for (let i = 0; i < this.inputs.length; i++) {
						// 			if (!this.inputs[i].link) {
						// 				indexesToRemove.push(i)
						// 			}
						// 		}

						// 		if (indexesToRemove.length) {
						// 			removeNodeInputs(this, indexesToRemove, "conditioning")
						// 		}
						// 		renameNodeInputs(this, "conditioning")
						// 	},
						// },
				// 	);
				// }

				// this.onRemoved = function () {
				// 	// When removing this node we need to remove the input from the DOM
				// 	for (let y in this.widgets) {
				// 		if (this.widgets[y].canvas) {
				// 			this.widgets[y].canvas.remove();
				// 		}
				// 	}
				// };
			
				// this.onSelected = function () {
				// 	this.selected = true
				// }
				// this.onDeselected = function () {
				// 	this.selected = false
				// }

				return r;
			};
		}
	},
	loadedGraphNode(node, app) {
        if (node.type === "MarasitBusNode") {
            // console.log({"type": node.type, "title": node.title})
			// node.widgets[node.index].options["max"] = node.properties["values"].length-1
		}
	},
	
});

console.log("Maras IT Bus Node loaded");