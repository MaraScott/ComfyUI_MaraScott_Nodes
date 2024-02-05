import { app } from "/scripts/app.js";

const ExtName = "Comfy.MarasIT.BusNode";
const ExtNodeName = "MarasitBusNode";

const ext = {
	// Unique name for the extension
	name: ExtName,
	async init(app) {
		// Any initial setup to run as soon as the page loads
		// console.log("[MarasIT]",ext.name+" Initialization");
	},
	async setup(app) {
		// Any setup to run after the app is created
		// console.log("[MarasIT]",ext.name+" loaded");
	},
	async addCustomNodeDefs(defs, app) {
		// Add custom node definitions
		// These definitions will be configured and registered automatically
		// defs is a lookup core nodes, add yours into this
		// console.log("[MarasIT]", "add "+ext.name+" definitions", "current nodes:", JSON.stringify(defs['MarasitBusNode']));
	},
	async getCustomWidgets(app) {
		// Return custom widget types
		// See ComfyWidgets for widget examples
		// console.log("[MarasIT]", "provide "+ext.name+" widgets");
	},
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		// Run custom logic before a node definition is registered with the graph
		if (nodeData.name !== ExtNodeName) {
			return;
		}
		// console.log("[MarasIT]", "before register node: ", nodeData.name);

		const onNodeCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function () {
			const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
			this.getExtraMenuOptions = function (_, options) {
				this.index = 10
				options.unshift(
					{
						content: "Add Input",
						callback: () => {
							const name = "any_" + this.inputs.length;
							const type = "*";
							this.addInput(name, type);
							this.addOutput(name, type);
							
							const inputLenth = this.inputs.length-1;
							const outputLenth = this.outputs.length-1;
							const index = this.widgets[this.index].value;

							for (let i = inputLenth; i > index+1; i--) {
								swapInputs(this, i, i-1);
								swapOutputs(this, i, i-1);
							}

							renameNodeInputs(this, name);
							renameNodeOutputs(this, name);

							this.properties["values"].splice(index+1, 0, [0, 0, 0, 0, 1]);
							this.widgets[this.index].options.max = inputLenth;

							// this.setDirtyCanvas(true);

							console.log('In/Out put '+name+' added');
						}
					},
					{
						content: "Remove Last Input",
						callback: () => {
							const inputLenth = this.inputs.length-1
							const outputLenth = this.outputs.length-1

							this.removeInput(inputLenth);
							this.removeOutput(outputLenth);
						}
					},
				);
			};
			this.onRemoved = function() {
				for (let y in this.widgets) {
					if (this.widgets[y].canvas) {
						this.widgets[y].canvas.remove();
					}
				}
			};
			this.onSelected = function() {
				this.selected = true;
			};
			this.onDeselected = function() {
				this.selected = false;
			};
			return r;
		}
		
		// delete ext.beforeRegisterNodeDef;

		// This fires for every node definition so only log once
	},
	async registerCustomNodes(app) {
		// Register any custom node implementations here allowing for more flexability than a custom node def
		// console.log("[MarasIT]", "register "+this.name);
	},
	loadedGraphNode(node, app) {
		// Fires for each node when loading/dragging/etc a workflow json or png
		// If you break something in the backend and want to patch workflows in the frontend
		// This is the place to do this
		if(node.type === ExtNodeName){
			// console.log("[MarasIT]", "loaded graph node: ", node, node.widgets[node.index].options["max"], node.properties["values"].length-1);

			// This fires for every node on each load so only log once
			// delete ext.loadedGraphNode;
		}

	},
	nodeCreated(node, app) {
		// Fires every time a node is constructed
		// You can modify widgets/add handlers/etc here
		console.log("[MarasIT]", "node created: ", node);

		// This fires for every node so only log once
		delete ext.nodeCreated;
	}
};

app.registerExtension(ext);
