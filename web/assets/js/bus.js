import { app } from "/scripts/app.js";

const ExtName = "Comfy.MarasIT.BusNode";
const ExtNodeName = "MarasitBusNode";

const addMenuHandler = (nodeType, cb) => {
	const getOpts = nodeType.prototype.getExtraMenuOptions;
	nodeType.prototype.getExtraMenuOptions = function () {
		const r = getOpts.apply(this, arguments);
		cb.apply(this, arguments);
		return r;
	};
	nodeType.prototype.onRemoved = function() {
	};
	nodeType.prototype.onSelected = function() {
		this.selected = true;
	};
	nodeType.prototype.onDeselected = function() {
		this.selected = false;
	};
};

const action = {
	addInput: (nodeType, name, type) => {
		console.log(nodeType.prototype.addInput(name, type));
		console.log('Input '+name+' added');
	},
	removeInput: (nodeType) => {
		console.log('remove input');
	}
};

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
		addMenuHandler(nodeType, (_, options) => {
			options.unshift(
				{
					content: "Add Input",
					callback: () => {
						action.addInput(nodeType, "any", "*");
					}
				},
				{
					content: "Remove Last Input",
					callback: () => {
						action.removeInput(nodeType);
					}
				},
			);
		});
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
			// console.log("[MarasIT]", "loaded graph node: ", node);

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
