import { app } from "/scripts/app.js";

function addMenuHandler(nodeType, cb) {
	const getOpts = nodeType.prototype.getExtraMenuOptions;
	nodeType.prototype.getExtraMenuOptions = function () {
		const r = getOpts.apply(this, arguments);
		cb.apply(this, arguments);
		return r;
	};
	nodeType.prototype.onRemoved = function () {
	};
	
	nodeType.prototype.onSelected = function () {
		this.selected = true
	}
	nodeType.prototype.onDeselected = function () {
		this.selected = false
	}
}

function addInput() {
	console.log('add input');
}

function removeInput() {
	console.log('remove input');
}

app.registerExtension({
	name: "Comfy.MarasIT.BusNode",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {

		if (nodeData.name === "MarasITBusNode") {
			addMenuHandler(nodeType, function (_, options) {
				options.unshift(
					{
						content: "Add Input",
						callback: () => {
							addInput(this);
						}
					},
					{
						content: "Remove Last Input",
						callback: () => {
							removeInput(this);
						}
					},
				);
			});
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