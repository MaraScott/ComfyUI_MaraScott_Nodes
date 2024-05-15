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

import { app } from "../../scripts/app.js";
import { core as MaraScottAnyBusNode_v3 } from './nodes/anyBus_v3/core.js'
import { flow as MaraScottAnyBusNode_v3Flow } from './nodes/anyBus_v3/flow.js'
import { menu as MaraScottAnyBusNode_v3Menus } from './nodes/anyBus_v3/menu.js'
import { liteGraph as MaraScottAnyBusNode_v3LiteGraph } from './nodes/anyBus_v3/litegraph.js'

const name_version = "anyBus_v3";
window.marascott[name_version] = MaraScottAnyBusNode_v3.get(name_version)

const MaraScottAnyBusNode_v3Extension = {
	// Unique name for the extension
	name: "Comfy.marascott."+name_version,
	init(app) {
		// Any initial setup to run as soon as the page loads
		// console.log("[MaraScott - logging " + this.name + "]", "extension init");
	},
	setup(app) {
		// Any setup to run after the app is created
		// console.log("[MaraScott - logging " + this.name + "]", "extension setup");
	},
	// !TODO should I find a way to define defs based on profile ?
	addCustomNodeDefs(defs, app) {
		// Add custom node definitions
		// These definitions will be configured and registered automatically
		// defs is a lookup core nodes, add yours into this
		const withNodesNames = false
		if (withNodesNames) {
			// console.log("[MaraScott - logging " + this.name + "]", "add custom node definitions", "current nodes:", defs[MaraScottAnyBusNode_v3.TYPE], JSON.stringify(Object.keys(defs)));

		} else {
			// console.log("[MaraScott - logging " + this.name + "]", "add custom node definitions", "current nodes:", defs[MaraScottAnyBusNode_v3.TYPE]);
		}
	},
	getCustomWidgets(app) {
		// Return custom widget types
		// See ComfyWidgets for widget examples
		// console.log("[MaraScott - logging " + this.name + "]", "provide custom widgets");
	},
	registerCustomNodes(app) {
		// Register any custom node implementations here allowing for more flexability than a custom node def
		// console.log("[MaraScott - logging " + this.name + "]", "register custom nodes");
	},
	loadedGraphNode(node, app) {
		// Fires for each node when loading/dragging/etc a workflow json or png
		// If you break something in the backend and want to patch workflows in the frontend
		// This is the place to do this
		if (node.type == MaraScottAnyBusNode_v3.TYPE) {

			node.setProperty('uuid', node.id)
			MaraScottAnyBusNode_v3Flow.setFlows(node);
			// console.log("[MaraScott - logging " + this.name + "]", "Loaded Graph", { "id": node.id, "properties": node.properties });

		}

		// This fires for every node on each load so only log once
		// delete MaraScottAnyBusNode_v3.loadedGraphNode;
	},
	// this is the python node created
	nodeCreated(node, app) {
		// Fires every time a node is constructed
		// You can modify widgets/add handlers/etc here
		// console.log("[MaraScott - logging " + this.name + "]", "node created: ", { ...node });

		// This fires for every node so only log once
		// delete MaraScottAnyBusNode_v3.nodeCreated;
	},
	beforeRegisterNodeDef(nodeType, nodeData, app) {
		// Run custom logic before a node definition is registered with the graph

		if (nodeData.name === MaraScottAnyBusNode_v3.TYPE) {
			// This fires for every node definition so only log once
			// console.log("[MaraScott - logging " + this.name + "]", "before register node: ", nodeData, typeof MaraScottAnyBusNode_v3LiteGraph, typeof MaraScottAnyBusNode_v3LiteGraph.onNodeCreated);

			// MaraScottAnyBusNode_v3LiteGraph.onExecuted(nodeType)
			MaraScottAnyBusNode_v3LiteGraph.onNodeCreated(nodeType)
			MaraScottAnyBusNode_v3LiteGraph.getExtraMenuOptions(nodeType, MaraScottAnyBusNode_v3Menus.viewProfile)
			MaraScottAnyBusNode_v3LiteGraph.onConnectionsChange(nodeType)
			// delete MaraScottAnyBusNode_v3.beforeRegisterNodeDef;
			MaraScottAnyBusNode_v3LiteGraph.onRemoved(nodeType)

		}
	},
	beforeConfigureGraph(app) {
		// console.log("[MaraScott - logging " + this.name + "]", "extension beforeConfigureGraph");
		window.marascott.anyBus_v3.init = false
	},
	afterConfigureGraph(app) {
		// console.log("[MaraScott - logging " + this.name + "]", "extension afterConfigureGraph");
		window.marascott.anyBus_v3.init = true
	},

};

app.registerExtension(MaraScottAnyBusNode_v3Extension);
