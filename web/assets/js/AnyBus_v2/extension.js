import { Bus } from "./AnyBus_v2/bus.js";
import { Flow } from "./AnyBus_v2/flow.js";
import { LiteGraph_Hooks } from "./AnyBus_v2/litegraph-hooks.js";

export const Extension = {
    // Unique name for the extension
    name: "Comfy.MaraScott.AnyBus_v2",
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
            // console.log("[MaraScott - logging " + this.name + "]", "add custom node definitions", "current nodes:", defs[Bus.TYPE], JSON.stringify(Object.keys(defs)));

        } else {
            // console.log("[MaraScott - logging " + this.name + "]", "add custom node definitions", "current nodes:", defs[Bus.TYPE]);
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
        if (node.type == Bus.TYPE) {

            node.setProperty('uuid', node.id)
            Flow.setFlows(node);
            // console.log("[MaraScott - logging " + this.name + "]", "Loaded Graph", { "id": node.id, "properties": node.properties });

        }

        // This fires for every node on each load so only log once
        // delete Bus.loadedGraphNode;
    },
    // this is the python node created
    nodeCreated(node, app) {
        // Fires every time a node is constructed
        // You can modify widgets/add handlers/etc here
        // console.log("[MaraScott - logging " + this.name + "]", "node created: ", { ...node });

        // This fires for every node so only log once
        // delete Bus.nodeCreated;
    },
    beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Run custom logic before a node definition is registered with the graph

        if (nodeData.name === Bus.TYPE) {
            // This fires for every node definition so only log once
            // console.log("[MaraScott - logging " + this.name + "]", "before register node: ", nodeData, typeof LiteGraph_Hooks, typeof LiteGraph_Hooks.onNodeCreated);

            // LiteGraph_Hooks.onExecuted(nodeType)
            LiteGraph_Hooks.onNodeCreated(nodeType)
            // LiteGraph_Hooks.getExtraMenuOptions(nodeType)
            LiteGraph_Hooks.onConnectionsChange(nodeType)
            // delete Bus.beforeRegisterNodeDef;
            LiteGraph_Hooks.onRemoved(nodeType)

        }
    },
    beforeConfigureGraph(app) {
        // console.log("[MaraScott - logging " + this.name + "]", "extension beforeConfigureGraph");
        window.marascott.AnyBus_v2.init = false
    },
    afterConfigureGraph(app) {
        // console.log("[MaraScott - logging " + this.name + "]", "extension afterConfigureGraph");
        window.marascott.AnyBus_v2.init = true
    },

};