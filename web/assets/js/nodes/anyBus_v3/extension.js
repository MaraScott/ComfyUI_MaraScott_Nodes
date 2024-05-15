import { $t } from './../../utils/i18n.js'
import { core } from './core.js'
import { widget } from './widget.js'
import { flow } from './flow.js'
import { menu } from './menu.js'
import { litegraph } from './litegraph.js'

const getExtension = (ext) => {
    return {
        // Unique name for the extension
        name: "Comfy."+ext.prefix+"."+ext.name,
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
                // console.log("[MaraScott - logging " + this.name + "]", "add custom node definitions", "current nodes:", defs[ext.core.TYPE], JSON.stringify(Object.keys(defs)));

            } else {
                // console.log("[MaraScott - logging " + this.name + "]", "add custom node definitions", "current nodes:", defs[ext.core.TYPE]);
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
            if (node.type == ext.core.TYPE) {

                node.setProperty('uuid', node.id)
                ext.flow.setFlows(node);
                // console.log("[MaraScott - logging " + this.name + "]", "Loaded Graph", { "id": node.id, "properties": node.properties });

            }

            // This fires for every node on each load so only log once
            // delete ext.core.loadedGraphNode;
        },
        // this is the python node created
        nodeCreated(node, app) {
            // Fires every time a node is constructed
            // You can modify widgets/add handlers/etc here
            // console.log("[MaraScott - logging " + this.name + "]", "node created: ", { ...node });

            // This fires for every node so only log once
            // delete ext.core.nodeCreated;
        },
        beforeRegisterNodeDef(nodeType, nodeData, app) {
            // Run custom logic before a node definition is registered with the graph

            if (nodeData.name === ext.core.TYPE) {
                // This fires for every node definition so only log once
                // console.log("[MaraScott - logging " + this.name + "]", "before register node: ", nodeData, typeof ext.litegraph, typeof ext.litegraph.onNodeCreated);

                // ext.litegraph.onExecuted(nodeType)
                ext.litegraph.onNodeCreated(nodeType)
                ext.litegraph.getExtraMenuOptions(nodeType, ext.menu.viewProfile)
                ext.litegraph.onConnectionsChange(nodeType)
                // delete ext.core.beforeRegisterNodeDef;
                ext.litegraph.onRemoved(nodeType)

            }
        },
        beforeConfigureGraph(app) {
            // console.log("[MaraScott - logging " + this.name + "]", "extension beforeConfigureGraph");
            window.marascott[ext.name].init = false
        },
        afterConfigureGraph(app) {
            // console.log("[MaraScott - logging " + this.name + "]", "extension afterConfigureGraph");
            window.marascott[ext.name].init = true
        },
    }
}


class extension {

	TYPE = "MaraScottAnyBusNode_v3"
    prefix = 'MaraScott'
    
    name = 'anyBus_v3'
    lSP = extension.prefix+'.'+this.name // localStorage Prefix

    $t = $t
    
    _core = null
    _widget = null
    _flow = null
    _menu = null
    _litegraph = null

    constructor(type, name) {
        this.TYPE = type
        this.name = name
        this.lSP = extension.prefix+'.'+this.name
        this.init()
        this.core = new core(this)
        this.widget = new widget(this)
        this.flow = new flow(this)
        this.menu = new menu(this)
        this.litegrah = new litegraph(this)

        return getExtension(this)
    }

    init() {

        if (!window.marascott) {
            window.marascott = {}
        }
        
        if (!window.marascott[this.name]) {
            window.marascott[this.name] = {}
        }
        
        window.marascott[this.name] = this.getJson()
    }

    getJson() {
        return {
            init: false,
            sync: false,
            input: {
                label: "0",
                index: 0,
            },
            clean: false,
            nodeToSync: null,
            flows: {
                start: [],
                list: [],
                end: [],
            },
            nodes: {},
            profiles: localStorage[this.lSP+'.profiles'] || {
                _default: {},
                _basic_pipe: {},
                _detailer_pipe: {},
            },
        }
    }

    get core(){
        return this._core;
    }
    
    set core(core){
        this._core = core;
    }

    get widget(){
        return this._widget;
    }

    set widget(widget){
        this._widget = widget;
    }

    get flow(){
        return this._flow;
    }

    set flow(flow){
        this._flow = flow;
    }

    get menu(){
        return this._menu;
    }

    set menu(menu){
        this._menu = menu;
    }

    get litegrah(){
        return this._litegrah;
    }

    set litegrah(litegrah){
        this._litegrah = litegrah;
    }

};

export { extension }