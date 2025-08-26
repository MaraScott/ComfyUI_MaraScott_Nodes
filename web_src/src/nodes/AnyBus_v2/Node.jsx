import { setWidgetValue, updateNodeIO } from "./Widget.jsx";

// ---------- Bus traversal utilities ----------
const NAMESPACE = "MaraScott";
const TYPE = "AnyBus_v2";
const ANYBUS_TYPE = NAMESPACE+TYPE;

const ALLOWED_REROUTE = ["Reroute (rgthree)"];
const ALLOWED_GETSET = ["SetNode", "GetNode"];


const BUS_SLOT = 0;
const FIRST_INDEX = 1;

const ALLOWED_REROUTE_TYPE = ALLOWED_REROUTE.slice();
const ALLOWED_GETSET_TYPE = ALLOWED_GETSET.slice();
const ALLOWED_NODE_TYPE = [];

const PROFILE = { name: "profile", default: "default" };
const INPUTS = { name: "inputs", min: 2, max: 24, default: 2 };
const CLEAN = { name: "Clean Inputs", default: false };

const updateNodeTitle = (node) => {
    node.title = "AnyBus - " + node.properties[PROFILE.name];
}

const setValue = (node, name, value, __noPropagate = false) => {
    const oldValue = node.properties[name];
    node.properties[name] = value;

    // Keep widget UI synchronized locally
    setWidgetValue(node, name, value);

    if (name === PROFILE.name) {
        updateNodeTitle(node);
        node.properties['prevProfileName'] = value;
    } else if (name === INPUTS.name && oldValue !== value) {
        updateNodeIO(node);
    }

    // if (__noPropagate) {
    //     notifyAnyBusChange();       // <— ensure sidebar refresh even for internal sync
    //     return;
    // }
    // Only propagate to others if we are bus-connected to at least one AnyBus node
    // if (hasBusLink(node)) {
    //     syncTitleAndInputsFrom(node);
    //     reconcileSlotLabels(node);
    // }

    // notifyAnyBusChange();
}

const init = (node) => {

    node.properties = node.properties || {};
    node.properties[PROFILE.name] = node.properties[PROFILE.name] ?? PROFILE.default;
    node.properties[INPUTS.name] = node.properties[INPUTS.name] ?? INPUTS.default;
    // node.properties['prevProfileName'] = node.properties[PROFILE.name];

    node.shape = globalThis.LiteGraph.CARD_SHAPE;
    node.color = globalThis.LGraphCanvas.node_colors.green.color;
    node.bgcolor = globalThis.LGraphCanvas.node_colors.green.bgcolor;
    node.groupcolor = globalThis.LGraphCanvas.node_colors.green.groupcolor;

    updateNodeTitle(node);

    node.setDirtyCanvas?.(true, true);

}

export { PROFILE, ANYBUS_TYPE, INPUTS, init, setValue };
