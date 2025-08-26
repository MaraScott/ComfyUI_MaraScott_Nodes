import { PROFILE, INPUTS, setValue as setNodeValue } from "./Node.jsx";
import { getGraphLinkById } from "./Graph.jsx";

const getByName = (node, name) => {
    return node.widgets?.find(w => w.name === name);
}

const setWidgetValue = (node, name, value) => {
    const widget = getByName(node, name);
    if (widget) {
        widget.value = value;
        node.setDirtyCanvas?.(true, true);
    }
};

const initWidgetProfile = (node) => {
    if (!getByName(node, PROFILE.name)) {
        node.addWidget(
            "text",
            PROFILE.name,
            node.properties[PROFILE.name],
            (value) => setNodeValue(node, PROFILE.name, value),
            { title: "Profile name for this bus" }
        );
    } else {
        setWidgetValue(PROFILE.name, node.properties[PROFILE.name]);
    }
}

const initWidgetInputs = (node) => {
    if (!getByName(node, INPUTS.name)) {
        const values = Array.from({ length: INPUTS.max - INPUTS.min + 1 }, (_, i) => i + INPUTS.min);
        node.addWidget(
            "combo",
            INPUTS.name,
            node.properties[INPUTS.name],
            (value) => setNodeValue(node, INPUTS.name, value),
            { values, title: "Number of input/output pairs" }
        );
    } else {
        setWidgetValue(INPUTS.name, node.properties[INPUTS.name]);
    }
}

const updateNodeIO = (node) => {
    if (!node.graph) return;
    const numInputs = node.properties[INPUTS.name];

    const oldConnections = (node.inputs || []).map((input, idx) => {
        if (input?.link != null) {
            const link = getGraphLinkById(node.graph, input.link);
            if (link) return { idx, link };
        }
        return null;
    }).filter(Boolean);

    node.inputs = [];
    node.outputs = [];

    node.addInput("bus", "BUS");
    node.addOutput("bus", "BUS");

    for (let i = 1; i <= numInputs; i++) {
        const name = `* ${i.toString().padStart(2, '0')}`.toLowerCase();
        node.addInput(name, "*");
        node.addOutput(name, "*");
    }

    oldConnections.forEach(({ idx, link }) => {
        const originNode = node.graph.getNodeById(link.origin_id);
        if (originNode) node.connect(idx, originNode, link.origin_slot);
    });

    node.setDirtyCanvas?.(true, true);
}

const init = (node) => {
    initWidgetProfile(node);
    initWidgetInputs(node);
}

export { init, setWidgetValue, updateNodeIO }
