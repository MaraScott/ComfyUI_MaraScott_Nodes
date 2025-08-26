/* eslint-disable no-undef */

// Basic AnyBus node with default profile and inputs

const DEFAULT_PROFILE = "default";
const DEFAULT_INPUTS = 2;
const MIN_INPUTS = 2;
const MAX_INPUTS = 24;

function initNode(node) {
    node.properties = node.properties || {};
    if (node.properties.profile === undefined) node.properties.profile = DEFAULT_PROFILE;
    if (node.properties.inputs === undefined) node.properties.inputs = DEFAULT_INPUTS;

    node.shape = LiteGraph.CARD_SHAPE;
    node.color = LGraphCanvas.node_colors.green.color;
    node.bgcolor = LGraphCanvas.node_colors.green.bgcolor;
    node.groupcolor = LGraphCanvas.node_colors.green.groupcolor;
    node.title = `AnyBus - ${node.properties.profile}`;

    setWidgets(node);
    updateIO(node);
}

function setWidgets(node) {
    if (!node.widgets) node.widgets = [];

    if (!node.widgets.find((w) => w.name === "profile")) {
        node.addWidget("text", "profile", node.properties.profile, (v) => {
            node.properties.profile = v || DEFAULT_PROFILE;
            node.title = `AnyBus - ${node.properties.profile}`;
            adjustSize(node);
        });
    }

    if (!node.widgets.find((w) => w.name === "inputs")) {
        const values = Array.from({ length: MAX_INPUTS - MIN_INPUTS + 1 }, (_, i) => i + MIN_INPUTS);
        node.addWidget("combo", "inputs", node.properties.inputs, (v) => {
            node.properties.inputs = v;
            updateIO(node);
        }, { values });
    }
}

function updateIO(node) {
    const count = Number(node.properties.inputs) || DEFAULT_INPUTS;
    node.inputs = [];
    node.outputs = [];
    node.addInput("bus", "BUS");
    node.addOutput("bus", "BUS");
    for (let i = 1; i <= count; i++) {
        const label = `* ${i.toString().padStart(2, "0")}`.toLowerCase();
        node.addInput(label, "*");
        node.addOutput(label, "*");
    }
    adjustSize(node);
}

function adjustSize(node) {
    const size = node.computeSize(node.size);
    node.size[1] = size[1];
    node.setDirtyCanvas?.(true, true);
}

const MaraScottAnyBus_v2 = (ComfyWidgets) => ({
    name: "ComfyUI.MaraScott.AnyBus_v2",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "MaraScottAnyBus_v2") return;

        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onCreated?.apply(this, arguments);
            initNode(this);
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            onConfigure?.apply(this, arguments);
            initNode(this);
        };
    },
});

export { MaraScottAnyBus_v2 };

