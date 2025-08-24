import { app } from "../../scripts/app.js";

class MaraScottAnyBusNodeWidget {
    static NAMESPACE = "MaraScott";
    static TYPE = "AnyBus_v2";
    static BUS_SLOT = 0;
    static FIRST_INDEX = 1;

    static ALLOWED_REROUTE_TYPE = [
        "Reroute (rgthree)",
    ];

    static ALLOWED_GETSET_TYPE = [
        "SetNode",
        "GetNode",
    ];

    static ALLOWED_NODE_TYPE = [
        MaraScottAnyBusNodeWidget.NAMESPACE + MaraScottAnyBusNodeWidget.TYPE,
        ...MaraScottAnyBusNodeWidget.ALLOWED_REROUTE_TYPE,
        ...MaraScottAnyBusNodeWidget.ALLOWED_GETSET_TYPE,
    ];

    static PROFILE = { name: "profile", default: "default" };
    static INPUTS = { name: "inputs", min: 2, max: 24, default: 2 };

    static init(node) {
        node.properties = node.properties || {};
        node.properties[this.PROFILE.name] = node.properties[this.PROFILE.name] ?? this.PROFILE.default;
        node.properties[this.INPUTS.name] = node.properties[this.INPUTS.name] ?? this.INPUTS.default;
        node.properties['prevProfileName'] = node.properties[this.PROFILE.name];

        node.shape = LiteGraph.CARD_SHAPE;
        node.color = LGraphCanvas.node_colors.green.color;
        node.bgcolor = LGraphCanvas.node_colors.green.bgcolor;
        node.groupcolor = LGraphCanvas.node_colors.green.groupcolor;
        node.size[0] = 150;

        this.updateNodeTitle(node);
    }

    static updateNodeTitle(node) {
        node.title = "AnyBus - " + node.properties[this.PROFILE.name];
    }

    static updateNodeIO(node) {
        if (!node.graph) return;

        const numInputs = node.properties[this.INPUTS.name];
        const oldConnections = (node.inputs || []).map((input, idx) => {
            if (input.link != null) {
                const link = node.graph.links[input.link];
                if (link) {
                    return { idx, link };
                }
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

        node.setDirtyCanvas(true);
    }

    static setWidgets(node) {
        const updateWidgetValue = (name, value) => {
            const widget = node.widgets.find(w => w.name === name);
            if (widget) widget.value = value;
        };

        if (!this.getByName(node, this.PROFILE.name)) {
            node.addWidget("text", this.PROFILE.name, node.properties[this.PROFILE.name], value => this.setValue(node, this.PROFILE.name, value));
        } else {
            updateWidgetValue(this.PROFILE.name, node.properties[this.PROFILE.name]);
        }

        if (!this.getByName(node, this.INPUTS.name)) {
            const values = Array.from({ length: this.INPUTS.max - this.INPUTS.min + 1 }, (_, i) => i + this.INPUTS.min);
            node.addWidget("combo", this.INPUTS.name, node.properties[this.INPUTS.name], value => this.setValue(node, this.INPUTS.name, value), { values });
        } else {
            updateWidgetValue(this.INPUTS.name, node.properties[this.INPUTS.name]);
        }
    }

    static getByName(node, name) {
        return node.widgets?.find(w => w.name === name);
    }

    static setValue(node, name, value) {
        node.properties[name] = value;
        if (name === this.PROFILE.name) this.updateNodeTitle(node);
        if (name === this.INPUTS.name) this.updateNodeIO(node);
    }
}

const MaraScottAnyBusNodeExtension = {
    name: "Comfy.MaraScott.AnyBus_v2",
    async beforeRegisterNodeDef(nodeType) {
        if (nodeType.comfyClass === MaraScottAnyBusNodeWidget.NAMESPACE + MaraScottAnyBusNodeWidget.TYPE) {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                MaraScottAnyBusNodeWidget.init(this);
                MaraScottAnyBusNodeWidget.setWidgets(this);
                MaraScottAnyBusNodeWidget.updateNodeIO(this);
                onNodeCreated?.apply(this, arguments);
                this.serialize_widgets = true;
            };
        }
    },
    async setup() {
        console.log("AnyBus_v2 extension setup complete");
    }
};

app.registerExtension(MaraScottAnyBusNodeExtension);
