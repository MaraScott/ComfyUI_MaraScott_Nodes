// import * as React from "react";
// import { createRoot } from "react-dom/client";

const MaraScottAnyBusNodeWidget = {
    NAMESPACE: "MaraScott",
    TYPE: "AnyBus_v2",
    BUS_SLOT: 0,
    FIRST_INDEX: 1,

    ALLOWED_REROUTE_TYPE: [
        "Reroute (rgthree)",
    ],

    ALLOWED_GETSET_TYPE: [
        "SetNode",
        "GetNode",
    ],

    ALLOWED_NODE_TYPE: [],

    PROFILE: { name: "profile", default: "default" },
    INPUTS: { name: "inputs", min: 2, max: 24, default: 2 },

    init(node) {
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
    },

    updateNodeTitle(node) {
        node.title = "AnyBus - " + node.properties[this.PROFILE.name];
    },

    updateNodeIO(node) {
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
    },

    setWidgets(node) {
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
    },

    getByName(node, name) {
        return node.widgets?.find(w => w.name === name);
    },

    setValue(node, name, value) {
        node.properties[name] = value;
        if (name === this.PROFILE.name) this.updateNodeTitle(node);
        if (name === this.INPUTS.name) this.updateNodeIO(node);
    }
}

// Compute ALLOWED_NODE_TYPE after the object is fully defined
MaraScottAnyBusNodeWidget.ALLOWED_NODE_TYPE = [
    MaraScottAnyBusNodeWidget.NAMESPACE + MaraScottAnyBusNodeWidget.TYPE,
    ...MaraScottAnyBusNodeWidget.ALLOWED_REROUTE_TYPE,
    ...MaraScottAnyBusNodeWidget.ALLOWED_GETSET_TYPE,
];

const nodeName = "Comfy.MaraScott.AnyBus_v2";
const nodeId = nodeName.replace(/\./g, '-');

const MaraScottAnyBusNodeExtension = () => {
    return {
        name: nodeName,
        aboutPageBadges: [
            {
                label: "Website - MaraScott",
                url: "https://www.marascott.ai/",
                icon: "pi pi-home"
            },
            {
                label: "Donate - MaraScott",
                url: "https://github.com/sponsors/MaraScott",
                icon: "pi pi-heart"
            },
            {
              label: "GitHub - MaraScott",
              url: "https://github.com/MaraScott/ComfyUI_MaraScott_Nodes",
              icon: "pi pi-github"
            }
        ],
        bottomPanelTabs: [
          {
            id: nodeId,
            title: "MaraScott Tab",
            type: "custom",
            render: (el) => {
              el.innerHTML = '<div>This is Mara Scott tab content</div>';
            }
          }
        ],
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
    }
};

const MaraScottAnyBusNodeSidebarTab = () => {
    return {
        id: nodeId,
        icon: "mdi mdi-vector-polyline",
        title: "Any Bus",
        tooltip: "Any Bus Dashboard",
        type: "custom",
        render: (el) => {
            el.innerHTML = '<div>MaraScott</div>';
            // const container = document.createElement("div");
            // container.id = `${nodeId}-container`;
            // el.appendChild(container);

            // // Define a simple React component
            // function SidebarContent() {
            //   const [count, setCount] = React.useState(0);

            //   return (
            //     <div style={{ padding: "10px" }}>
            //       <h3>React Sidebar</h3>
            //       <p>Count: {count}</p>
            //       <button onClick={() => setCount(count + 1)}>
            //         Increment
            //       </button>
            //     </div>
            //   );
            // }

            // // Mount React component
            // createRoot(container).render(
            //   <React.StrictMode>
            //     <SidebarContent />
            //   </React.StrictMode>
            // );
        }
    }
}

export { MaraScottAnyBusNodeExtension, MaraScottAnyBusNodeSidebarTab }