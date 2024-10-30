/*
 * TextConcatenate_v1
 */

import { app } from "../../scripts/app.js";

if (!window.marascott) {
	window.marascott = {}
}
if (!window.marascott.TextConcatenate_v1) {
	window.marascott.TextConcatenate_v1 = {
        init: false,
	}
}

class MaraScottTextConcatenate_v1NodeWidget {

    static INPUTS = {
        name: "Nb Inputs",
        default: 2,
        min: 2,
        max: 24,
    };

    static setValueInputs(node, name, value) {
        let qty = 0;
        let _value = value + MaraScottTextConcatenate_v1.FIRST_INDEX;
        if (node.inputs.length > _value) {
            qty = node.inputs.length - _value;
            for (let i = qty; i > 0; i--) {
                node.removeInput(node.inputs.length - 1);
            }
        }
        if (node.inputs.length < _value) {
            qty = _value - node.inputs.length;
            for (let i = 0; i < qty; i++) {
                const name = "string " + (node.inputs.length + 1).toString().padStart(2, '0');
                const type = "STRING";
                node.addInput(name, type);
            }
        }
    }

    static setValue(node, name, value) {
        const nodeWidget = this.getByName(node, name);
        nodeWidget.value = value;
        node.setProperty(name, nodeWidget.value ?? node.properties[name]);
        this.setValueInputs(node, name, value);
        node.setDirtyCanvas(true);
    }

    static setInputsSelect(node) {

        const nodeWidget = this.getByName(node, this.INPUTS.name);

        if (nodeWidget == undefined) {    
            let values = [];
            for (let i = this.INPUTS.min; i <= this.INPUTS.max; i++) {
                values.push(i);
            }

            node.addWidget(
                "combo",
                this.INPUTS.name,
                node.properties[this.INPUTS.name] ?? this.INPUTS.default,
                (value, LGraphCanvas, Node, Coordinate, PointerEvent) => {
                    this.setValue(node, this.INPUTS.name, value);
                },
                {  "values": values  }
            );
            node.setProperty(this.INPUTS.name, this.INPUTS.default);
            this.setValue(node, this.INPUTS.name, this.INPUTS.default);

            for (let i = node.widgets.length - 1; i >= 0; i--) {
                if (node.widgets[i].name.startsWith("string") ) {
                    node.widgets.splice(i, 1);  // Remove the widget from node.widgets
                    if(node.widgets_values) {
                        node.widgets_values.splice(i, 1);  // Remove the corresponding value from node.widgets_values
                    }
                }
            }

        }
    }

    static getByName(node, name) {
        return node.widgets?.find((w) => w.name === name);
    }
}

class MaraScottTextConcatenate_v1 {

	static TYPE = "MaraScottTextConcatenate_v1"

    static FIRST_INDEX = 0;

    static configure(node) {
        node.shape = LiteGraph.CARD_SHAPE;
        node.color = LGraphCanvas.node_colors.green.color;
        node.bgcolor = LGraphCanvas.node_colors.green.bgcolor;
        node.size[0] = 150;
		// node.title = "Text Concatenate"

    }

    static setWidgets(node) {
        MaraScottTextConcatenate_v1NodeWidget.setInputsSelect(node);
    }
}

class MaraScottTextConcatenate_v1LiteGraph {
    static onNodeCreated(nodeType) {
		const onNodeCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function () {
			const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            MaraScottTextConcatenate_v1.configure(this);
            MaraScottTextConcatenate_v1.setWidgets(this);
    
			return r;
		}
    }
}

const MaraScottTextConcatenate_v1NodeExtension = {
	// Unique name for the extension
    name: "Comfy.MaraScott.TextConcatenate_v1",

    // init(app) {},
	// setup(app) {},
	// addCustomNodeDefs(defs, app) {
	// 	const withNodesNames = false
	// 	if (withNodesNames) {
	// 	} else {
	// 	}
	// },
	// getCustomWidgets(app) {
	// },
	// registerCustomNodes(app) {
	// },
	loadedGraphNode(node, app) {
        if (node.type == MaraScottTextConcatenate_v1.TYPE) {
			node.setProperty('uuid', node.id)
        }
	},
	// nodeCreated(node, app) {
	// },
	beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name == MaraScottTextConcatenate_v1.TYPE) {
			MaraScottTextConcatenate_v1LiteGraph.onNodeCreated(nodeType)
        }
	},
	beforeConfigureGraph(app) {
		window.marascott.TextConcatenate_v1.init = false
	},
	afterConfigureGraph(app) {
		window.marascott.TextConcatenate_v1.init = true
	},

};

app.registerExtension(MaraScottTextConcatenate_v1NodeExtension);
