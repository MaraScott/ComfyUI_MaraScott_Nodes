import { Bus } from "./bus.js";
import { Flow } from "./flow.js";

export class Widget {

    static PROFILE = {
        name: 'Profile',
        default: 'default',
    }
    static INPUTS = {
        name: "Nb Inputs",
        default: 5,
        min: 3,
        max: 25,
    }
    static CLEAN = {
        default: false,
        name: 'Clean Inputs',
    }

    static init(node) {

        this.setProfileInput(node)
        this.setInputsSelect(node)
        this.setCleanSwitch(node)

    }

    static getByName(node, name) {
        return node.widgets?.find((w) => w.name === name);
    }

    // static removeByName(node, name) {
    // 	if (node.widgets) node.widgets = node.widgets.filter((w) => w.name !== name);
    // }

    static setValueProfile(node, name, value) {
        node.title = "AnyBus - " + node.properties[name];
    }

    static setValueInputs(node, name, value) {
        let qty = 0
        let _value = value + Bus.FIRST_INDEX
        if (node.inputs.length > _value) {
            qty = node.inputs.length - _value
            for (let i = qty; i > 0; i--) {
                node.removeInput(node.inputs.length - 1)
                node.removeOutput(node.outputs.length - 1)
            }
        } else if (node.inputs.length < _value) {
            qty = _value - node.inputs.length
            for (let i = 0; i < qty; i++) {
                const name = "* " + node.inputs.length.toString().padStart(2, '0')
                const type = "*"
                node.addInput(name, type)
                node.addOutput(name, type)
            }
        }
    }

    static setValue(node, name, value) {

        const nodeWidget = this.getByName(node, name);
        nodeWidget.value = value
        node.setProperty(name, nodeWidget.value ?? node.properties[name])
        if (name == this.PROFILE.name) this.setValueProfile(node, name, value)
        if (name == this.INPUTS.name) this.setValueInputs(node, name, value)
        node.setDirtyCanvas(true)

    }

    static setProfileInput(node) {

        const nodeWidget = this.getByName(node, this.PROFILE.name);

        if (nodeWidget == undefined) {
            node.addWidget(
                "text",
                this.PROFILE.name,
                node.properties[this.PROFILE.name] ?? this.PROFILE.default,
                (value, LGraphCanvas, Node, Coordinate, PointerEvent) => {
                    this.setValue(node, this.PROFILE.name, value)
                    window.marascott.AnyBus_v2.sync = Flow.FULLSYNC;
                    Flow.syncProfile(node, this.PROFILE.name, null)
                    node.setProperty('prevProfileName', node.properties[this.PROFILE.name])

                },
                {}
            )
            this.setValue(node, this.PROFILE.name, this.PROFILE.default)
            node.setProperty('prevProfileName', node.properties[this.PROFILE.name])
        }

    }

    static setInputsSelect(node) {

        const nodeWidget = this.getByName(node, this.INPUTS.name);

        if (nodeWidget == undefined) {

            let values = []

            for (let i = this.INPUTS.min; i <= this.INPUTS.max; i++) {
                values.push(i);
            }

            node.addWidget(
                "combo",
                this.INPUTS.name,
                node.properties[this.INPUTS.name] ?? this.INPUTS.default,
                (value, LGraphCanvas, Node, Coordinate, PointerEvent) => {
                    this.setValue(node, this.INPUTS.name, value)
                    window.marascott.AnyBus_v2.sync = Flow.FULLSYNC;
                    Flow.syncProfile(node, this.INPUTS.name, null)
                },
                {
                    "values": values
                }
            )
            node.setProperty(this.INPUTS.name, this.INPUTS.default)
            this.setValue(node, this.INPUTS.name, this.INPUTS.default)
        }

    }

    static setCleanSwitch(node) {

        const nodeWidget = this.getByName(node, this.CLEAN.name);
        if (nodeWidget == undefined) {
            node.addWidget(
                "toggle",
                this.CLEAN.name,
                this.CLEAN.clean,
                (value, LGraphCanvas, Node, Coordinate, PointerEvent) => {
                    for (const index in window.marascott.AnyBus_v2.flows.end) {
                        const _node = node.graph.getNodeById(window.marascott.AnyBus_v2.flows.end[index])
                        Flow.clean(_node)
                    }
                    this.setValue(node, this.CLEAN.name, this.CLEAN.clean)
                },
                {}
            )
            this.setValue(node, this.CLEAN.name, this.CLEAN.clean)
        }

    }

}