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
        // Check if node and graph are properly initialized
        if (!node || !node.graph) {
            console.warn("Node or graph not initialized yet");
            return;
        }

        try {
            const targetLength = value + Bus.FIRST_INDEX;
            const currentLength = node.inputs?.length || 0;

            // Remove inputs if we have too many
            if (currentLength > targetLength) {
                for (let i = currentLength - 1; i >= targetLength; i--) {
                    if (node.inputs[i] && !node.inputs[i].link) {
                        // Only remove unconnected inputs
                        node.removeInput(i);
                        node.removeOutput(i);
                    }
                }
            }
            // Add inputs if we need more
            else if (currentLength < targetLength) {
                for (let i = currentLength; i < targetLength; i++) {
                    const name = `* ${i.toString().padStart(2, '0')}`;
                    const type = "*";
                    node.addInput(name, type);
                    node.addOutput(name, type);
                }
            }
        } catch (error) {
            console.error("Error modifying node inputs/outputs:", error);
        }
    }

    static setValue(node, name, value) {
        try {
            const nodeWidget = this.getByName(node, name);
            if (!nodeWidget) {
                console.warn(`Widget '${name}' not found`);
                return;
            }

            nodeWidget.value = value;
            node.setProperty(name, nodeWidget.value ?? node.properties[name]);

            if (name === this.PROFILE.name) {
                this.setValueProfile(node, name, value);
            }
            if (name === this.INPUTS.name) {
                // Defer input modification to next tick to ensure graph is ready
                setTimeout(() => this.setValueInputs(node, name, value), 0);
            }

            node.setDirtyCanvas(true);
        } catch (error) {
            console.error("Error setting widget value:", error);
        }
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
        if (!node || !node.graph) {
            console.warn("Node or graph not initialized yet");
            return;
        }

        try {
            const nodeWidget = this.getByName(node, this.INPUTS.name);
            if (nodeWidget !== undefined) return;

            const values = Array.from(
                { length: this.INPUTS.max - this.INPUTS.min + 1 },
                (_, i) => i + this.INPUTS.min
            );

            node.addWidget(
                "combo",
                this.INPUTS.name,
                node.properties[this.INPUTS.name] ?? this.INPUTS.default,
                (value) => {
                    this.setValue(node, this.INPUTS.name, value);
                    window.marascott.AnyBus_v2.sync = Flow.FULLSYNC;
                    Flow.syncProfile(node, this.INPUTS.name, null);
                },
                { values }
            );

            // Defer initial setup to ensure graph is ready
            setTimeout(() => {
                node.setProperty(this.INPUTS.name, this.INPUTS.default);
                this.setValue(node, this.INPUTS.name, this.INPUTS.default);
            }, 0);
        } catch (error) {
            console.error("Error setting up inputs select:", error);
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