import { Bus } from "./bus.js";
import { Flow } from "./flow.js";

export class LiteGraph_Hooks {

    static onExecuted(nodeType) {
        const onExecuted = nodeType.prototype.onExecuted
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments)
            // console.log("[MaraScott - logging " + this.name + "]", "on Executed", { "id": this.id, "properties": this.properties });
        }

    }

    static onNodeCreated(nodeType) {

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            // console.log("[MaraScott - logging " + this.name + "]", 'onNodeCreated')
            Bus.configure(this)
            Bus.setWidgets(this)

            return r;
        }

    }

    static getExtraMenuOptions(nodeType) {
        const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (_, options) {

            // console.log("[MaraScott - logging " + this.name + "]", "on Extra Menu Options", { "id": this.id, "properties": this.properties });

        }
    }

    static onConnectionsChange(nodeType) {

        const onConnectionsChange = nodeType.prototype.onConnectionsChange
        nodeType.prototype.onConnectionsChange = function (
            slotType,	//1 = input, 2 = output
            slot,
            isChangeConnect,
            link_info,
            output
        ) {

            const r = onConnectionsChange ? onConnectionsChange.apply(this, arguments) : undefined

            if (!window.marascott.AnyBus_v2.init) return r

            window.marascott.AnyBus_v2.sync = Flow.NOSYNC
            window.marascott.AnyBus_v2.input.index = slot + 1 - Bus.FIRST_INDEX

            //On Disconnect
            if (!isChangeConnect && slotType == 1 && typeof link_info != 'undefined') {
                // console.log('disconnect');

                if (slot < Bus.FIRST_INDEX) {
                    // bus
                    if (slot == 0 && this.inputs) {
                        window.marascott.AnyBus_v2.sync = Bus.disConnectBus(this)
                    }

                } else {

                    window.marascott.AnyBus_v2.sync = Bus.disConnectInput(this, slot)

                }

            }
            if (!isChangeConnect && slotType == 2 && typeof link_info != 'undefined') {
                // do something
                // this.inputs[slot].name = "* ("+slot.toString().padStart(2, '0')+")"
            }

            //On Connect
            if (isChangeConnect && slotType == 1 && typeof link_info != 'undefined' && this.graph) {
                // console.log('connect');

                // do something
                let link_info_node = this.graph._nodes.find(
                    (otherNode) => otherNode.id == link_info.origin_id
                )

                if (slot < Bus.FIRST_INDEX) {
                    // bus
                    window.marascott.AnyBus_v2.sync = Bus.connectBus(this, slot, link_info_node, link_info.origin_slot)

                } else {

                    window.marascott.AnyBus_v2.sync = Bus.connectInput(this, slot, link_info_node, link_info.origin_slot)


                }

            }

            if (isChangeConnect && slotType == 2 && typeof link_info != 'undefined') {
                // do something
                // this.inputs[slot].name = ":) ("+slot.toString().padStart(2, '0')+")"
            }

            Flow.syncProfile(this, null, isChangeConnect)

            return r;
        }

    }

    static onRemoved(nodeType) {
        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            onRemoved?.apply(this, arguments);
            // console.log('onRemoved')
        };
    }


}