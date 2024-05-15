import { core as MaraScottAnyBusNode_v3 } from './core.js'
import { flow as MaraScottAnyBusNode_v3Flow } from './flow.js'
import { menu as MaraScottAnyBusNode_v3Menus } from './menu.js'


class liteGraph {

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
			MaraScottAnyBusNode_v3.configure(this)
			MaraScottAnyBusNode_v3.setWidgets(this)

			return r;
		}

	}

	static getExtraMenuOptions(nodeType) {
		MaraScottAnyBusNode_v3Menus.addMenuHandler(nodeType, MaraScottAnyBusNode_v3Menus.viewProfile)
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

			if (!window.marascott.anyBus_v3.init) return r

			window.marascott.anyBus_v3.sync = MaraScottAnyBusNode_v3Flow.NOSYNC
			window.marascott.anyBus_v3.input.index = slot + 1 - MaraScottAnyBusNode_v3.FIRST_INDEX

			//On Disconnect
			if (!isChangeConnect && slotType == 1 && typeof link_info != 'undefined') {
				// console.log('disconnect');

				if (slot < MaraScottAnyBusNode_v3.FIRST_INDEX) {
					// bus
					if (slot == 0 && this.inputs) {
						window.marascott.anyBus_v3.sync = MaraScottAnyBusNode_v3.disConnectBus(this)
					}

				} else {

					window.marascott.anyBus_v3.sync = MaraScottAnyBusNode_v3.disConnectInput(this, slot)
					
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
					
				if (slot < MaraScottAnyBusNode_v3.FIRST_INDEX) {
					// bus
					window.marascott.anyBus_v3.sync = MaraScottAnyBusNode_v3.connectBus(this, slot, link_info_node, link_info.origin_slot)

				} else {

					window.marascott.anyBus_v3.sync = MaraScottAnyBusNode_v3.connectInput(this, slot, link_info_node, link_info.origin_slot)


				}

			}

			if (isChangeConnect && slotType == 2 && typeof link_info != 'undefined') {
				// do something
				// this.inputs[slot].name = ":) ("+slot.toString().padStart(2, '0')+")"
			}

			MaraScottAnyBusNode_v3Flow.syncProfile(this, null, isChangeConnect)

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

export { liteGraph }