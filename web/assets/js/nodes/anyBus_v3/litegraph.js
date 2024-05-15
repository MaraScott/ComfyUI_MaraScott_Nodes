class litegraph {

	_ext = null

	constructor(extension) {
        this.ext = extension
	}

	onExecuted(nodeType) {
		const onExecuted = nodeType.prototype.onExecuted
		nodeType.prototype.onExecuted = function (message) {
			onExecuted?.apply(this, arguments)
			// console.log("[MaraScott - logging " + this.name + "]", "on Executed", { "id": this.id, "properties": this.properties });
		}

	}

	onNodeCreated(nodeType) {

		const onNodeCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function () {
			const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

			// console.log("[MaraScott - logging " + this.name + "]", 'onNodeCreated')
			this.ext.core.configure(this)
			this.ext.core.setWidgets(this)

			return r;
		}

	}

	getExtraMenuOptions(nodeType) {
		this.ext.menu.addMenuHandler(nodeType, this.ext.menu.viewProfile)
	}

	onConnectionsChange(nodeType) {

		const onConnectionsChange = nodeType.prototype.onConnectionsChange
		nodeType.prototype.onConnectionsChange = function (
			slotType,	//1 = input, 2 = output
			slot,
			isChangeConnect,
			link_info,
			output
			) {

			const r = onConnectionsChange ? onConnectionsChange.apply(this, arguments) : undefined

			if (!window.marascott[this.ext.name].init) return r

			window.marascott[this.ext.name].sync = this.ext.flow.NOSYNC
			window.marascott[this.ext.name].input.index = slot + 1 - this.ext.core.FIRST_INDEX

			//On Disconnect
			if (!isChangeConnect && slotType == 1 && typeof link_info != 'undefined') {
				// console.log('disconnect');

				if (slot < this.ext.core.FIRST_INDEX) {
					// bus
					if (slot == 0 && this.inputs) {
						window.marascott[this.ext.name].sync = this.ext.core.disConnectBus(this)
					}

				} else {

					window.marascott[this.ext.name].sync = this.ext.core.disConnectInput(this, slot)
					
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
					
				if (slot < this.ext.core.FIRST_INDEX) {
					// bus
					window.marascott[this.ext.name].sync = this.ext.core.connectBus(this, slot, link_info_node, link_info.origin_slot)

				} else {

					window.marascott[this.ext.name].sync = this.ext.core.connectInput(this, slot, link_info_node, link_info.origin_slot)


				}

			}

			if (isChangeConnect && slotType == 2 && typeof link_info != 'undefined') {
				// do something
				// this.inputs[slot].name = ":) ("+slot.toString().padStart(2, '0')+")"
			}

			this.ext.flow.syncProfile(this, null, isChangeConnect)

			return r;
		}

	}

	onRemoved(nodeType) {
		const onRemoved = nodeType.prototype.onRemoved;
		nodeType.prototype.onRemoved = function () {
			onRemoved?.apply(this, arguments);
			// console.log('onRemoved')
		};
	}

    get ext(){
        return this._ext;
    }
    
    set ext(extension){
        this._ext = extension;
    }

}

export { litegraph }