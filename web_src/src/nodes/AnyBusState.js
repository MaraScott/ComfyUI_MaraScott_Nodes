export class AnyBusState {
    constructor() {
        this.init = false;
        this.sync = 0;
        this.input = { label: "0", index: 0 };
        this.clean = false;
        this.nodeToSync = null;
        this.flows = { start: [], list: [], end: [] };
        this.nodes = {};
        this.__syncing = false; // reentrancy guard for group sync
    }

    reset() {
        this.flows = { start: [], list: [], end: [] };
        this.nodes = {};
    }
}
