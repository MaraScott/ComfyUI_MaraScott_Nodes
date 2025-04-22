export const NODES = {}
export const FLOWS = {
    start: [],
    list: [],
    end: [],
}
export const BUS =  {
    init: false,
    sync: false,
    input: {
        label: "0",
        index: 0,
    },
    clean: false,
    nodeToSync: null,
    flows: FLOWS,
    nodes: NODES,
}
