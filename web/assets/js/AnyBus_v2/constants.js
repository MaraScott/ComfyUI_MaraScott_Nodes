export const CONSTANTS = {
    BUS_SLOT: 0,
    BASIC_PIPE_SLOT: 0,
    REFINER_PIPE_SLOT: 0,
    FIRST_INDEX: 1,
    NODE_TYPE: 'MaraScottAnyBus_v2',
    ALLOWED_REROUTE_TYPE: [
        "Reroute (rgthree)", // SUPPORTED - RgThree Custom Node
        // "Reroute", // UNSUPPORTED - ComfyUI native - do not allow connection on Any Type if origin Type is not Any Type too
        // "ReroutePrimitive|pysssss", // UNSUPPORTED - Pysssss Custom Node - do not display the name of the origin slot
        // "0246.CastReroute", //  UNSUPPORTED - 0246 Custom Node
    ],
    ALLOWED_GETSET_TYPE: [
        "SetNode", // SUPPORTED - ComfyUI-KJNodes Custom Node
        "GetNode", // SUPPORTED - ComfyUI-KJNodes Custom Node
    ]
}
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
