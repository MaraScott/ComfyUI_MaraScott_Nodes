const initialState = {
    init: false,
    sync: 0,
    input: { label: "0", index: 0 },
    clean: false,
    nodeToSync: null,
    flows: { start: [], list: [], end: [] },
    nodes: {},
    __syncing: false,
};

let state = { ...initialState };
const listeners = new Set();

export function getAnyBusState() {
    return state;
}

export function setAnyBusState(updater) {
    state = typeof updater === 'function' ? updater(state) : { ...state, ...updater };
    for (const l of listeners) l(state);
}

export function useAnyBusState() {
    const { useState, useEffect } = globalThis.React;
    const [s, setS] = useState(state);
    useEffect(() => {
        listeners.add(setS);
        return () => listeners.delete(setS);
    }, []);
    return [s, setAnyBusState];
}

export function resetAnyBusState() {
    setAnyBusState({ ...initialState });
}
