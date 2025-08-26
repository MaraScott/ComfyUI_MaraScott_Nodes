/**
 * Graph traversal utilities for AnyBus v2 nodes.
 * @module AnyBusTraversals
 */

/**
 * @typedef {Object} GraphNode
 * @property {number} id
 * @property {string} type
 * @property {{link?: number, name?: string, type?: string, links?: number[]}?[]} [inputs]
 * @property {{links?: number[]}?[]} [outputs]
 * @property {any} graph
 * @property {Record<string, any>} [properties]
 */

/**
 * @typedef {Object} GraphLink
 * @property {number} id
 * @property {number} origin_id
 * @property {number} origin_slot
 * @property {number} target_id
 */

export const BUS_SLOT = 0;
export const FIRST_INDEX = 1;

export const ALLOWED_REROUTE = ["Reroute (rgthree)"];
export const ALLOWED_GETSET = ["SetNode", "GetNode"];
export const ANYBUS_TYPE = "MaraScottAnyBus_v2";

/**
 * Fetch a link by id from a graph, handling MapProxy variations.
 * @param {any} graph
 * @param {number} id
 * @returns {GraphLink|null}
 */
export function getGraphLinkById(graph, id) {
    const ln = graph?.links;
    if (!ln) return null;
    if (ln.get && ln.values) return ln.get(id) ?? null;
    if (Array.isArray(ln)) return ln.find(l => l && l.id === id) ?? null;
    return ln[id] ?? null;
}

/**
 * Return the active LiteGraph instance.
 * @returns {any}
 */
export function getActiveGraph() {
    return globalThis.app?.graph ?? globalThis.graph ?? null;
}

/**
 * Emit custom event notifying that AnyBus state changed.
 */
export function notifyAnyBusChange() {
    try {
        window.dispatchEvent(new CustomEvent('marascott:anybus:changed'));
    } catch {
        /* noop */
    }
}

/**
 * Get upstream AnyBus node connected to the bus input.
 * @param {GraphNode} node
 * @returns {GraphNode|null}
 */
export function getBusPrev(node) {
    const in0 = node.inputs?.[BUS_SLOT];
    if (!in0 || in0.link == null) return null;
    const l = getGraphLinkById(node.graph, in0.link);
    return l ? node.graph.getNodeById(l.origin_id) : null;
}

/**
 * Get downstream AnyBus nodes connected to the bus output.
 * @param {GraphNode} node
 * @returns {GraphNode[]}
 */
export function getBusNext(node) {
    const out0 = node.outputs?.[BUS_SLOT];
    const next = [];
    if (!out0?.links?.length) return next;
    for (const linkId of out0.links) {
        const l = getGraphLinkById(node.graph, linkId);
        if (!l) continue;
        const t = node.graph.getNodeById(l.target_id);
        if (t?.type === ANYBUS_TYPE) next.push(t);
    }
    return next;
}

/**
 * Check whether the node participates in any bus connection.
 * @param {GraphNode} node
 * @returns {boolean}
 */
export function hasBusLink(node) {
    const inLinked = node.inputs?.[BUS_SLOT]?.link != null;
    const outLinked = (node.outputs?.[BUS_SLOT]?.links?.length ?? 0) > 0;
    return !!(inLinked || outLinked);
}

/**
 * Breadth-first traversal of connected AnyBus nodes starting from `start`.
 * @param {GraphNode} start
 * @returns {GraphNode[]}
 */
export function getBusGroup(start) {
    const seen = new Set(), q = [start], group = [];
    while (q.length) {
        const n = q.shift();
        if (!n || seen.has(n.id)) continue;
        seen.add(n.id);
        group.push(n);
        const p = getBusPrev(n);
        if (p?.type === ANYBUS_TYPE) q.push(p);
        for (const nx of getBusNext(n)) q.push(nx);
    }
    return group;
}

/**
 * Return the connected group of AnyBus nodes or null if isolated.
 * @param {GraphNode} node
 * @returns {GraphNode[]|null}
 */
export function getConnectedGroupOrNull(node) {
    if (!hasBusLink(node)) return null;
    const g = getBusGroup(node);
    return g.length > 1 ? g : null;
}

/**
 * Build path list for a group of AnyBus nodes.
 * @param {GraphNode[]} group
 * @returns {GraphNode[][]}
 */
export function getBusPaths(group) {
    const byId = new Map(group.map(n => [n.id, n]));
    const heads = group.filter(n => !getBusPrev(n));
    const paths = [];
    function dfs(node, pathIds) {
        const nexts = getBusNext(node);
        if (!nexts.length || nexts.every(nx => !byId.has(nx.id))) {
            paths.push([...pathIds, node.id]);
            return;
        }
        for (const nx of nexts) {
            if (!byId.has(nx.id)) continue;
            if (pathIds.includes(nx.id)) continue;
            dfs(nx, [...pathIds, node.id]);
        }
    }
    if (heads.length) for (const h of heads) dfs(h, []);
    else for (const n of group) dfs(n, []);
    return paths.map(ids => ids.map(id => byId.get(id)));
}

/**
 * Collect AnyBus nodes grouped by profile name.
 * @param {any} graph
 * @param {string} profileKey
 * @param {string} defaultProfile
 * @returns {Map<string, GraphNode[]>}
 */
export function collectAnyBusByProfile(graph, profileKey, defaultProfile) {
    const all = (graph?._nodes ?? []).filter(n => n?.type === ANYBUS_TYPE);
    const byProfile = new Map();
    for (const n of all) {
        const profile = n?.properties?.[profileKey] ?? defaultProfile;
        if (!byProfile.has(profile)) byProfile.set(profile, []);
        byProfile.get(profile).push(n);
    }
    return byProfile;
}

/**
 * Downstream nodes within provided id set.
 * @param {GraphNode} node
 * @param {Set<number>} idSet
 * @returns {GraphNode[]}
 */
export function nextWithinProfile(node, idSet) {
    return getBusNext(node).filter(nx => idSet.has(nx.id));
}

/**
 * Upstream node within provided id set.
 * @param {GraphNode} node
 * @param {Set<number>} idSet
 * @returns {GraphNode|null}
 */
export function prevWithinProfile(node, idSet) {
    const p = getBusPrev(node);
    return p && idSet.has(p.id) ? p : null;
}

/**
 * Return all bus chains for nodes belonging to the same profile.
 * @param {GraphNode[]} nodes
 * @returns {GraphNode[][]}
 */
export function chainsForProfile(nodes) {
    if (!nodes.length) return [];
    const idSet = new Set(nodes.map(n => n.id));
    const heads = nodes.filter(n => !prevWithinProfile(n, idSet));
    const chains = [];
    function dfs(node, path) {
        const nexts = nextWithinProfile(node, idSet);
        if (!nexts.length) {
            chains.push([...path, node]);
            return;
        }
        for (const nx of nexts) {
            if (path.includes(nx)) {
                chains.push([...path, node]);
                continue;
            }
            dfs(nx, [...path, node]);
        }
    }
    if (heads.length) heads.forEach(h => dfs(h, []));
    else dfs(nodes[0], []);
    return chains;
}

/**
 * Summaries of input slots for a node.
 * @param {GraphNode} node
 * @returns {{slot:number,name:string,type:string,linked:boolean}[]}
 */
export function summarizeSlots(node) {
    const arr = [];
    const max = node.inputs?.length ?? 0;
    for (let s = FIRST_INDEX; s < max; s++) {
        const inp = node.inputs[s];
        if (!inp) continue;
        arr.push({
            slot: s,
            name: inp.name,
            type: inp.type,
            linked: inp.link != null
        });
    }
    return arr;
}

/**
 * Details about the bus edge between two nodes.
 * @param {any} graph
 * @param {GraphNode} fromNode
 * @param {GraphNode} toNode
 * @returns {{id:number|null, valid:boolean}}
 */
export function busEdgeInfo(graph, fromNode, toNode) {
    const link = toNode?.inputs?.[BUS_SLOT]?.link;
    const l = link != null ? getGraphLinkById(graph, link) : null;
    if (l && l.origin_id === fromNode.id && l.origin_slot === BUS_SLOT) {
        return { id: l.id, valid: true };
    }
    return { id: l?.id ?? null, valid: false };
}
