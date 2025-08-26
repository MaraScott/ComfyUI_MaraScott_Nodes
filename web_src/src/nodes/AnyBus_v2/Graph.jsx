// ---------- MapProxy-safe helpers for graph.links ----------
function _linksIsMapLike(ln) {
    return ln && typeof ln.get === 'function' && typeof ln.values === 'function';
}
function getGraphLinksArray(graph) {
    const ln = graph?.links;
    if (!ln) return [];
    if (Array.isArray(ln)) return ln;
    if (_linksIsMapLike(ln)) return Array.from(ln.values());
    const arr = [];
    for (const k in ln) if (Object.prototype.hasOwnProperty.call(ln, k)) arr.push(ln[k]);
    return arr;
}
function getGraphLinkById(graph, id) {
    const ln = graph?.links;
    if (!ln) return null;
    if (_linksIsMapLike(ln)) return ln.get(id) ?? null;
    if (Array.isArray(ln)) return ln.find(l => l && l.id === id) ?? null;
    return ln[id] ?? null;
}
function getActiveGraph() {
    return globalThis.app?.graph ?? globalThis.graph ?? null;
}

export { getGraphLinksArray, getGraphLinkById, getActiveGraph };