import { getGraphLinkById } from "./Graph.jsx";

function getBusPrev(node) {
    const in0 = node.inputs?.[MaraScottAnyBus_v2.BUS_SLOT];
    if (!in0 || in0.link == null) return null;
    const l = getGraphLinkById(node.graph, in0.link);
    return l ? node.graph.getNodeById(l.origin_id) : null;
}
function getBusNext(node) {
    const out0 = node.outputs?.[MaraScottAnyBus_v2.BUS_SLOT];
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
function hasBusLink(node) {
    const inLinked = node.inputs?.[MaraScottAnyBus_v2.BUS_SLOT]?.link != null;
    const outLinked = (node.outputs?.[MaraScottAnyBus_v2.BUS_SLOT]?.links?.length ?? 0) > 0;
    return !!(inLinked || outLinked);
}
function getBusGroup(start) {
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
function getConnectedGroupOrNull(node) {
    if (!hasBusLink(node)) return null;
    const g = getBusGroup(node);
    return g.length > 1 ? g : null; // only meaningful if > 1
}
function getBusPaths(group) {
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
    else for (const n of group) dfs(n, []); // cycle/singleton fallback
    return paths.map(ids => ids.map(id => byId.get(id)));
}

// Connected AnyBus nodes for a given profile
function collectAnyBusByProfile(graph) {
    const all = (graph?._nodes ?? []).filter(n => n?.type === ANYBUS_TYPE);
    const byProfile = new Map();
    for (const n of all) {
        const profile = n?.properties?.[MaraScottAnyBusNodeWidget.PROFILE.name] ?? MaraScottAnyBusNodeWidget.PROFILE.default;
        if (!byProfile.has(profile)) byProfile.set(profile, []);
        byProfile.get(profile).push(n);
    }
    return byProfile;
}
function nextWithinProfile(node, idSet) {
    return getBusNext(node).filter(nx => idSet.has(nx.id));
}
function prevWithinProfile(node, idSet) {
    const p = getBusPrev(node);
    return p && idSet.has(p.id) ? p : null;
}
function chainsForProfile(nodes, graph) {
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
    else dfs(nodes[0], []); // cycle fallback
    return chains;
}
function summarizeSlots(node) {
    const arr = [];
    const first = MaraScottAnyBus_v2.FIRST_INDEX;
    const max = node.inputs?.length ?? 0;
    for (let s = first; s < max; s++) {
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
function busEdgeInfo(graph, fromNode, toNode) {
    const link = toNode?.inputs?.[MaraScottAnyBus_v2.BUS_SLOT]?.link;
    const l = link != null ? getGraphLinkById(graph, link) : null;
    if (l && l.origin_id === fromNode.id && l.origin_slot === MaraScottAnyBus_v2.BUS_SLOT) {
        return { id: l.id, valid: true };
    }
    return { id: l?.id ?? null, valid: false };
}

// ---------- Bus traversal utilities ----------
const ALLOWED_REROUTE = ["Reroute (rgthree)"];
const ALLOWED_GETSET = ["SetNode", "GetNode"];
const ANYBUS_TYPE = "MaraScottAnyBus_v2";

// ---------- Group sync (title & input-count) ----------
function syncTitleAndInputsFrom(initiator) {
    if (!initiator?.graph || getAnyBusState().__syncing) return;

    const group = getConnectedGroupOrNull(initiator);
    if (!group) return; // disconnected or single node ⇒ no sync

    setAnyBusState(s => ({ ...s, __syncing: true }));
    try {
        const title = initiator.properties?.[MaraScottAnyBusNodeWidget.PROFILE.name]
            ?? MaraScottAnyBusNodeWidget.PROFILE.default;
        const inputs = initiator.properties?.[MaraScottAnyBusNodeWidget.INPUTS.name]
            ?? MaraScottAnyBusNodeWidget.INPUTS.default;

        for (const n of group) {
            MaraScottAnyBusNodeWidget.setValue(n, MaraScottAnyBusNodeWidget.PROFILE.name, title, true);
            if (n.properties?.[MaraScottAnyBusNodeWidget.INPUTS.name] !== inputs) {
                MaraScottAnyBusNodeWidget.setValue(n, MaraScottAnyBusNodeWidget.INPUTS.name, inputs, true);
            }
        }
        for (const n of group) n.setDirtyCanvas?.(true, true);
    } finally {
        setAnyBusState(s => ({ ...s, __syncing: false }));
    }
}

// ---------- Per-slot reconciliation (later overrides earlier if same type) ----------
function isDefaultStarLabel(name) {
    return typeof name === 'string' && /^\*\s*\d+/.test(name.trim());
}
function reconcileSlotLabels(initiator) {
    if (!initiator?.graph) return;

    const group = getConnectedGroupOrNull(initiator);
    if (!group) return; // disconnected or single node ⇒ no reconciliation

    const paths = getBusPaths(group);
    if (!paths.length) return;

    const maxInputs = Math.max(
        ...group.map(n => (n.properties?.[MaraScottAnyBusNodeWidget.INPUTS.name]
            ?? MaraScottAnyBusNodeWidget.INPUTS.default))
    );

    for (let slot = MaraScottAnyBus_v2.FIRST_INDEX; slot <= maxInputs; slot++) {
        let best = null; // {type, label, depth}
        for (const path of paths) {
            let canonType = null;
            let canonLabel = null;
            for (let depth = 0; depth < path.length; depth++) {
                const node = path[depth];
                const inp = node.inputs?.[slot];
                if (!inp) continue;
                if (!canonType && inp.type && inp.type !== '*') {
                    canonType = inp.type;
                    if (inp.name && !isDefaultStarLabel(inp.name)) canonLabel = inp.name;
                } else if (!canonType) {
                    if (inp.name && !isDefaultStarLabel(inp.name)) canonLabel = canonLabel ?? inp.name;
                } else {
                    if (inp.type === canonType && inp.name && !isDefaultStarLabel(inp.name) && inp.name !== canonLabel) {
                        canonLabel = inp.name; // later wins if same type
                    }
                }
                if (canonType && canonLabel) {
                    const candidate = { type: canonType, label: canonLabel, depth };
                    if (!best || candidate.depth > best.depth) best = candidate;
                }
            }
        }
        if (!best) continue;

        for (const n of group) {
            const inp = n.inputs?.[slot];
            const out = n.outputs?.[slot];
            if (!inp || !out) continue;

            inp.name = best.label;
            if (inp.type === '*' || inp.type === best.type) {
                inp.type = best.type;
            }
            out.name = inp.name;

            const outHasLinks = out.links && out.links.length > 0;
            if (!outHasLinks || out.type === best.type || out.type === '*') {
                out.type = best.type;
            }
        }
    }
    for (const n of group) n.setDirtyCanvas?.(true, true);
}

// ---------- Widget object (keeps your exported API) ----------
const MaraScottAnyBusNodeWidget = {
    NAMESPACE: "MaraScott",
    TYPE: "AnyBus_v2",
    BUS_SLOT: 0,
    FIRST_INDEX: 1,

    ALLOWED_REROUTE_TYPE: ALLOWED_REROUTE.slice(),
    ALLOWED_GETSET_TYPE: ALLOWED_GETSET.slice(),
    ALLOWED_NODE_TYPE: [],

    PROFILE: { name: "profile", default: "default" },
    INPUTS: { name: "inputs", min: 2, max: 24, default: 2 },
    CLEAN: { name: "Clean Inputs", default: false },

    init(node) {
        node.properties = node.properties || {};
        node.properties[this.PROFILE.name] = node.properties[this.PROFILE.name] ?? this.PROFILE.default;
        node.properties[this.INPUTS.name] = node.properties[this.INPUTS.name] ?? this.INPUTS.default;
        node.properties['prevProfileName'] = node.properties[this.PROFILE.name];

        node.shape = LiteGraph.CARD_SHAPE;
        node.color = LGraphCanvas.node_colors.green.color;
        node.bgcolor = LGraphCanvas.node_colors.green.bgcolor;
        node.groupcolor = LGraphCanvas.node_colors.green.groupcolor;
        node.size[0] = 150;

        this.updateNodeTitle(node);
    },

    updateNodeTitle(node) {
        node.title = "AnyBus - " + node.properties[this.PROFILE.name];
    },

    getByName(node, name) {
        return node.widgets?.find(w => w.name === name);
    },

    setWidgetValue(node, name, value) {
        const w = this.getByName(node, name);
        if (w) {
            w.value = value;
            node.setDirtyCanvas?.(true, true);
        }
    },

    updateNodeIO(node) {
        if (!node.graph) return;
        const numInputs = node.properties[this.INPUTS.name];

        const oldConnections = (node.inputs || []).map((input, idx) => {
            if (input?.link != null) {
                const link = getGraphLinkById(node.graph, input.link);
                if (link) return { idx, link };
            }
            return null;
        }).filter(Boolean);

        node.inputs = [];
        node.outputs = [];

        node.addInput("bus", "BUS");
        node.addOutput("bus", "BUS");

        for (let i = 1; i <= numInputs; i++) {
            const name = `* ${i.toString().padStart(2, '0')}`.toLowerCase();
            node.addInput(name, "*");
            node.addOutput(name, "*");
        }

        oldConnections.forEach(({ idx, link }) => {
            const originNode = node.graph.getNodeById(link.origin_id);
            if (originNode) node.connect(idx, originNode, link.origin_slot);
        });

        node.setDirtyCanvas(true);
    },

    setWidgets(node) {
        const updateWidgetValue = (name, value) => {
            const widget = node.widgets?.find(w => w.name === name);
            if (widget) widget.value = value;
        };

        if (!this.getByName(node, this.PROFILE.name)) {
            node.addWidget(
                "text",
                this.PROFILE.name,
                node.properties[this.PROFILE.name],
                (value) => this.setValue(node, this.PROFILE.name, value),
                { title: "Profile name for this bus" }
            );
        } else {
            updateWidgetValue(this.PROFILE.name, node.properties[this.PROFILE.name]);
        }

        if (!this.getByName(node, this.INPUTS.name)) {
            const values = Array.from({ length: this.INPUTS.max - this.INPUTS.min + 1 }, (_, i) => i + this.INPUTS.min);
            node.addWidget(
                "combo",
                this.INPUTS.name,
                node.properties[this.INPUTS.name],
                (value) => this.setValue(node, this.INPUTS.name, value),
                { values, title: "Number of input/output pairs" }
            );
        } else {
            updateWidgetValue(this.INPUTS.name, node.properties[this.INPUTS.name]);
        }

        if (!this.getByName(node, this.CLEAN.name)) {
            node.addWidget(
                "toggle",
                this.CLEAN.name,
                this.CLEAN.default,
                () => {
                    try { MaraScottAnyBus_v2.clean(node); }
                    catch (e) { console.error('[AnyBus] clean toggle error', e); }
                },
                { title: "Clean and reset all connections" }
            );
        }
    },

    // __noPropagate guards group-wide recursion
    setValue(node, name, value, __noPropagate = false) {
        const oldValue = node.properties[name];
        node.properties[name] = value;

        // Keep widget UI synchronized locally
        this.setWidgetValue(node, name, value);

        if (name === this.PROFILE.name) {
            this.updateNodeTitle(node);
            node.properties['prevProfileName'] = value;
        } else if (name === this.INPUTS.name && oldValue !== value) {
            this.updateNodeIO(node);
        }

        node.setDirtyCanvas?.(true, true);

        if (__noPropagate) {
            notifyAnyBusChange();       // <— ensure sidebar refresh even for internal sync
            return;
        }
        // Only propagate to others if we are bus-connected to at least one AnyBus node
        if (hasBusLink(node)) {
            syncTitleAndInputsFrom(node);
            reconcileSlotLabels(node);
        }

        notifyAnyBusChange();
    }
};

MaraScottAnyBusNodeWidget.ALLOWED_NODE_TYPE = [
    MaraScottAnyBusNodeWidget.NAMESPACE + MaraScottAnyBusNodeWidget.TYPE,
    ...MaraScottAnyBusNodeWidget.ALLOWED_REROUTE_TYPE,
    ...MaraScottAnyBusNodeWidget.ALLOWED_GETSET_TYPE,
];

// ---------- Core AnyBus mechanics used on connections ----------
class MaraScottAnyBus_v2 {
    static TYPE = ANYBUS_TYPE;
    static BUS_SLOT = 0;
    static FIRST_INDEX = 1;

    // Enforce: target can connect to bus only if its profile is default or equals origin's profile
    static connectBus(targetNode, slot, originNode, origin_slot) {
        const profileKey = MaraScottAnyBusNodeWidget.PROFILE.name;
        const targetProfile = targetNode.properties?.[profileKey] ?? MaraScottAnyBusNodeWidget.PROFILE.default;
        const originProfile = originNode.properties?.[profileKey] ?? MaraScottAnyBusNodeWidget.PROFILE.default;

        const isAllowed = (targetProfile === MaraScottAnyBusNodeWidget.PROFILE.default) ||
            (targetProfile === originProfile);

        if (!isAllowed) {
            // reject the connection: keep target values as-is
            targetNode.disconnectInput(slot);
            targetNode.setDirtyCanvas?.(true, true);
            notifyAnyBusChange();
            return;
        }

        // Sync inputs-count from origin into the connected group (initiated by target)
        const inputsKey = MaraScottAnyBusNodeWidget.INPUTS.name;
        const originInputs = originNode.properties?.[inputsKey] ?? MaraScottAnyBusNodeWidget.INPUTS.default;
        MaraScottAnyBusNodeWidget.setValue(targetNode, inputsKey, originInputs);

        // Reconcile labels/types across the connected group
        reconcileSlotLabels(targetNode);
        notifyAnyBusChange();
    }

    static connectInput(targetNode, slot, originNode, origin_slot) {
        const originOutput = originNode.outputs?.[origin_slot];
        if (!originOutput) return;

        const label = (originOutput.name || '').toLowerCase();
        const type = originOutput.type || '*';

        if (targetNode.inputs?.[slot]) {
            targetNode.inputs[slot].name = label || `* ${slot.toString().padStart(2, '0')}`;
            targetNode.inputs[slot].type = type;
            if (targetNode.outputs?.[slot]) {
                targetNode.outputs[slot].name = targetNode.inputs[slot].name;
                targetNode.outputs[slot].type = type;
            }
        }
        targetNode.setDirtyCanvas?.(true, true);

        // Only reconcile if the node is bus-connected to at least one AnyBus node
        if (hasBusLink(targetNode)) {
            reconcileSlotLabels(targetNode);
        }
        notifyAnyBusChange();
    }

    static disConnectBus(node) {
        // Do nothing: disconnected node keeps its values exactly as they are
        // No propagation — guards elsewhere ensure sync only happens when bus-connected.
        node.setDirtyCanvas?.(true, true);
        notifyAnyBusChange();
    }

    static disConnectInput(node, slot) {
        // Do nothing: keep current label/type for the slot and its mirrored output
        node.setDirtyCanvas?.(true, true);
        notifyAnyBusChange();
    }

    static clean(node) {
        // Cleaning only refreshes connections; does not alter properties/labels/types
        if (!node?.graph) return;
        let touched = false;

        // If bus is connected, re-plug to refresh drawing, but keep values
        if (node.inputs?.[this.BUS_SLOT]?.link != null) {
            const link = getGraphLinkById(node.graph, node.inputs[this.BUS_SLOT].link);
            if (link) {
                const origin = node.graph.getNodeById(link.origin_id);
                if (origin) {
                    node.disconnectInput(this.BUS_SLOT);
                    origin.connect(this.BUS_SLOT, node, this.BUS_SLOT);
                    touched = true;
                }
            }
        } else {
            // For regular inputs, just re-plug existing links; don't change labels or types
            for (let s = this.FIRST_INDEX; s < (node.inputs?.length ?? 0); s++) {
                const inp = node.inputs[s];
                if (!inp) continue;
                if (inp.link != null) {
                    const link = getGraphLinkById(node.graph, inp.link);
                    if (link) {
                        const originNode = node.graph.getNodeById(link.origin_id);
                        if (originNode) {
                            node.disconnectInput(s);
                            originNode.connect(link.origin_slot, node, s);
                            touched = true;
                        }
                    }
                }
            }
        }

        if (touched) {
            const cleanedLabel = " ... cleaned";
            node.title = node.title + cleanedLabel; node.setDirtyCanvas?.(true, true);
            setTimeout(() => {
                node.title = node.title.replace(cleanedLabel, "");
                node.setDirtyCanvas?.(true, true);
            }, 500);
            notifyAnyBusChange();
        }
    }
}

const MaraScott_onNodeCreated =function (_this) {
    MaraScottAnyBusNodeWidget.init(_this);
    MaraScottAnyBusNodeWidget.setWidgets(_this);
    MaraScottAnyBusNodeWidget.updateNodeIO(_this);
    // Ensure widget values reflect properties on creation
    MaraScottAnyBusNodeWidget.setWidgetValue(_this, MaraScottAnyBusNodeWidget.PROFILE.name, _this.properties[MaraScottAnyBusNodeWidget.PROFILE.name]);
    MaraScottAnyBusNodeWidget.setWidgetValue(_this, MaraScottAnyBusNodeWidget.INPUTS.name, _this.properties[MaraScottAnyBusNodeWidget.INPUTS.name]);
    onNodeCreated?.apply(_this, arguments);
    _this.serialize_widgets = true;
    setAnyBusState(s => ({ ...s, init: true }));
}

const MaraScott_onConnectionsChange = function (_this, type, slotIndex, isConnected, link, ioSlot) {
    if (!_this.graph) return;

    // Update "current input index" (1-based display)
        setAnyBusState(s => ({
            ...s,
            input: { ...s.input, index: slotIndex + 1 - MaraScottAnyBus_v2.FIRST_INDEX },
        }));

    if (isConnected && link) {
        const originNode = _this.graph.getNodeById(link.origin_id);
        if (originNode) {
            if (slotIndex === MaraScottAnyBus_v2.BUS_SLOT) {
                // Only bus→bus allowed AND enforce profile condition
                if (originNode.type === ANYBUS_TYPE && link.origin_slot === MaraScottAnyBus_v2.BUS_SLOT) {
                    MaraScottAnyBus_v2.connectBus(_this, slotIndex, originNode, link.origin_slot);
                } else {
                    _this.disconnectInput(slotIndex); // invalid bus connection
                }
            } else {
                MaraScottAnyBus_v2.connectInput(_this, slotIndex, originNode, link.origin_slot);
            }
        }
    } else if (!isConnected) {
        if (slotIndex === MaraScottAnyBus_v2.BUS_SLOT) {
            MaraScottAnyBus_v2.disConnectBus(_this);
        } else {
            MaraScottAnyBus_v2.disConnectInput(_this, slotIndex);
        }
    }

    // Keep title/inputs consistent ONLY when we still have a bus link
    if (slotIndex === MaraScottAnyBus_v2.BUS_SLOT && _this.inputs?.[MaraScottAnyBus_v2.BUS_SLOT]?.link != null) {
        syncTitleAndInputsFrom(_this);
    }
    notifyAnyBusChange();
    onConnectionsChange?.apply(_this, arguments);
}

export { MaraScottAnyBus_v2, MaraScottAnyBusNodeWidget, MaraScott_onNodeCreated, MaraScott_onConnectionsChange };
