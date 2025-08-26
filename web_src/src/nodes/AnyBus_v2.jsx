// ===============================================
// AnyBus v2 (ComfyUI) — React UMD + Safe Linking
// Enforced bus connect policy + flow inspector
// JSX with Fragments (<>...</>) — no imports needed
//
// IMPORTANT: Build your *nodes* with esbuild JSX classic pointing to globals:
//   esbuild: {
//     jsx: 'transform',
//     jsxFactory: 'globalThis.React.createElement',
//     jsxFragment: 'globalThis.React.Fragment',
//   }
// ===============================================

import { AnyBusState } from "./AnyBus_v2/State.js";
import {
    BUS_SLOT,
    FIRST_INDEX,
    getGraphLinkById,
    getActiveGraph,
    notifyAnyBusChange,
    ALLOWED_REROUTE,
    ALLOWED_GETSET,
    ANYBUS_TYPE,
    hasBusLink,
    getConnectedGroupOrNull,
    getBusPaths,
    collectAnyBusByProfile,
    chainsForProfile,
    summarizeSlots,
    busEdgeInfo
} from "./AnyBus_v2/Traversal.js";

const state = new AnyBusState();

// ---------- React UMD loader (no imports / no JSX-runtime imports) ----------
async function ensureReactGlobals() {
    if (globalThis.React && globalThis.ReactDOM) return;
    const base = 'extensions/ComfyUI_MaraScott_Nodes/web/assets/js/vendor/react';
    try {
        await loadScript(`${base}/react.production.min.js`);
        await loadScript(`${base}/react-dom.production.min.js`);
    } catch {
        await loadScript('https://unpkg.com/react@19.1.1/umd/react.production.min.js');
        await loadScript('https://unpkg.com/react-dom@19.1.1/umd/react-dom.production.min.js');
    }
    if (!globalThis.React || !globalThis.ReactDOM) {
        console.error('[AnyBus] React/ReactDOM UMD not available');
        throw new Error('React UMD not loaded');
    }
}
function loadScript(src) {
    return new Promise((res, rej) => {
        const s = document.createElement('script');
        s.src = src; s.async = true;
        s.onload = res; s.onerror = () => rej(new Error('Failed to load ' + src));
        document.head.appendChild(s);
    });
}
function mountJSX(el, vnode) {
    if (!el.__ms_mount) {
        const host = document.createElement('div');
        (el.attachShadow ? el.attachShadow({ mode: 'open' }) : el).appendChild(host);
        el.__ms_mount = { container: host, root: globalThis.ReactDOM.createRoot(host) };
    }
    el.__ms_mount.root.render(vnode);
}

// ---------- Group sync (title & input-count) ----------
function syncTitleAndInputsFrom(initiator) {
    if (!initiator?.graph || state.__syncing) return;

    const group = getConnectedGroupOrNull(initiator);
    if (!group) return; // disconnected or single node ⇒ no sync

    state.__syncing = true;
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
        state.__syncing = false;
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
    BUS_SLOT: BUS_SLOT,
    FIRST_INDEX: FIRST_INDEX,

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
    static BUS_SLOT = BUS_SLOT;
    static FIRST_INDEX = FIRST_INDEX;

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

        // If target is default, adopt origin's profile
        if (targetProfile === MaraScottAnyBusNodeWidget.PROFILE.default && originProfile !== targetProfile) {
            MaraScottAnyBusNodeWidget.setValue(targetNode, profileKey, originProfile);
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

// ---------- Extension exports ----------
const nodeName = "Comfy.MaraScott.AnyBus_v2";
const nodeId = nodeName.replace(/\./g, '-');

// Small chip component (inline, no CSS files)
function Chip({ text, kind = 'default' }) {
    const bg = kind === 'warn' ? '#f59e0b22' : kind === 'ok' ? '#16a34a22' : '#6b728022';
    const bd = kind === 'warn' ? '#f59e0b' : kind === 'ok' ? '#16a34a' : '#6b7280';
    return (
        <>
            <span
                style={{
                    display: 'inline-block', padding: '2px 8px', marginRight: 6, marginBottom: 6,
                    fontSize: 12, borderRadius: 999, background: bg, border: '1px solid ' + bd
                }}
            >
                {text}
            </span>
        </>
    );
}

const MaraScottAnyBusNodeExtension = () => {
    return {
        name: nodeName,
        aboutPageBadges: [
            { label: "Website - MaraScott", url: "https://www.marascott.ai/", icon: "pi pi-home" },
            { label: "Donate - MaraScott", url: "https://github.com/sponsors/MaraScott", icon: "pi pi-heart" },
            { label: "GitHub - MaraScott", url: "https://github.com/MaraScott/ComfyUI_MaraScott_Nodes", icon: "pi pi-github" }
        ],
        bottomPanelTabs: [
            {
                id: nodeId,
                title: "MaraScott Flows",
                type: "custom",
                render: async (el) => {
                    await ensureReactGlobals();
                    const React = globalThis.React;

                    function FlowInspector() {
                        const [data, setData] = React.useState({ flows: [], total: 0, scanned: 0 });

                        const compute = React.useCallback(() => {
                            const graph = getActiveGraph();
                            if (!graph) {
                                setData({ flows: [], totalFlows: 0, nodesCount: 0, nodes: [], last: Date.now() });
                                return;
                            }

                            const byProfile = collectAnyBusByProfile(
                                graph,
                                MaraScottAnyBusNodeWidget.PROFILE.name,
                                MaraScottAnyBusNodeWidget.PROFILE.default
                            );

                            // flat list of all AnyBus nodes (for "Nodes" section)
                            const allNodes = Array.from(byProfile.values()).flat();
                            const nodes = allNodes.map((n) => ({
                                id: n.id,
                                title: n.title || `Node #${n.id}`,
                                profile: n.properties?.[MaraScottAnyBusNodeWidget.PROFILE.name] ?? MaraScottAnyBusNodeWidget.PROFILE.default,
                                connected: hasBusLink(n),
                            }));

                            // flows exist only when a chain has at least 2 nodes
                            const flows = [];
                            let totalFlows = 0;

                            for (const [profile, nodesList] of byProfile.entries()) {
                                const chainsRaw = chainsForProfile(nodesList, graph).map((chain, idx) => {
                                    const edges = [];
                                    for (let i = 0; i < chain.length - 1; i++) {
                                        const e = busEdgeInfo(graph, chain[i], chain[i + 1]);
                                        edges.push({ from: chain[i], to: chain[i + 1], linkId: e.id, valid: e.valid });
                                    }
                                    const slotSummaries = {};
                                    for (const n of chain) slotSummaries[n.id] = summarizeSlots(n);
                                    return { nodes: chain, edges, slotSummaries, index: idx + 1 };
                                });

                                // keep only chains with length >= 2
                                const chains = chainsRaw.filter((ch) => ch.nodes.length >= 2);
                                if (chains.length > 0) {
                                    flows.push({ profile, count: chains.length, chains });
                                    totalFlows += chains.length;
                                }
                            }

                            setData({
                                flows,
                                totalFlows,
                                nodesCount: nodes.length,
                                nodes,
                                last: Date.now(),
                            });
                        }, []);

                        React.useEffect(() => { compute(); }, [compute]);

                        return (
                            <>
                                <div style={{ fontFamily: 'sans-serif', fontSize: 13, padding: 8 }}>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8 }}>
                                        <strong>AnyBus Overview</strong>
                                        <Chip text={`${data.nodesCount} node(s)`} />
                                        <Chip text={`${data.totalFlows} flow(s)`} kind={data.totalFlows ? 'ok' : 'warn'} />
                                        <div style={{ flex: 1 }} />
                                        <button onClick={compute} style={{ padding: '2px 8px', cursor: 'pointer' }}>Refresh</button>
                                    </div>

                                    <div style={{ borderTop: '1px solid #e5e7eb', paddingTop: 6, marginTop: 6 }}>
                                        <div style={{ marginBottom: 6 }}>
                                            <strong>Nodes</strong> <Chip text={`${data.nodesCount}`} />
                                        </div>
                                        {data.nodesCount === 0 ? (
                                            <div style={{ opacity: 0.7 }}>No AnyBus nodes detected.</div>
                                        ) : (
                                            <div>
                                                {data.nodes.map((n) => (
                                                    <div key={n.id} style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4, flexWrap: 'wrap' }}>
                                                        <Chip text={n.title} />
                                                        <Chip text={`profile: ${n.profile}`} />
                                                        <Chip text={n.connected ? 'connected' : 'solo'} kind={n.connected ? 'ok' : 'warn'} />
                                                    </div>
                                                ))}
                                            </div>
                                        )}
                                    </div>

                                    <div style={{ borderTop: '1px solid #e5e7eb', paddingTop: 6, marginTop: 6 }}>
                                        <div style={{ marginBottom: 6 }}>
                                            <strong>Flows</strong> <Chip text={`${data.totalFlows}`} kind={data.totalFlows ? 'ok' : 'warn'} />
                                        </div>

                                        {data.flows.length === 0 && (
                                            <div style={{ opacity: 0.7 }}>
                                                No flows yet — connect at least two AnyBus nodes sharing the same profile (or default) via BUS to form a flow.
                                            </div>
                                        )}

                                        {data.flows.map((flow) => (
                                            <div
                                                key={flow.profile + '_chain_' + ch.index}
                                                style={{ padding: '8px 10px', border: '1px solid #e5e7eb', borderRadius: 8, marginBottom: 8, background: '#f9fafb' }}
                                            >
                                                {/* Chain visualization */}
                                                <div style={{ marginBottom: 6, overflowX: 'auto', whiteSpace: 'nowrap' }}>
                                                    {ch.nodes.map((n, i) => (
                                                        <React.Fragment key={n.id + '_frag'}>
                                                            <Chip text={n.title || (`Node #${n.id}`)} />
                                                            {i < ch.nodes.length - 1 && <span style={{ margin: '0 6px' }}>→</span>}
                                                        </React.Fragment>
                                                    ))}
                                                </div>

                                                {/* Edges with bus link IDs */}
                                                <div style={{ marginBottom: 6 }}>
                                                    <div style={{ fontWeight: 600, marginBottom: 4 }}>Links along the way</div>
                                                    {ch.edges.length === 0 ? (
                                                        <div style={{ opacity: 0.7 }}>No BUS edges in this chain.</div>
                                                    ) : (
                                                        ch.edges.map((e, i) => (
                                                            <div key={i} style={{ marginBottom: 2 }}>
                                                                {(e.from.title || e.from.id)} → {(e.to.title || e.to.id)}{' '}
                                                                <Chip text={`bus link: ${e.linkId ?? 'n/a'}`} kind={e.valid ? 'ok' : 'warn'} />
                                                            </div>
                                                        ))
                                                    )}
                                                </div>

                                                {/* Slot summaries */}
                                                <details>
                                                    <summary style={{ cursor: 'pointer', fontWeight: 600 }}>Slot details (per node)</summary>
                                                    {ch.nodes.map((n) => (
                                                        <div key={'slots_' + n.id} style={{ marginTop: 6 }}>
                                                            <div style={{ fontWeight: 600 }}>{n.title || `Node #${n.id}`}</div>
                                                            {(ch.slotSummaries[n.id] ?? []).length === 0 ? (
                                                                <div style={{ opacity: 0.7 }}>No input slots.</div>
                                                            ) : (
                                                                <div>
                                                                    {ch.slotSummaries[n.id].map((s) => (
                                                                        <div key={n.id + '_s_' + s.slot} style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                                                                            <Chip text={`#${s.slot}`} />
                                                                            <Chip text={s.name || '(unnamed)'} />
                                                                            <Chip text={s.type || '*'} />
                                                                            <Chip text={s.linked ? 'linked' : 'free'} kind={s.linked ? 'ok' : 'default'} />
                                                                        </div>
                                                                    ))}
                                                                </div>
                                                            )}
                                                        </div>
                                                    ))}
                                                </details>
                                            </div>
                                        ))}
                                    </div>

                                </div>
                            </>
                        );
                    }

                    el.innerHTML = '';
                    mountJSX(el, <FlowInspector />);
                }
            }
        ],
        async beforeRegisterNodeDef(nodeType, nodeData, app) {
            if (nodeType.comfyClass === MaraScottAnyBusNodeWidget.NAMESPACE + MaraScottAnyBusNodeWidget.TYPE) {

                // On creation: initialize widgets and IO
                const onNodeCreated = nodeType.prototype.onNodeCreated;
                nodeType.prototype.onNodeCreated = function () {
                    MaraScottAnyBusNodeWidget.init(this);
                    MaraScottAnyBusNodeWidget.setWidgets(this);
                    MaraScottAnyBusNodeWidget.updateNodeIO(this);
                    // Ensure widget values reflect properties on creation
                    MaraScottAnyBusNodeWidget.setWidgetValue(this, MaraScottAnyBusNodeWidget.PROFILE.name, this.properties[MaraScottAnyBusNodeWidget.PROFILE.name]);
                    MaraScottAnyBusNodeWidget.setWidgetValue(this, MaraScottAnyBusNodeWidget.INPUTS.name, this.properties[MaraScottAnyBusNodeWidget.INPUTS.name]);
                    onNodeCreated?.apply(this, arguments);
                    this.serialize_widgets = true;
                    state.init = true;
                };

                // Handle connections
                const onConnectionsChange = nodeType.prototype.onConnectionsChange;
                nodeType.prototype.onConnectionsChange = function (type, slotIndex, isConnected, link, ioSlot) {
                    if (!this.graph) return;

                    // Update "current input index" (1-based display)
                    state.input.index = slotIndex + 1 - MaraScottAnyBus_v2.FIRST_INDEX;

                    if (isConnected && link) {
                        const originNode = this.graph.getNodeById(link.origin_id);
                        if (originNode) {
                            if (slotIndex === MaraScottAnyBus_v2.BUS_SLOT) {
                                // Only bus→bus allowed AND enforce profile condition
                                if (originNode.type === ANYBUS_TYPE && link.origin_slot === MaraScottAnyBus_v2.BUS_SLOT) {
                                    MaraScottAnyBus_v2.connectBus(this, slotIndex, originNode, link.origin_slot);
                                } else {
                                    this.disconnectInput(slotIndex); // invalid bus connection
                                }
                            } else {
                                MaraScottAnyBus_v2.connectInput(this, slotIndex, originNode, link.origin_slot);
                            }
                        }
                    } else if (!isConnected) {
                        if (slotIndex === MaraScottAnyBus_v2.BUS_SLOT) {
                            MaraScottAnyBus_v2.disConnectBus(this);
                        } else {
                            MaraScottAnyBus_v2.disConnectInput(this, slotIndex);
                        }
                    }

                    // Keep title/inputs consistent ONLY when we still have a bus link
                    if (slotIndex === MaraScottAnyBus_v2.BUS_SLOT && this.inputs?.[MaraScottAnyBus_v2.BUS_SLOT]?.link != null) {
                        syncTitleAndInputsFrom(this);
                    }
                    notifyAnyBusChange();
                    onConnectionsChange?.apply(this, arguments);
                };
            }
        },
        async setup() {
            console.log("[AnyBus_v2] extension setup complete");
        }
    };
};

const MaraScottAnyBusNodeSidebarTab = () => {
    return {
        id: nodeId,
        icon: "mdi mdi-vector-polyline",
        title: "Any Bus",
        tooltip: "Any Bus Dashboard",
        type: "custom",
        render: async (el) => {
            await ensureReactGlobals();
            const React = globalThis.React;

            function FlowSidebar() {
                const [data, setData] = React.useState({
                    flows: [],
                    totalFlows: 0,
                    nodesCount: 0,
                    nodes: [],
                    last: 0,
                });

                const compute = React.useCallback(() => {
                    const graph = getActiveGraph();
                    if (!graph) { setData({ flows: [], total: 0, scanned: 0, last: Date.now() }); return; }

                    const byProfile = collectAnyBusByProfile(
                        graph,
                        MaraScottAnyBusNodeWidget.PROFILE.name,
                        MaraScottAnyBusNodeWidget.PROFILE.default
                    );
                    const flows = [];
                    let scanned = 0;

                    for (const [profile, nodes] of byProfile.entries()) {
                        scanned += nodes.length;
                        const chains = chainsForProfile(nodes, graph).map((chain, idx) => {
                            const edges = [];
                            for (let i = 0; i < chain.length - 1; i++) {
                                const e = busEdgeInfo(graph, chain[i], chain[i + 1]);
                                edges.push({ from: chain[i], to: chain[i + 1], linkId: e.id, valid: e.valid });
                            }
                            const slotSummaries = {};
                            for (const n of chain) slotSummaries[n.id] = summarizeSlots(n);
                            return { nodes: chain, edges, slotSummaries, index: idx + 1 };
                        });
                        flows.push({ profile, count: chains.length, chains });
                    }

                    setData({ flows, total: flows.length, scanned, last: Date.now() });
                }, []);

                React.useEffect(() => {
                    const handler = () => compute();
                    window.addEventListener('marascott:anybus:changed', handler);
                    return () => window.removeEventListener('marascott:anybus:changed', handler);
                }, [compute]);

                return (
                    <>
                        <div style={{ fontFamily: 'sans-serif', fontSize: 12, padding: 8 }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8 }}>
                                <strong>AnyBus Flows</strong>
                                <Chip text={`${data.total} flow(s)`} kind={data.total ? 'ok' : 'warn'} />
                                <Chip text={`${data.scanned} node(s)`} />
                                <div style={{ flex: 1 }} />
                                <button onClick={compute} style={{ padding: '2px 8px', cursor: 'pointer' }}>Refresh</button>
                            </div>

                            {data.flows.length === 0 && <div style={{ opacity: 0.7 }}>No AnyBus nodes detected.</div>}

                            {data.flows.map((flow) => (
                                <details key={flow.profile} open style={{ borderTop: '1px solid #e5e7eb', paddingTop: 6, marginTop: 6 }}>
                                    <summary style={{ cursor: 'pointer' }}>
                                        <span>Profile: </span><strong>{flow.profile}</strong> <Chip text={`${flow.count} chain(s)`} />
                                    </summary>

                                    {flow.chains.map((ch) => (
                                        <div
                                            key={flow.profile + '_chain_' + ch.index}
                                            style={{ padding: '6px 8px', border: '1px solid #e5e7eb', borderRadius: 6, marginBottom: 6, background: '#fafafa' }}
                                        >
                                            {/* Chain visualization (compact) */}
                                            <div style={{ marginBottom: 6, overflowX: 'auto', whiteSpace: 'nowrap' }}>
                                                {ch.nodes.map((n, i) => (
                                                    <React.Fragment key={n.id + '_frag'}>
                                                        <Chip text={n.title || (`Node #${n.id}`)} />
                                                        {i < ch.nodes.length - 1 && <span style={{ margin: '0 6px' }}>→</span>}
                                                    </React.Fragment>
                                                ))}
                                            </div>

                                            {/* Edges with bus link IDs (compact) */}
                                            <div style={{ marginBottom: 6 }}>
                                                <div style={{ fontWeight: 600, marginBottom: 2 }}>Links</div>
                                                {ch.edges.length === 0 ? (
                                                    <div style={{ opacity: 0.7 }}>No BUS edges.</div>
                                                ) : (
                                                    ch.edges.map((e, i) => (
                                                        <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                                                            <span>{(e.from.title || e.from.id)} → {(e.to.title || e.to.id)}</span>
                                                            <Chip text={`id: ${e.linkId ?? 'n/a'}`} kind={e.valid ? 'ok' : 'warn'} />
                                                        </div>
                                                    ))
                                                )}
                                            </div>

                                            {/* Slot summaries (collapsible per node for tight UI) */}
                                            <details>
                                                <summary style={{ cursor: 'pointer', fontWeight: 600 }}>Slots</summary>
                                                {ch.nodes.map((n) => (
                                                    <div key={'slots_' + n.id} style={{ marginTop: 4 }}>
                                                        <div style={{ fontWeight: 600 }}>{n.title || `Node #${n.id}`}</div>
                                                        {(ch.slotSummaries[n.id] ?? []).length === 0 ? (
                                                            <div style={{ opacity: 0.7 }}>No input slots.</div>
                                                        ) : (
                                                            <div>
                                                                {ch.slotSummaries[n.id].map((s) => (
                                                                    <div key={n.id + '_s_' + s.slot} style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                                                                        <Chip text={`#${s.slot}`} />
                                                                        <Chip text={s.name || '(unnamed)'} />
                                                                        <Chip text={s.type || '*'} />
                                                                        <Chip text={s.linked ? 'linked' : 'free'} kind={s.linked ? 'ok' : 'default'} />
                                                                    </div>
                                                                ))}
                                                            </div>
                                                        )}
                                                    </div>
                                                ))}
                                            </details>
                                        </div>
                                    ))}
                                </details>
                            ))}
                        </div>
                    </>
                );
            }

            el.innerHTML = '';
            mountJSX(el, <FlowSidebar />);
        }
    };
};

export { MaraScottAnyBusNodeExtension, MaraScottAnyBusNodeSidebarTab };
