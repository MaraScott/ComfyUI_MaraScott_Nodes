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

import { getAnyBusState, setAnyBusState, useAnyBusState } from "./AnyBus_v2/State.jsx";
import { ensureReactGlobals, mountJSX } from "./AnyBus_v2/React.jsx";
import { getActiveGraph, getGraphLinkById } from "./AnyBus_v2/Graph.jsx";

import { ANYBUS_TYPE, init as NodeInit } from "./AnyBus_v2/Node.jsx";
import { init as WidgetInit } from "./AnyBus_v2/Widget.jsx";

// import { MaraScottAnyBusNodeWidget } from "./AnyBus_v2/Bus.jsx";

function notifyAnyBusChange() {
    setAnyBusState((s) => ({ ...s, sync: (s.sync || 0) + 1 }));
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
                        return (
                            <>
                                <div>
                                    Mara Scott
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
            if (nodeType.comfyClass === ANYBUS_TYPE) {

                // On creation: initialize widgets and IO
                const onNodeCreated = nodeType.prototype.onNodeCreated;
                onNodeCreated?.apply(this, arguments);
                nodeType.prototype.onNodeCreated = function() {
                    globalThis.console.log("TEST",this.addWidget);
                    globalThis.console.log("TEST2",this.addProperty);
                    NodeInit(this);
                    WidgetInit(this);
                    // MaraScottAnyBusNodeWidget.setWidgets(node);
                    // MaraScottAnyBusNodeWidget.updateNodeIO(node);
                    // // Ensure widget values reflect properties on creation
                    // MaraScottAnyBusNodeWidget.setWidgetValue(node, MaraScottAnyBusNodeWidget.PROFILE.name, node.properties[MaraScottAnyBusNodeWidget.PROFILE.name]);
                    // MaraScottAnyBusNodeWidget.setWidgetValue(node, MaraScottAnyBusNodeWidget.INPUTS.name, node.properties[MaraScottAnyBusNodeWidget.INPUTS.name]);
                    // node.serialize_widgets = true;
                    // setAnyBusState(s => ({ ...s, init: true }));
                    // notifyAnyBusChange();
                }

                // Handle connections
                const onConnectionsChange = nodeType.prototype.onConnectionsChange;
                onConnectionsChange?.apply(this, arguments);
                nodeType.prototype.onConnectionsChange = function(type, slotIndex, isConnected, link, ioSlot) {
                    if (!this.graph) return;

                    // Update "current input index" (1-based display)
                    // setAnyBusState(s => ({
                    //     ...s,
                    //     input: { ...s.input, index: slotIndex + 1 - MaraScottAnyBus_v2.FIRST_INDEX },
                    // }));

                    // if (isConnected && link) {
                    //     const originNode = this.graph.getNodeById(link.origin_id);
                    //     if (originNode) {
                    //         if (slotIndex === MaraScottAnyBus_v2.BUS_SLOT) {
                    //             // Only bus→bus allowed AND enforce profile condition
                    //             if (originNode.type === ANYBUS_TYPE && link.origin_slot === MaraScottAnyBus_v2.BUS_SLOT) {
                    //                 MaraScottAnyBus_v2.connectBus(this, slotIndex, originNode, link.origin_slot);
                    //             } else {
                    //                 this.disconnectInput(slotIndex); // invalid bus connection
                    //             }
                    //         } else {
                    //             MaraScottAnyBus_v2.connectInput(this, slotIndex, originNode, link.origin_slot);
                    //         }
                    //     }
                    // } else if (!isConnected) {
                    //     if (slotIndex === MaraScottAnyBus_v2.BUS_SLOT) {
                    //         MaraScottAnyBus_v2.disConnectBus(this);
                    //     } else {
                    //         MaraScottAnyBus_v2.disConnectInput(this, slotIndex);
                    //     }
                    // }

                    // // Keep title/inputs consistent ONLY when we still have a bus link
                    // if (slotIndex === MaraScottAnyBus_v2.BUS_SLOT && this.inputs?.[MaraScottAnyBus_v2.BUS_SLOT]?.link != null) {
                    //     syncTitleAndInputsFrom(this);
                    // }
                    notifyAnyBusChange();
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
                const [busState] = useAnyBusState();

                const compute = React.useCallback(() => {
                    const graph = getActiveGraph();
                    if (!graph) {
                        setData({ flows: [], total: 0, scanned: 0, last: Date.now() });
                        setAnyBusState((s) => ({ ...s, flows: { start: [], list: [], end: [] }, nodes: {} }));
                        return;
                    }

                    const byProfile = collectAnyBusByProfile(graph);
                    const flows = [];
                    const startSet = new Set();
                    const listSet = new Set();
                    const endSet = new Set();
                    const nodesMap = {};
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
                            for (const n of chain) {
                                slotSummaries[n.id] = summarizeSlots(n);
                                listSet.add(n.id);
                                nodesMap[n.id] = { id: n.id, title: n.title || `Node #${n.id}` };
                            }
                            if (chain.length) {
                                startSet.add(chain[0].id);
                                endSet.add(chain[chain.length - 1].id);
                            }
                            return { nodes: chain, edges, slotSummaries, index: idx + 1 };
                        });
                        flows.push({ profile, count: chains.length, chains });
                    }

                    setData({ flows, total: flows.length, scanned, last: Date.now() });
                    setAnyBusState((s) => ({
                        ...s,
                        flows: { start: Array.from(startSet), list: Array.from(listSet), end: Array.from(endSet) },
                        nodes: nodesMap,
                    }));
                }, []);

                React.useEffect(() => {
                    compute();
                }, [busState.sync, compute]);

                return (
                    <>
                        <div>
                            <div>
                                <strong>AnyBus State</strong>
                                <pre>{JSON.stringify(busState, null, 2)}</pre>
                            </div>

                            <div>
                                <strong>AnyBus Flows</strong>
                                <Chip text={`${data.total} flow(s)`} kind={data.total ? 'ok' : 'warn'} />
                                <Chip text={`${data.scanned} node(s)`} />
                                <div />
                                <button onClick={compute} style={{ padding: '2px 8px', cursor: 'pointer' }}>Refresh</button>
                            </div>

                            {data.flows.map((flow) => (
                                <details key={flow.profile} open>
                                    <summary>
                                        <span>Profile: </span><strong>{flow.profile}</strong> <Chip text={`${flow.count} chain(s)`} />
                                    </summary>

                                    {flow.chains.map((ch) => (
                                        <div
                                            key={flow.profile + '_chain_' + ch.index}
                                        >
                                            <div>
                                                {ch.nodes.map((n, i) => (
                                                    <React.Fragment key={n.id + '_frag'}>
                                                        <Chip text={n.title || (`Node #${n.id}`)} />
                                                        {i < ch.nodes.length - 1 && <span>→</span>}
                                                    </React.Fragment>
                                                ))}
                                            </div>

                                            <div>
                                                <div>Links</div>
                                                {ch.edges.length === 0 ? (
                                                    <div>No BUS edges.</div>
                                                ) : (
                                                    ch.edges.map((e, i) => (
                                                        <div key={i}>
                                                            <span>{(e.from.title || e.from.id)} → {(e.to.title || e.to.id)}</span>
                                                            <Chip text={`id: ${e.linkId ?? 'n/a'}`} kind={e.valid ? 'ok' : 'warn'} />
                                                        </div>
                                                    ))
                                                )}
                                            </div>

                                            <details>
                                                <summary>Slots</summary>
                                                {ch.nodes.map((n) => (
                                                    <div key={'slots_' + n.id}>
                                                        <div>{n.title || `Node #${n.id}`}</div>
                                                        {(ch.slotSummaries[n.id] ?? []).length === 0 ? (
                                                            <div>No input slots.</div>
                                                        ) : (
                                                            <div>
                                                                {ch.slotSummaries[n.id].map((s) => (
                                                                    <div key={n.id + '_s_' + s.slot}>
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
