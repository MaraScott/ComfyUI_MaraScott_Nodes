import { app } from "/scripts/app.js";

class MaraScottMcBoatyv6NodeLiteGraph {

    static async setPrompts(node) {

        const nodeGraphLinks = Array.isArray(node.graph.links)
        ? node.graph.links
        : Object.values(node.graph.links);

        const parentLink = nodeGraphLinks.find(
            (otherLink) => otherLink?.id == node.inputs[0].link
        );

        if (parentLink != undefined) {
            
            const refreshWidget = node.widgets.find((w) => w.name === "refresh_prompts");
            const tilesWidget = node.widgets.find((w) => w.name === "tiles_to_process");
            const positiveWidget = node.widgets.find((w) => w.name === "positive");
            const negativeWidget = node.widgets.find((w) => w.name === "negative");
    
            const fetchPrompts = async (tiles) => {
                try {
                    const response = await fetch("/marascott/McBoaty_v6/get_prompts", {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/json",
                        },
                        body: JSON.stringify({
                            "tiles":tiles,
                            "parentId":parentLink.origin_id,
                        }),
                    });
    
                    if (response.ok) {
                        const prompts = await response.json();
                        console.debug("Fetched prompts:", prompts);
                        return prompts;
                    } else {
                        console.error(`Failed to fetch prompts: ${response.status}`);
                        return [];
                    }
                } catch (error) {
                    console.error(`Error fetching prompts`, error);
                    return [];
                }
            };
    
            const refreshPrompts = async () => {
    
                const positive = positiveWidget.value;
                const negative = negativeWidget.value;
                const tiles = tilesWidget != undefined ? tilesWidget.value : ""    
                const prompts = await fetchPrompts(tiles);                
                positiveWidget.value == "" ? positiveWidget.value = prompts.positive : null
                negativeWidget.value == "" ? negativeWidget.value = prompts.negative : null
                setTimeout(() => {
                    refreshWidget.value = false
                }, 50);

            };
            const updatePrompts = async () => {
                refreshWidget.value = true
                refreshPrompts();
            };
    
            if(tilesWidget != undefined) tilesWidget.callback = updatePrompts;
            refreshWidget.callback = refreshPrompts;
    
            const dummy = async () => {
                // calling async method will update the widgets with actual value from the browser and not the default from Node definition.
            }
    
            // Initial update
            await dummy(); // this will cause the widgets to obtain the actual value from web page.
            await updatePrompts();    

        }

    }

    static async setModels(node) {

        const urlWidget = node.widgets.find((w) => w.name === "ollama_url");
        const modelWidget = node.widgets.find((w) => w.name === "vlm_model");

        const fetchModels = async (url) => {
            try {
                const response = await fetch("/marascott/ollama/get_models", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify({
                        url,
                    }),
                });

                if (response.ok) {
                    const models = await response.json();
                    console.debug("Fetched models:", models);
                    return models;
                } else {
                    console.error(`Failed to fetch models: ${response.status}`);
                    return [];
                }
            } catch (error) {
                console.error(`Error fetching models`, error);
                return [];
            }
        };

        const updateModels = async () => {
            const url = urlWidget.value;
            const prevValue = modelWidget.value
            modelWidget.value = ''
            modelWidget.options.values = []

            const models = await fetchModels(url);

            // Update modelWidget options and value
            modelWidget.options.values = models;
            console.debug("Updated modelWidget.options.values:", modelWidget.options.values);

            if (models.includes(prevValue)) {
                modelWidget.value = prevValue; // stay on current.
            } else if (models.length > 0) {
                modelWidget.value = models[0]; // set first as default.
            }

            console.debug("Updated modelWidget.value:", modelWidget.value);
        };

        urlWidget.callback = updateModels;

        const dummy = async () => {
            // calling async method will update the widgets with actual value from the browser and not the default from Node definition.
        }

        // Initial update
        await dummy(); // this will cause the widgets to obtain the actual value from web page.
        await updateModels();

    }

}

app.registerExtension({
    name: "ComfyUI.MaraScott.McBoaty_v6",

    async loadedGraphNode(node, app) {
        if (["MaraScottMcBoatyTilePrompter_v6", "MaraScottUntiler_v1"].includes(node.type)) {
            MaraScottMcBoatyv6NodeLiteGraph.setPrompts(node)
        }
    },

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (["MaraScottMcBoatyConfigurator_v6", "MaraScottMcBoaty_v6"].includes(nodeData.name)) {
            const originalNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = async function () {
                if (originalNodeCreated) {
                    originalNodeCreated.apply(this, arguments);
                }
                MaraScottMcBoatyv6NodeLiteGraph.setModels(this)
            }
        }

        if (["MaraScottMcBoatyTilePrompter_v6"].includes(nodeData.name)) {
            const onConnectionsChange = nodeType.prototype.onConnectionsChange            
            nodeType.prototype.onConnectionsChange = async function (slotType, slot, isChangeConnect, link_info, output) {
                if (onConnectionsChange) {
                    onConnectionsChange.apply(this, arguments);
                }
                MaraScottMcBoatyv6NodeLiteGraph.setPrompts(this)
            }
            const onExecuted = nodeType.prototype.onExecuted
            nodeType.prototype.onExecuted = function (message) {
                if (onExecuted) {
                    onExecuted.apply(this, arguments);
                    MaraScottMcBoatyv6NodeLiteGraph.setPrompts(this)
                }
            }
        
        }
    },

});
