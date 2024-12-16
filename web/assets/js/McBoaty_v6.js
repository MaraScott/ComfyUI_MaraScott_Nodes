import { app } from "/scripts/app.js";

app.registerExtension({
    name: "ComfyUI.MaraScott.McBoaty_v6",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (["MaraScottMcBoatyConfigurator_v6", "MaraScottMcBoaty_v6"].includes(nodeData.name)) {
            const originalNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = async function () {
                if (originalNodeCreated) {
                    originalNodeCreated.apply(this, arguments);
                }

                const urlWidget = this.widgets.find((w) => w.name === "ollama_url");
                const modelWidget = this.widgets.find((w) => w.name === "vlm_model");

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
            };
        }
        if (["McBoatyTilePrompter_v6"].includes(nodeData.name)) {
            const originalNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = async function () {
                if (originalNodeCreated) {
                    originalNodeCreated.apply(this, arguments);
                }

                const tilesWidget = this.widgets.find((w) => w.name === "tiles_to_process");
                const positiveWidget = this.widgets.find((w) => w.name === "positive");
                const negativeWidget = this.widgets.find((w) => w.name === "negative");

                const fetchPrompts = async (tiles) => {
                    try {
                        const response = await fetch("/marascott/McBoaty_v6/get_prompts", {
                            method: "POST",
                            headers: {
                                "Content-Type": "application/json",
                            },
                            body: JSON.stringify({
                                tiles,
                            }),
                        });

                        if (response.ok) {
                            const prompts = await response.json();
                            console.debug("Fetched prompts:", prompts);
                            return models;
                        } else {
                            console.error(`Failed to fetch prompts: ${response.status}`);
                            return [];
                        }
                    } catch (error) {
                        console.error(`Error fetching prompts`, error);
                        return [];
                    }
                };

            }

            const updatePrompts = async () => {
                const positive = positiveWidget.value;
                positiveWidget.value = ''
                const negative = negativeWidget.value;
                negativeWidget.value = ''
                const tiles = tilesWidget.value

                const prompts = await fetchPrompts(tiles);

                // Update modelWidget options and value
                console.debug("Updated modelWidget.options.values:", prompts);

                positiveWidget.value = prompts[0].positive
                negativeWidget.value = prompts[0].negative

                console.debug("Updated modelWidget.value:", positiveWidget.value);
            };

            urlWidget.callback = updatePrompts;

            const dummy = async () => {
                // calling async method will update the widgets with actual value from the browser and not the default from Node definition.
            }

            // Initial update
            await dummy(); // this will cause the widgets to obtain the actual value from web page.
            await updateModels();
        }
    },
});
