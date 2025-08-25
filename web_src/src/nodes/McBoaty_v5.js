// Ensure global namespace
if (!window.marascott) {
    window.marascott = {};
}
if (!window.marascott.McBoaty_v5) {
    window.marascott.McBoaty_v5 = {
        init: false,
        clean: false,
        params: {
            denoise: { values: [] },
        },
        message: {
            prompts: [],
            tiles: [],
            denoises: [],
        },
        inputs: {
            prompts: [],
            tiles: [],
            denoises: [],
        },
    };
}

// Use window.api/window.app to avoid "api is not defined" in loose contexts
function imageDataToUrl(data) {
    return window.api.apiURL(
        `/view?filename=${encodeURIComponent(data.filename)}&type=${data.type}&subfolder=${data.subfolder}${window.app.getPreviewFormatParam()}${window.app.getRandParam()}`
    );
}

function clearProcessedFlag() {
    document.querySelectorAll('textarea[data-processed]').forEach((textarea) => {
        textarea.removeAttribute('data-processed');
    });
}

export const McBoatyWidgets = {
    validateDenoiseValue: (value) => {
        if (value === "") return true;
        const num = parseFloat(value);
        return !isNaN(num) && num >= 0.0 && num <= 1.0;
    },

    formatDenoiseValue: (value) => {
        if (value === "") return value;
        let strValue = value.toString();
        if (strValue.startsWith(".")) strValue = "0" + strValue;
        return parseFloat(strValue).toFixed(2);
    },

    WRAPPER: (key, index, prompt, tile, denoise, node) => {
        const inputEl = document.createElement("div");
        inputEl.className = "comfy-wrapper-mcboaty";

        const wrapper = document.createElement("div");
        wrapper.style.height = "100%";
        wrapper.style.display = "flex";
        wrapper.style.alignItems = "center";
        wrapper.style.gap = "10px";

        const text = document.createElement("p");
        text.textContent = String(index + 1).padStart(2, "0");

        const textarea = document.createElement("textarea");
        textarea.style.opacity = 0.6;
        textarea.style.flexGrow = 1;
        textarea.style.height = "100%";
        textarea.className = "comfy-multiline-input";
        textarea.value = prompt || "";
        textarea.placeholder = "tile " + text.textContent;
        textarea.dataId = "tile " + index;
        textarea.dataNodeId = node.id;

        textarea.addEventListener("focusout", async function () {
            this.value = this.value.trim();
            if (window.marascott.McBoaty_v5.message.prompts[index] != this.value) {
                window.marascott.McBoaty_v5.message.prompts[index] = this.value;
                await (await fetch(`/MaraScott/McBoaty/v5/set_prompt?index=${index}&prompt=${encodeURIComponent(this.value)}&node=${this.dataNodeId}&clientId=${window.api.clientId}`)).json();
                const nodeWidget = MaraScottMcBoatyNodeWidget.getByName(node, "requeue");
                if (nodeWidget) {
                    MaraScottMcBoatyNodeWidget.setValue(node, "requeue", ++nodeWidget.value);
                }
            }
        });

        const input = document.createElement("input");
        input.style.opacity = 0.6;
        input.style.height = "100%";
        input.style.maxWidth = "1.8rem";
        input.style.flexShrink = "0";
        input.className = "comfy-multiline-input";
        input.value = denoise || "";
        input.placeholder = "denoise " + text.textContent;
        input.dataId = "tile " + index;
        input.dataNodeId = node.id;

        input.addEventListener("focusout", async function () {
            this.value = this.value.trim();
            if (!McBoatyWidgets.validateDenoiseValue(this.value)) {
                this.value = "";
            } else {
                this.value = McBoatyWidgets.formatDenoiseValue(this.value);
            }
            if (window.marascott.McBoaty_v5.message.denoises[index] != this.value) {
                window.marascott.McBoaty_v5.message.denoises[index] = this.value;
                await (await fetch(`/MaraScott/McBoaty/v5/set_denoise?index=${index}&denoise=${encodeURIComponent(this.value)}&node=${this.dataNodeId}&clientId=${window.api.clientId}`)).json();
                const nodeWidget = MaraScottMcBoatyNodeWidget.getByName(node, "requeue");
                if (nodeWidget) {
                    MaraScottMcBoatyNodeWidget.setValue(node, "requeue", ++nodeWidget.value);
                }
            }
        });

        const img = document.createElement("img");
        img.src = imageDataToUrl(tile);
        img.alt = prompt;
        img.style.height = "100%";
        img.style.maxWidth = "128px";
        img.style.maxHeight = "128px";
        img.style.flexShrink = "0";

        wrapper.appendChild(text);
        wrapper.appendChild(img);
        wrapper.appendChild(textarea);
        wrapper.appendChild(input);
        inputEl.appendChild(wrapper);

        // FIX: use provided `key` instead of undefined `name`
        const widget = node.addDOMWidget(key, "customtext", inputEl, {
            getValue() {
                return inputEl.value;
            },
            setValue(v) {
                inputEl.value = v;
            },
        });
        widget.inputEl = inputEl;
        MaraScottMcBoatyNodeWidget.setValue(node, widget.name, prompt);

        textarea.addEventListener("input", () => {
            widget.callback?.(widget.value);
        });

        return widget;
    },
};

class MaraScottMcBoatyNodePrompter {
    static async clean(node) {
        window.marascott.McBoaty_v5.clean = false;

        const cleanedLabel = " ... cleaned";
        const nodeTitle = node.title;
        node.title = nodeTitle + cleanedLabel;

        const res_prompts = await (await fetch(`/MaraScott/McBoaty/v5/get_input_prompts?node=${node.id}`)).json();
        const res_denoises = await (await fetch(`/MaraScott/McBoaty/v5/get_input_denoises?node=${node.id}`)).json();

        window.marascott.McBoaty_v5.message.prompts = window.marascott.McBoaty_v5.inputs.prompts = res_prompts.prompts_in;
        window.marascott.McBoaty_v5.inputs.denoises = window.marascott.McBoaty_v5.message.denoises = res_denoises.denoises_in;
        window.marascott.McBoaty_v5.message.tiles = window.marascott.McBoaty_v5.inputs.tiles;

        MaraScottMcBoatyNodeWidget.setValue(node, MaraScottMcBoatyNodeWidget.INDEX.name, MaraScottMcBoatyNodeWidget.INDEX.default);
        MaraScottMcBoatyNodeWidget.setValue(node, MaraScottMcBoatyNodeWidget.PROMPT.name, MaraScottMcBoatyNodeWidget.PROMPT.default);
        MaraScottMcBoatyNodeWidget.setValue(node, MaraScottMcBoatyNodeWidget.DENOISE.name, MaraScottMcBoatyNodeWidget.DENOISE.default);
        MaraScottMcBoatyNodeWidget.setValue(node, "requeue", 0);

        node.widgets = node.widgets.filter((widget) => {
            const focusOutEvent = new Event("focusout");
            if (widget.type == "customtext") {
                const textarea = widget.inputEl.querySelector('[placeholder^="tile "]');
                if (textarea != null) {
                    const dataId = textarea.getAttribute("placeholder");
                    const indexValue = parseInt(dataId.replace("tile ", ""), 10);
                    const realIndexValue = indexValue - 1;
                    const value = window.marascott.McBoaty_v5.inputs.prompts[realIndexValue];
                    MaraScottMcBoatyNodeWidget.setValue(node, widget.name, value);
                    textarea.value = value;
                    textarea.dispatchEvent(focusOutEvent);
                }
            }
            return true;
        });

        MaraScottMcBoatyNodeWidget.refresh(node);
        setTimeout(() => {
            node.title = nodeTitle;
        }, 500);
    }
}

class MaraScottMcBoatyNodeWidget {
    static INDEX = { name: "Filter by Indexes", default: "" };
    static PROMPT = { name: "Prompt", default: "" };
    static DENOISE = {
        name: "Denoise",
        values: ["unchanged", "Use Global Denoise"],
        default: "unchanged",
        min: 0.0,
        max: 1.0,
        step: 0.01,
    };
    static CLEAN = { name: "Reset", default: false };

    static refresh(node) {
        node.widgets = node.widgets.filter((widget) => {
            if (widget.type == "customtext") {
                widget.onRemove?.();
                return false;
            }
            if (widget.name === this.CLEAN.name) {
                widget.onRemove?.();
                return false;
            }
            return true;
        });

        this.setIndexInput(node);
        this.setPromptInput(node);
        this.setDenoiseInput(node);

        node.onResize?.(node.size);
        node.graph.setDirtyCanvas(true, true);

        this.setPrompterInputs(node);
        this.setCleanSwitch(node);

        node.onResize?.(node.size);
        node.graph.setDirtyCanvas(true, true);
    }

    static init(node) {
        this.setIndexInput(node);
        this.setPromptInput(node);
        this.setDenoiseInput(node);
        this.setCleanSwitch(node);
    }

    static getByName(node, name) {
        return node.widgets?.find((w) => w.name === name);
    }

    static setValue(node, name, value) {
        const nodeWidget = this.getByName(node, name);
        if (!nodeWidget) return;
        nodeWidget.value = value;
        node.setProperty(name, nodeWidget.value ?? node.properties[name]);
        node.setDirtyCanvas(true);
    }

    static setIndexInput(node) {
        const nodeWidget = this.getByName(node, this.INDEX.name);
        if (nodeWidget == undefined) {
            node.addWidget(
                "text",
                this.INDEX.name,
                this.INDEX.default,
                (value) => {
                    this.setValue(node, this.INDEX.name, value);
                    this.setValue(node, this.PROMPT.name, this.PROMPT.default);
                    this.setValue(node, this.DENOISE.name, this.DENOISE.default);
                    this.refresh(node);
                },
                {}
            );
            this.setValue(node, this.INDEX.name, this.INDEX.default);
        }
    }

    static setPromptInput(node) {
        const nodeWidget = this.getByName(node, this.PROMPT.name);
        if (nodeWidget == undefined) {
            node.addWidget(
                "text",
                this.PROMPT.name,
                this.PROMPT.default,
                (value) => {
                    this.setValue(node, this.PROMPT.name, value);

                    const input_list = node.properties[this.INDEX.name] ?? this.INDEX.default;
                    node.widgets = node.widgets.filter((widget) => {
                        const focusOutEvent = new Event("focusout");
                        const index_filtered = input_list.split(",").map((num) => Number(num) - 1);
                        if (widget.type == "customtext") {
                            const textarea = widget.inputEl.querySelector('[placeholder^="tile "]');
                            if (textarea != null) {
                                const dataId = textarea.getAttribute("placeholder");
                                const indexValue = parseInt(dataId.replace("tile ", ""), 10);
                                const realIndexValue = indexValue - 1;
                                const indexFound = index_filtered.indexOf(realIndexValue);
                                if ((input_list != "" && indexFound > -1) || input_list == "") {
                                    textarea.value = value;
                                    textarea.dispatchEvent(focusOutEvent);
                                }
                            }
                        }
                        return true;
                    });
                    this.setValue(node, this.PROMPT.name, this.PROMPT.default);
                },
                {}
            );
            this.setValue(node, this.PROMPT.name, this.PROMPT.default);
        }
    }

    static setDenoiseInput(node) {
        const nodeWidget = this.getByName(node, this.DENOISE.name);
        if (nodeWidget == undefined) {
            node.addWidget(
                "combo",
                this.DENOISE.name,
                this.DENOISE.default,
                (value) => {
                    this.setValue(node, this.DENOISE.name, value);

                    if (value != "unchanged") {
                        if (value == "Use Global Denoise") value = "";

                        const input_list = node.properties[this.INDEX.name] ?? this.INDEX.default;
                        node.widgets = node.widgets.filter((widget) => {
                            const focusOutEvent = new Event("focusout");
                            const index_filtered = input_list.split(",").map((num) => Number(num) - 1);
                            if (widget.type == "customtext") {
                                const input = widget.inputEl.querySelector('[placeholder^="denoise "]');
                                if (input != null) {
                                    const dataId = input.getAttribute("placeholder");
                                    const indexValue = parseInt(dataId.replace("denoise ", ""), 10);
                                    const realIndexValue = indexValue - 1;
                                    const indexFound = index_filtered.indexOf(realIndexValue);
                                    if ((input_list != "" && indexFound > -1) || input_list == "") {
                                        input.value = value;
                                        input.dispatchEvent(focusOutEvent);
                                    }
                                }
                            }
                            return true;
                        });
                        this.setValue(node, this.DENOISE.name, this.DENOISE.default);
                    }
                },
                {
                    values: window.marascott.McBoaty_v5.params.denoise.values,
                }
            );
            this.setValue(node, this.DENOISE.name, this.DENOISE.default);
        }
    }

    static setCleanSwitch(node) {
        const nodeWidget = this.getByName(node, this.CLEAN.name);
        if (nodeWidget == undefined) {
            node.addWidget(
                "toggle",
                this.CLEAN.name,
                this.CLEAN.default,
                () => {
                    this.setValue(node, this.CLEAN.name, this.CLEAN.default);
                    MaraScottMcBoatyNodePrompter.clean(node);
                },
                {}
            );
            this.setValue(node, this.CLEAN.name, this.CLEAN.default);
        }
    }

    static setPrompterInputs(node) {
        let index_list = (node.properties[this.INDEX.name] ?? this.INDEX.default).trim().split(",");
        index_list = index_list.length === 1 && index_list[0] === "" ? [] : index_list;
        const index_filtered = index_list.map((num) => Number(num) - 1);

        for (const [index] of window.marascott.McBoaty_v5.message.prompts.entries()) {
            if (index_list.length == 0 || index_filtered.indexOf(index) > -1) {
                McBoatyWidgets.WRAPPER(
                    "tile " + index,
                    index,
                    window.marascott.McBoaty_v5.message.prompts[index],
                    window.marascott.McBoaty_v5.message.tiles[index],
                    window.marascott.McBoaty_v5.message.denoises[index],
                    node
                );
            }
        }
    }
}

// Populate denoise dropdown values
window.marascott.McBoaty_v5.params.denoise.values = MaraScottMcBoatyNodeWidget.DENOISE.values.slice();
for (
    let i = MaraScottMcBoatyNodeWidget.DENOISE.min;
    i <= MaraScottMcBoatyNodeWidget.DENOISE.max;
    i = parseFloat((i + MaraScottMcBoatyNodeWidget.DENOISE.step).toFixed(2))
) {
    window.marascott.McBoaty_v5.params.denoise.values.push(i.toFixed(2));
}

// --- Fixed: accept and store api/app; avoid free variable "api" in methods
class McBoaty_v5 {
    constructor(api, app) {
        if (!window.__McBoaty_v5__) {
            window.__McBoaty_v5__ = Symbol("__McBoaty_v5__");
        }
        this.symbol = window.__McBoaty_v5__;
        this.api = api || window.api;
        this.app = app || window.app;
    }

    getState(node) {
        return node[this.symbol] || {};
    }

    setState(node, state) {
        node[this.symbol] = state;
        (this.app?.canvas || window.app?.canvas)?.setDirty(true);
    }

    addStatusTagHandler(nodeType) {
        if (nodeType[this.symbol]?.statusTagHandler) return;
        if (!nodeType[this.symbol]) nodeType[this.symbol] = {};
        nodeType[this.symbol] = { statusTagHandler: true };

        // FIX: use this.api instead of free "api"
        this.api.addEventListener("MaraScott/McBoaty_v5/update_status", ({ detail }) => {
            let { node, progress, text } = detail;
            const n = window.app.graph.getNodeById(+(node || window.app.runningNodeId));
            if (!n) return;
            const state = this.getState(n);
            state.status = Object.assign(state.status || {}, {
                progress: text ? progress : null,
                text: text || null,
            });
            this.setState(n, state);
        });

        const self = this;
        const onDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            const r = onDrawForeground?.apply?.(this, arguments);
            const state = self.getState(this);
            if (!state?.status?.text) return r;

            const { fgColor, bgColor, text, progress, progressColor } = { ...state.status };

            ctx.save();
            ctx.font = "12px sans-serif";
            const sz = ctx.measureText(text);
            ctx.fillStyle = bgColor || "dodgerblue";
            ctx.beginPath();
            ctx.roundRect(0, -LiteGraph.NODE_TITLE_HEIGHT - 20, sz.width + 12, 20, 5);
            ctx.fill();

            if (progress) {
                ctx.fillStyle = progressColor || "green";
                ctx.beginPath();
                ctx.roundRect(0, -LiteGraph.NODE_TITLE_HEIGHT - 20, (sz.width + 12) * progress, 20, 5);
                ctx.fill();
            }

            ctx.fillStyle = fgColor || "#fff";
            ctx.fillText(text, 6, -LiteGraph.NODE_TITLE_HEIGHT - 6);
            ctx.restore();
            return r;
        };
    }
}

// Extension factory
const MaraScottMcBoaty_v5 = (api) => {
    return {
        name: "ComfyUI.MaraScott.McBoaty_v5",
        async beforeRegisterNodeDef(nodeType, nodeData, app) {
            const mcBoaty_v5 = new McBoaty_v5(api, app);
            mcBoaty_v5.addStatusTagHandler(nodeType);

            if (nodeData.name === "MaraScottMcBoatyTilePrompter_v5") {
                const onExecuted = nodeType.prototype.onExecuted;
                nodeType.prototype.onExecuted = function (message) {
                    const r = onExecuted?.apply?.(this, arguments);

                    window.marascott.McBoaty_v5.inputs.prompts = message.prompts_in || [];
                    window.marascott.McBoaty_v5.message.prompts = message.prompts_out || [];
                    window.marascott.McBoaty_v5.inputs.denoises = message.denoises_in || [];
                    window.marascott.McBoaty_v5.message.denoises = message.denoises_out || [];
                    window.marascott.McBoaty_v5.inputs.tiles = window.marascott.McBoaty_v5.message.tiles = message.tiles || [];

                    MaraScottMcBoatyNodeWidget.refresh(this);

                    return r;
                };

                const onNodeCreated = nodeType.prototype.onNodeCreated;
                nodeType.prototype.onNodeCreated = function () {
                    const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                    this.widgets = this.widgets.filter((widget) => {
                        if (widget.name?.startsWith?.("tile ")) {
                            widget.onRemove?.();
                            return false;
                        }
                        return true;
                    });

                    MaraScottMcBoatyNodeWidget.init(this);
                    this.onResize?.(this.size);
                    return r;
                };
            } else {
                const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
                nodeType.prototype.getExtraMenuOptions = function (_, options) {
                    const r = getExtraMenuOptions?.apply?.(this, arguments);
                    let img;
                    if (this.imageIndex != null) {
                        img = this.imgs[this.imageIndex];
                    } else if (this.overIndex != null) {
                        img = this.imgs[this.overIndex];
                    }
                    if (img) {
                        let pos = options.findIndex((o) => o.content === "Save Image");
                        if (pos === -1) {
                            pos = 0;
                        } else {
                            pos++;
                        }
                        options.splice(pos, 0, {
                            content: "TilePrompt (McBoaty)",
                            callback: async () => {
                                let src = img.src;
                                src = src.replace(
                                    "/view?",
                                    `/MaraScott/McBoaty_v5/tile_prompt?node=${this.id}&clientId=${api.clientId}&`
                                );
                                const res = await (await fetch(src)).json();
                                alert(res);
                            },
                        });
                    }
                    return r;
                };
            }
        },
    };
};

export { MaraScottMcBoaty_v5 };
