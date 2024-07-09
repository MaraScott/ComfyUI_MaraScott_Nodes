import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";
import { api } from "../../scripts/api.js";

if (!window.marascott) {
	window.marascott = {}
}
if (!window.marascott.McBoaty_v5) {
	window.marascott.McBoaty_v5 = {
		init: false,
		clean: false,
        message: null,
	}
}


function imageDataToUrl(data) {
    return api.apiURL(`/view?filename=${encodeURIComponent(data.filename)}&type=${data.type}&subfolder=${data.subfolder}${app.getPreviewFormatParam()}${app.getRandParam()}`);
}

export const McBoatyWidgets = {

    WRAPPER: (key, index, prompt, tile, node) => {

        const inputEl = document.createElement("div");
        inputEl.className = "comfy-wrapper-mcboaty";
        
        const wrapper = document.createElement("div");
        wrapper.style.height = "100%";
        wrapper.style.display = "flex";
        wrapper.style.alignItems = "center";
        wrapper.style.gap = "10px";

        const text = document.createElement("p");
        text.textContent = String(index + 1).padStart(2, '0');
        
        const textarea = document.createElement("textarea");
        textarea.style.opacity = 0.6;
        textarea.style.flexGrow = 1;
        textarea.style.height = "100%";
        textarea.className = "comfy-multiline-input";
        textarea.value = prompt || "";
        textarea.placeholder = prompt || "";
        textarea.placeholder = "tile "+text.textContent;
        textarea.dataId = "tile "+index;
        textarea.dataNodeId = node.id;
        textarea.addEventListener('focusout', async function() {
            const res = await (await fetch(`/MaraScott/McBoaty/v4/set_prompt?index=${index}&prompt=${this.value}&node=${this.dataNodeId}&clientId=${api.clientId}`)).json();
            // You can add more functionality here that should run when the input loses focus
            
        });
        
        var img = document.createElement('img');
        img.src = imageDataToUrl(tile);  // Replace with the actual image path
        img.alt = prompt;
        img.style.height = "100%";
        img.style.maxWidth = "128px";
        img.style.maxHeight = "128px";
        img.style.flexShrink = "0";
        wrapper.appendChild(text);
        wrapper.appendChild(img);
        wrapper.appendChild(textarea);        
        inputEl.appendChild(wrapper);        
        
        const widget = node.addDOMWidget(name, "customtext", inputEl, {
            getValue() {
                return inputEl.value;
            },
            setValue(v) {
                inputEl.value = v;
            },
        });
        widget.inputEl = inputEl;
        widget.value = prompt;
        
        textarea.addEventListener("input", () => {
            widget.callback?.(widget.value);
        });
    
        return widget;
    }
}

class MaraScottMcBoatyNodePrompter {

	static clean(node) {

		window.marascott.McBoaty_v5.clean = false

        const cleanedLabel = " ... cleaned"
		node.title = node.title + cleanedLabel
		setTimeout(() => {
			// Remove " (cleaned)" from the title
			node.title = node.title.replace(cleanedLabel, "");
		}, 500);

	}

}

class MaraScottMcBoatyNodeWidget {

	static INDEX = {
		name: "Filter by Indexes",
		default: "",
	}
	static PROMPT = {
		name: "Prompt (all filtered)",
		default: "",
	}
	static CLEAN = {
		name: 'Reset',
		default: false,
	}

	static init(node) {

		this.setIndexInput(node)
		this.setPromptInput(node)
		this.setCleanSwitch(node)

	}

	static getByName(node, name) {
		return node.widgets?.find((w) => w.name === name);
	}

	static setValue(node, name, value) {

		const nodeWidget = this.getByName(node, name);
		nodeWidget.value = value
		node.setProperty(name, nodeWidget.value ?? node.properties[name])
		node.setDirtyCanvas(true)

	}

	static setIndexInput(node) {

		const nodeWidget = this.getByName(node, this.INDEX.name);

		if (nodeWidget == undefined) {
			node.addWidget(
				"text",
				this.INDEX.name,
				node.properties[this.INDEX.name] ?? this.INDEX.default,
				(value, LGraphCanvas, Node, Coordinate, PointerEvent) => {
					this.setValue(node, this.INDEX.name, value);
                    this.setValue(node, this.PROMPT.name, this.PROMPT.default);
				},
				{}
			)
			this.setValue(node, this.INDEX.name, this.INDEX.default)
		}

	}

    static setPromptInput(node) {

		const nodeWidget = this.getByName(node, this.PROMPT.name);

		if (nodeWidget == undefined) {
			node.addWidget(
				"text",
				this.PROMPT.name,
				node.properties[this.PROMPT.name] ?? this.PROMPT.default,
				(value, LGraphCanvas, Node, Coordinate, PointerEvent) => {
					this.setValue(node, this.PROMPT.name, value);
                    
                    const input_list = node.properties[this.INDEX.name] ?? this.INDEX.default;
                    if(input_list != "") {
                        node.widgets = node.widgets.filter(widget => {

                            const focusOutEvent = new Event('focusout');
                            const index_filtered = input_list.split(",").map(num => Number(num) - 1);
                            if (widget.type == "customtext") {
                                const textarea = widget.inputEl.querySelector('[placeholder^="tile "]');
                                if ( textarea != null) {
                                    const dataId = textarea.getAttribute('placeholder');
                                    const indexValue = parseInt(dataId.replace('tile ', ''), 10);
                                    const indexFound = index_filtered.indexOf(indexValue - 1);
                                    if(indexFound > -1) {
                                        window.marascott.McBoaty_v5.message.prompts[indexValue] = value;
                                        textarea.value = value;
                                        textarea.dispatchEvent(focusOutEvent);
                                        return false;
                                    }
                                }
                            }
                            return true;

                        });
                    }
                    node.widgets = node.widgets.filter(widget => {
                        if (widget.type == "customtext") {
                            widget.onRemove?.();
                            return false;
                        }
                        return true;
                    });
    
                    node.widgets = node.widgets.filter(widget => {
                        if(widget.name === MaraScottMcBoatyNodeWidget.CLEAN.name) {
                            widget.onRemove?.();
                            return false;
                        }
                        return true;
                    });
    
                    MaraScottMcBoatyNodeWidget.setIndexInput(node)
                    MaraScottMcBoatyNodeWidget.setPromptInput(node)
            
                    node.onResize?.(node.size);
                    node.graph.setDirtyCanvas(true, true);
    
                    MaraScottMcBoatyNodeWidget.setPrompterInputs(node)
    
                    MaraScottMcBoatyNodeWidget.setCleanSwitch(node)
    
                    node.onResize?.(node.size);
                    node.graph.setDirtyCanvas(true, true);    
				},
				{}
			)
			this.setValue(node, this.PROMPT.name, this.PROMPT.default)
		}

	}

	static setCleanSwitch(node) {

		const nodeWidget = this.getByName(node, this.CLEAN.name);
		if (nodeWidget == undefined) {
			node.addWidget(
				"toggle",
				this.CLEAN.name,
				this.CLEAN.default,
				(value, LGraphCanvas, Node, Coordinate, PointerEvent) => {
                    MaraScottMcBoatyNodePrompter.clean(node)
					this.setValue(node, this.CLEAN.name, this.CLEAN.default)
				},
				{}
			)
			this.setValue(node, this.CLEAN.name, this.CLEAN.default)
		}

	}

    static setPrompterInputs(node) {

        let index_list = (node.properties[this.INDEX.name] ?? this.INDEX.default).trim().split(",");
		index_list = (index_list.length === 1 && index_list[0] === '') ? [] : index_list;
        const index_filtered = index_list.map(num => Number(num) - 1);

        for (const [index, prompt] of window.marascott.McBoaty_v5.message.prompts.entries()) {
            if (index_list.length == 0 || index_filtered.indexOf(index) > -1) {
                const w = McBoatyWidgets.WRAPPER("tile "+index, index, prompt, window.marascott.McBoaty_v5.message.tiles[index], node);
            }
        }


    }

}

class McBoaty_v5 {
	constructor() {
		if (!window.__McBoaty_v5__) {
			window.__McBoaty_v5__ = Symbol("__McBoaty_v5__");
		}
		this.symbol = window.__McBoaty_v5__;
	}

	getState(node) {
		return node[this.symbol] || {};
	}

	setState(node, state) {
		node[this.symbol] = state;
		app.canvas.setDirty(true);
	}

	addStatusTagHandler(nodeType) {
		if (nodeType[this.symbol]?.statusTagHandler) {
			return;
		}
		if (!nodeType[this.symbol]) {
			nodeType[this.symbol] = {};
		}
		nodeType[this.symbol] = {
			statusTagHandler: true,
		};

		api.addEventListener("MaraScott/McBoaty_v5/update_status", ({ detail }) => {
			let { node, progress, text } = detail;
			const n = app.graph.getNodeById(+(node || app.runningNodeId));
			if (!n) return;
			const state = this.getState(n);
			state.status = Object.assign(state.status || {}, { progress: text ? progress : null, text: text || null });
			this.setState(n, state);
		});

		const self = this;
		const onDrawForeground = nodeType.prototype.onDrawForeground;
		nodeType.prototype.onDrawForeground = function (ctx) {
			const r = onDrawForeground?.apply?.(this, arguments);
			const state = self.getState(this);
			if (!state?.status?.text) {
				return r;
			}

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

const mcBoaty_v5 = new McBoaty_v5();

app.registerExtension({
	name: "ComfyUI.MaraScott.McBoaty_v5",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {

        mcBoaty_v5.addStatusTagHandler(nodeType);

		if (nodeData.name === "McBoaty_TilePrompter_v5") {

            const onExecuted = nodeType.prototype.onExecuted;
			nodeType.prototype.onExecuted = function (message) {
                window.marascott.McBoaty_v5.message = message;
				const r = onExecuted?.apply?.(this, arguments);

                this.widgets = this.widgets.filter(widget => {
                    if (widget.type == "customtext") {
                        widget.onRemove?.();
                        return false;
                    }
                    return true;
                });

                this.widgets = this.widgets.filter(widget => {
                    if(widget.name === MaraScottMcBoatyNodeWidget.CLEAN.name) {
                        widget.onRemove?.();
                        return false;
                    }
                    return true;
                });

                MaraScottMcBoatyNodeWidget.setIndexInput(this)
                MaraScottMcBoatyNodeWidget.setPromptInput(this)
        
				this.onResize?.(this.size);
                this.graph.setDirtyCanvas(true, true);

                MaraScottMcBoatyNodeWidget.setPrompterInputs(this)

                MaraScottMcBoatyNodeWidget.setCleanSwitch(this)

				this.onResize?.(this.size);
                this.graph.setDirtyCanvas(true, true);

				return r;
			};
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                this.widgets = this.widgets.filter(widget => {
                    if (widget.name.startsWith("tile ")) {
                        widget.onRemove?.();
                        return false;
                    }
                    return true;
                });

                MaraScottMcBoatyNodeWidget.init(this);
                this.onResize?.(this.size);

                return r;
            }        
    
		} else {
			const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
			nodeType.prototype.getExtraMenuOptions = function (_, options) {
				const r = getExtraMenuOptions?.apply?.(this, arguments);
				let img;
				if (this.imageIndex != null) {
					// An image is selected so select that
					img = this.imgs[this.imageIndex];
				} else if (this.overIndex != null) {
					// No image is selected but one is hovered
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
							src = src.replace("/view?", `/MaraScott/McBoaty_v5/tile_prompt?node=${this.id}&clientId=${api.clientId}&`);
							const res = await (await fetch(src)).json();
							alert(res);
						},
					});
				}

				return r;
			};
		}
	},
});
