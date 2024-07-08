import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";
import { api } from "../../scripts/api.js";

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
        textarea.placeholder = "tile "+index;
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
				const r = onExecuted?.apply?.(this, arguments);

				const pos = this.widgets.findIndex((w) => w.name === "prompts");
				if (pos !== -1) {
					for (let i = pos; i < this.widgets.length; i++) {
                        if(this.widgets[i].name === "prompts") this.widgets[i].onRemove?.();
					}
					this.widgets.length = pos;
				}

                this.widgets = this.widgets.filter(widget => {
                    if (widget.name.startsWith("tile ")) {
                        widget.onRemove?.();
                        return false;
                    }
                    return true;
                });
				this.onResize?.(this.size);
                this.graph.setDirtyCanvas(true, true);

				for (const [index, prompt] of message.prompts.entries()) {
                    const w = McBoatyWidgets.WRAPPER("tile "+index, index, prompt, message.tiles[index], this)
				}

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
