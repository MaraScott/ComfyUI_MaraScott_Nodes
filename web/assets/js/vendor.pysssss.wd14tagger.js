import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";
import { api } from "../../scripts/api.js";

class MS_Pysssss {
	constructor() {
		if (!window.__MS_pysssss__) {
			window.__MS_pysssss__ = Symbol("__MS_pysssss__");
		}
		this.symbol = window.__MS_pysssss__;
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

		api.addEventListener("MaraScott/vendor/ComfyUI_WD14_Tagger/pysssss/update_status", ({ detail }) => {
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

const MS_pysssss = new MS_Pysssss();

app.registerExtension({
	name: "ComfyUI.MaraScott.vendor.pysssss.Wd14Tagger",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		MS_pysssss.addStatusTagHandler(nodeType);

		if (nodeData.name === "MaraScottUpscalerRefinerNode_v3") {
			const onExecuted = nodeType.prototype.onExecuted;
			nodeType.prototype.onExecuted = function (message) {
				const r = onExecuted?.apply?.(this, arguments);

				const pos = this.widgets.findIndex((w) => w.name === "tags");
				if (pos !== -1) {
					for (let i = pos; i < this.widgets.length; i++) {
						this.widgets[i].onRemove?.();
					}
					this.widgets.length = pos;
				}

				for (const list of message.tags) {
					const w = ComfyWidgets["STRING"](this, "tags", ["STRING", { multiline: true }], app).widget;
					w.inputEl.readOnly = true;
					w.inputEl.style.opacity = 0.6;
					w.value = list;
				}

				this.onResize?.(this.size);

				return r;
			};
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
						content: "WD14 Tagger (via MaraScott)",
						callback: async () => {
							let src = img.src;
							src = src.replace("/view?", `/MaraScott/vendor/ComfyUI_WD14_Tagger/pysssss/wd14tagger/tag?node=${this.id}&clientId=${api.clientId}&`);
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
