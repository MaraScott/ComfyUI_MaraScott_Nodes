import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

app.registerExtension({
    name: "ComfyUI.MaraScott.DisplayInfo_v2",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "MaraScottDisplayInfo_v2") {
            const onExecuted = nodeType.prototype.onExecuted;

            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);

                if (this.widgets) {
					for (let i = 1; i < this.widgets.length; i++) {
						this.widgets[i].onRemove?.();
					}
				}

                // Check if the "text" widget already exists.
                let textWidget = this.widgets && this.widgets.find(w => w.name === "text");
                if (!textWidget) {
                    textWidget = ComfyWidgets["STRING"](this, "text", ["STRING", { multiline: true }], app).widget;
                    textWidget.inputEl.readOnly = true;
                    textWidget.inputEl.style.opacity = 0.6;
                }
                textWidget.value = message["text"].join('');
            };
        }
    },
});