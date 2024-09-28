
class extension {

    s = {} // ComfyUI Scripts
    
    constructor(name, scripts) {
        const ext = this // ComfyUI Scripts

        this.s = scripts
     
        return {
            name: "ComfyUI.MaraScott." + name,
            async beforeRegisterNodeDef(nodeType, nodeData, app) {
                if (nodeData.name === "MaraScott"+name) {
                    const onExecuted = nodeType.prototype.onExecuted;
        
                    nodeType.prototype.onExecuted = function (message) {
                        onExecuted?.apply(this, arguments);
        
                        if (this.widgets) {
                            for (let i = 1; i < this.widgets.length; i++) {
                                this.widgets[i].onRemove?.();
                            }
                        }

                        ext.setWidget_text(this, message["text"])
        
                    };
                }
            },
        }

    }

    setWidget_text(node, text) {
        // Check if the "text" widget already exists.
        let widget = node.widgets && node.widgets.find(w => w.name === "text");
        if (!widget) {
            widget = this.s.ComfyWidgets["STRING"](node, "text", ["STRING", { multiline: true }], this.s.app).widget;
            widget.inputEl.readOnly = true;
            widget.inputEl.style.opacity = 0.6;
        }
        console.log(window.marascott.AnyBus_v2)
        // use window.marascott.AnyBus_v2.start for indexes of dashboard nodes
        text = JSON.stringify(window.marascott.AnyBus_v2.nodes[8].inputs, this.getCircularReplacer(), 2)
        widget.value = text;
    }

    // Use custom replacer to handle circular references
    getCircularReplacer() {
    const seen = new WeakSet();
    return (key, value) => {
        if (typeof value === "object" && value !== null) {
            if (seen.has(value)) {
                return; // Omit circular references
            }
            seen.add(value);
        }
        return value;
    };
}    

};

export { extension }