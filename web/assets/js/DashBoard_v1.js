import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";
import { extension } from './nodes/DashBoard_v1/extension.js'
app.registerExtension(new extension("DashBoard_v1", {
    "app": app,
    "ComfyWidgets": ComfyWidgets,
}));
