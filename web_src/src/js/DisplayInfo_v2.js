import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";
import { MaraScottDisplayInfo_v2 } from "./nodes/DisplayInfo_v2.js";

app.registerExtension(MaraScottDisplayInfo_v2(ComfyWidgets));