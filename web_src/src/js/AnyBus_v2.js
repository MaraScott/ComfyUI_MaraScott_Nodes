import { app } from "../../scripts/app.js";
import { MaraScottAnyBusNodeExtension, MaraScottAnyBusNodeSidebarTab } from "./nodes/AnyBus_v2.js";

app.registerExtension(MaraScottAnyBusNodeExtension());
app.extensionManager.registerSidebarTab(MaraScottAnyBusNodeSidebarTab());