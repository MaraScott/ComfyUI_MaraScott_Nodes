import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { MaraScottMcBoaty_v6 } from "./nodes/McBoaty_v6.js";

app.registerExtension(MaraScottMcBoaty_v6(api))