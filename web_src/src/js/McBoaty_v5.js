import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { MaraScottMcBoaty_v5 } from "./nodes/McBoaty_v5.js";

app.registerExtension(MaraScottMcBoaty_v5(api))