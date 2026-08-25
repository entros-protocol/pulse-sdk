import { runTouchScaleMeasurement } from "../test/support/touch-scale-measurement";

process.stdout.write(`${JSON.stringify(runTouchScaleMeasurement(), null, 2)}\n`);
