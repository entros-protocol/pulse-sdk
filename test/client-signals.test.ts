import { afterEach, describe, expect, it, vi } from "vitest";
import {
  PointerObservationTracker,
  buildInputCapabilitySignals,
  collectClientSignals,
  hasVariablePointerPressure,
} from "../src/client-signals/automation";

/**
 * Client-signals collector. These tests pin the privacy contract
 * (clean for real/privacy browsers, no false positives, never throws) and the
 * detection behaviour (each framework artifact maps to a stable label).
 */

// vitest runs in the node environment here, so `window`/`navigator`/`document`
// are undefined by default — that is the React Native / Node / SSR path. The
// browser cases stub minimal globals and tear them down afterwards.
function stubBrowser(opts: {
  webdriver?: unknown;
  product?: string;
  windowExtras?: Record<string, unknown>;
  windowOverride?: unknown;
  documentOverride?: "absent" | Record<string, unknown>;
  documentExtras?: Record<string, unknown>;
  documentElementAttrs?: Record<string, string>;
  webdriverThrows?: boolean;
}) {
  if (opts.windowOverride !== undefined) {
    vi.stubGlobal("window", opts.windowOverride);
  } else {
    vi.stubGlobal("window", { ...(opts.windowExtras ?? {}) });
  }

  const nav: Record<string, unknown> = {};
  if (opts.product !== undefined) nav.product = opts.product;
  if (opts.webdriverThrows) {
    Object.defineProperty(nav, "webdriver", {
      get() {
        throw new Error("blocked");
      },
    });
  } else {
    nav.webdriver = opts.webdriver;
  }
  vi.stubGlobal("navigator", nav);

  if (opts.documentOverride === "absent") {
    vi.stubGlobal("document", undefined);
  } else {
    const attrs = opts.documentElementAttrs ?? {};
    const doc: Record<string, unknown> = {
      documentElement: {
        getAttribute: (name: string) => (name in attrs ? attrs[name] : null),
      },
      ...(opts.documentOverride ?? {}),
      ...(opts.documentExtras ?? {}),
    };
    vi.stubGlobal("document", doc);
  }
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("collectClientSignals", () => {
  it("returns a clean non-browser envelope when window/navigator are absent", () => {
    const sig = collectClientSignals();
    expect(sig.env).toBe("non-browser");
    expect(sig.automation.webdriver).toBe(false);
    expect(sig.automation.tells).toEqual([]);
    expect(sig.v).toBe(1);
  });

  it("classifies React Native as non-browser even when window/navigator exist", () => {
    stubBrowser({ product: "ReactNative", windowExtras: { __puppeteer_evaluation_script__: {} } });
    const sig = collectClientSignals();
    expect(sig.env).toBe("non-browser");
    expect(sig.automation.webdriver).toBe(false);
    expect(sig.automation.tells).toEqual([]);
  });

  it("treats an ordinary browser (no automation) as clean", () => {
    stubBrowser({ webdriver: false });
    const sig = collectClientSignals();
    expect(sig.env).toBe("browser");
    expect(sig.automation.webdriver).toBe(false);
    expect(sig.automation.tells).toEqual([]);
  });

  it("treats a privacy-hardened browser (webdriver undefined) as clean", () => {
    // Tor Browser / RFP exposes navigator with no webdriver set; it must not
    // be flagged.
    stubBrowser({ webdriver: undefined });
    const sig = collectClientSignals();
    expect(sig.automation.webdriver).toBe(false);
    expect(sig.automation.tells).toEqual([]);
  });

  it("flags navigator.webdriver === true", () => {
    stubBrowser({ webdriver: true });
    expect(collectClientSignals().automation.webdriver).toBe(true);
  });

  it("detects Puppeteer", () => {
    stubBrowser({ windowExtras: { __puppeteer_evaluation_script__: {} } });
    expect(collectClientSignals().automation.tells).toContain("puppeteer");
  });

  it("detects Playwright", () => {
    stubBrowser({ windowExtras: { __pwInitScripts: {} } });
    expect(collectClientSignals().automation.tells).toContain("playwright");
  });

  it("detects PhantomJS via callPhantom", () => {
    stubBrowser({ windowExtras: { callPhantom: () => {} } });
    expect(collectClientSignals().automation.tells).toContain("phantom");
  });

  it("detects Selenium via a cdc_-prefixed window key", () => {
    stubBrowser({ windowExtras: { $cdc_asdjflasutopfhvcZLmcfl_: [] } });
    expect(collectClientSignals().automation.tells).toContain("selenium");
  });

  it("does NOT flag a benign global that merely contains 'cdc_' (no false positive)", () => {
    // A real page/library global like `webrtcdc_state` must not be read as a
    // ChromeDriver artifact — the regex is prefix-anchored on purpose.
    stubBrowser({ windowExtras: { webrtcdc_state: {}, my_cdc_helper: {} } });
    expect(collectClientSignals().automation.tells).not.toContain("selenium");
  });

  it("detects Selenium via a webdriver document-element attribute", () => {
    stubBrowser({ documentElementAttrs: { webdriver: "true" } });
    expect(collectClientSignals().automation.tells).toContain("selenium");
  });

  it("deduplicates a framework reported by multiple keys", () => {
    stubBrowser({
      windowExtras: { __webdriver_evaluate: {}, __selenium_evaluate: {} },
    });
    const tells = collectClientSignals().automation.tells;
    expect(tells.filter((t) => t === "selenium")).toHaveLength(1);
  });

  it("never throws when a probed property getter throws", () => {
    stubBrowser({ webdriverThrows: true });
    expect(() => collectClientSignals()).not.toThrow();
    expect(collectClientSignals().automation.webdriver).toBe(false);
  });

  it("never throws when window is a hostile Proxy", () => {
    const hostile = new Proxy(
      {},
      {
        has() {
          throw new Error("trap");
        },
        get() {
          throw new Error("trap");
        },
        ownKeys() {
          throw new Error("trap");
        },
      },
    );
    stubBrowser({ webdriver: false, windowOverride: hostile });
    expect(() => collectClientSignals()).not.toThrow();
  });

  it("never throws when document is absent but window is present", () => {
    stubBrowser({ webdriver: false, documentOverride: "absent" });
    expect(() => collectClientSignals()).not.toThrow();
    expect(collectClientSignals().env).toBe("browser");
  });

  it("attaches explicit input capability booleans without dropping non-browser env", () => {
    const input = buildInputCapabilitySignals({
      touchEventsObserved: false,
      pointerPressureObserved: false,
      motionEventsObserved: false,
    });
    const nonBrowser = collectClientSignals(input);
    expect(nonBrowser.env).toBe("non-browser");
    expect(nonBrowser.input).toEqual(input);

    stubBrowser({ webdriver: false });
    const browser = collectClientSignals(
      buildInputCapabilitySignals({
        touchEventsObserved: true,
        pointerPressureObserved: true,
        motionEventsObserved: true,
      }),
    );
    expect(browser.env).toBe("browser");
    expect(browser.input).toEqual({
      touch_events_observed: true,
      pointer_pressure_observed: true,
      motion_events_observed: true,
    });
  });
});

describe("PointerObservationTracker", () => {
  it("reports desktop mouse input with fixed 0.5 pressure as non-touch and unpressured", () => {
    const tracker = new PointerObservationTracker();
    tracker.observe("mouse", 0.5);
    tracker.observe("mouse", 0.5);
    expect(tracker.snapshot()).toEqual({
      touchEventsObserved: false,
      pointerPressureObserved: false,
    });
  });

  it("reports mobile fixed-pressure touch as touch without variable pressure", () => {
    const tracker = new PointerObservationTracker();
    tracker.observe("touch", 1);
    tracker.observe("touch", 1);
    expect(tracker.snapshot()).toEqual({
      touchEventsObserved: true,
      pointerPressureObserved: false,
    });
    expect(hasVariablePointerPressure([{ pressure: 1 }, { pressure: 1 }])).toBe(false);
  });

  it("detects variable pointer pressure from fractional or changing readings", () => {
    const fractional = new PointerObservationTracker();
    fractional.observe("touch", 0.34);
    expect(fractional.snapshot()).toEqual({
      touchEventsObserved: true,
      pointerPressureObserved: true,
    });

    const transition = new PointerObservationTracker();
    transition.observe("touch", 0.5);
    transition.observe("touch", 1);
    expect(transition.snapshot().pointerPressureObserved).toBe(true);
    expect(hasVariablePointerPressure([{ pressure: 0.5 }, { pressure: 0.72 }])).toBe(true);
  });
});
