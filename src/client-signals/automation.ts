/**
 * Client signals — Layer A1, observe-only.
 *
 * Produces the `client_signals` envelope attached to the `/validate-features`
 * request. Today it carries one signal group, `automation` — whether the
 * verification session is being driven by a browser AUTOMATION framework
 * (Selenium/WebDriver, Puppeteer, Playwright, PhantomJS, Nightmare, Cypress, or
 * a live Chrome DevTools-Protocol harness) — detected from the artifacts those
 * tools inject into the page. This is the signal class that separates a
 * scripted/headless bot from a real person in a real browser.
 *
 * The envelope is intentionally namespaced so later signal groups (interaction
 * realism, capture realism) slot in as SIBLINGS of `automation` without a
 * breaking wire change or a flat junk-drawer.
 *
 * PRIVACY CONTRACT — load-bearing, do not weaken:
 *   - Detects AUTOMATION, not the USER. Every signal here is a property of the
 *     automation harness driving the page, never of the human or their device.
 *   - NO fingerprinting. No canvas/WebGL/audio fingerprint, no font or plugin
 *     enumeration, no User-Agent parsing, no device enumeration, no
 *     locale/timezone/screen probing. A privacy-hardened browser — Tor Browser,
 *     Firefox resist-fingerprinting, a JS-locked profile — trips NONE of these
 *     checks. It is a real engine driven by a real person, and is treated as
 *     such.
 *   - No user data is read, stored, or transmitted beyond the boolean and the
 *     closed-vocabulary framework labels below. Labels are names only, never
 *     values; enumerated property names are matched and discarded, never
 *     emitted.
 *   - Open-source on purpose: an auditor can read this file and confirm we do
 *     not surveil. The checks are textbook (stealth kits already patch them);
 *     the value is server-side scoring, not secrecy here.
 *
 * Runtime-safe: returns a clean `non-browser` envelope under React Native,
 * Node, SSR, or any environment without `window`/`navigator`, and never throws.
 * Mobile (entros-mobile, React Native) has no browser-automation surface — its
 * anti-automation story is native device attestation, a separate track.
 */

/** Bump when the envelope shape changes so the server can interpret it safely. */
const SCHEMA_VERSION = 1;

/** Automation-framework detection group. Names only, never values. */
export interface AutomationSignals {
  /**
   * `navigator.webdriver === true` — the W3C WebDriver standard automation
   * flag, set by Selenium/Puppeteer/Playwright/CDP. False or undefined in every
   * ordinary human browser, including privacy-hardened ones.
   */
  webdriver: boolean;
  /**
   * Framework labels for automation artifacts found in the page (e.g.
   * "puppeteer", "playwright", "selenium"). Empty for a real human session.
   * Deduplicated; a closed vocabulary, never raw property values.
   */
  tells: string[];
}

/** The `client_signals` envelope. Namespaced for future signal groups. */
export interface ClientSignals {
  /** Envelope schema version. */
  v: number;
  /**
   * "browser" when a real browser runtime is present; "non-browser" for React
   * Native, Node, or SSR, where no browser automation tells exist.
   */
  env: "browser" | "non-browser";
  /** Automation-framework detection group. */
  automation: AutomationSignals;
}

/**
 * Window globals injected by automation frameworks. Presence of the KEY is the
 * tell; the value is never read. Multiple keys can map to one framework label —
 * the result deduplicates.
 */
const WINDOW_TELLS: ReadonlyArray<readonly [string, string]> = [
  ["__puppeteer_evaluation_script__", "puppeteer"],
  ["__playwright", "playwright"],
  ["__playwright__binding__", "playwright"],
  ["__pwInitScripts", "playwright"],
  ["__nightmare", "nightmare"],
  ["_phantom", "phantom"],
  ["callPhantom", "phantom"],
  ["domAutomation", "chrome-automation"],
  ["domAutomationController", "chrome-automation"],
  ["Cypress", "cypress"],
  ["__webdriver_evaluate", "selenium"],
  ["__selenium_evaluate", "selenium"],
  ["__webdriver_script_function", "selenium"],
  ["__webdriver_script_func", "selenium"],
  ["__webdriver_script_fn", "selenium"],
  ["__fxdriver_evaluate", "selenium"],
  ["__driver_evaluate", "selenium"],
  ["__webdriver_unwrapped", "selenium"],
  ["__selenium_unwrapped", "selenium"],
  ["__fxdriver_unwrapped", "selenium"],
  ["__driver_unwrapped", "selenium"],
  ["_Selenium_IDE_Recorder", "selenium"],
  ["__$webdriverAsyncExecutor", "selenium"],
  ["__lastWatirAlert", "watir"],
  ["__lastWatirConfirm", "watir"],
  ["__lastWatirPrompt", "watir"],
];

/**
 * ChromeDriver/Selenium inject `cdc_`-PREFIXED keys (e.g. `$cdc_asdjfla...`)
 * onto `window`/`document`. Anchored to the prefix on purpose — an unanchored
 * `/cdc_/` substring would flag benign globals (e.g. `webrtcdc_state`) and
 * misclassify a real user, breaking the privacy contract.
 */
const CDC_PATTERN = /^[$]?cdc_/;

/** A clean envelope for runtimes with no browser-automation surface. */
function nonBrowserSignals(): ClientSignals {
  return { v: SCHEMA_VERSION, env: "non-browser", automation: { webdriver: false, tells: [] } };
}

/**
 * Collect the client-signals envelope for the current runtime. Pure,
 * synchronous, cheap (one-time at submit), and exception-safe — every probe is
 * individually guarded so a hostile or unusual environment can never break
 * verification submission.
 */
export function collectClientSignals(): ClientSignals {
  // React Native exposes `navigator` (and often a polyfilled `window`) but has
  // no browser-automation surface — classify it as non-browser so it never
  // pollutes the browser calibration baseline. (`navigator.product` is a fixed
  // string, not a fingerprint.)
  if (
    typeof navigator !== "undefined" &&
    (navigator as { product?: string }).product === "ReactNative"
  ) {
    return nonBrowserSignals();
  }

  // Node / SSR / any runtime without a DOM.
  if (typeof window === "undefined" || typeof navigator === "undefined") {
    return nonBrowserSignals();
  }

  const found = new Set<string>();
  let webdriver = false;

  try {
    webdriver = navigator.webdriver === true;
  } catch {
    /* navigator.webdriver access threw — treat as not present */
  }

  const w = window as unknown as Record<string, unknown>;
  for (const [key, label] of WINDOW_TELLS) {
    try {
      if (key in w && w[key] != null) {
        found.add(label);
      }
    } catch {
      /* property probe threw (e.g. a hostile getter/Proxy) — skip this key */
    }
  }

  // Scan own-property names for the `cdc_` prefix. `k` is matched and discarded
  // here — it must NEVER be emitted (it could name a page-defined global).
  try {
    const scan = (obj: object) => {
      for (const k of Object.getOwnPropertyNames(obj)) {
        if (CDC_PATTERN.test(k)) {
          found.add("selenium");
          return;
        }
      }
    };
    scan(w);
    if (typeof document !== "undefined" && document) {
      scan(document as unknown as object);
    }
  } catch {
    /* key enumeration threw — skip */
  }

  // Automation can mark the document element with WebDriver attributes.
  try {
    const de = typeof document !== "undefined" ? document.documentElement : null;
    if (de) {
      for (const attr of ["webdriver", "selenium", "driver"]) {
        if (de.getAttribute(attr) != null) {
          found.add("selenium");
          break;
        }
      }
    }
  } catch {
    /* attribute probe threw — skip */
  }

  return {
    v: SCHEMA_VERSION,
    env: "browser",
    automation: { webdriver, tells: Array.from(found) },
  };
}
