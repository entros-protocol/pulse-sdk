import { describe, it, expect, afterEach, vi } from "vitest";
import { postJson, TransportError } from "../src/transport/post-json";
import {
  RETRYABLE_REASONS,
  COOLDOWN_REASONS,
  isVerificationReason,
  reasonDisposition,
} from "../src/reasons";

/**
 * Minimal stand-in for the browser's XMLHttpRequest.
 *
 * Node has no XHR, so `postJson` would otherwise always take the `fetch`
 * fallback and the path that actually ships would go untested. The fake
 * exposes hooks to drive upload progress, completion and failure by hand,
 * which is the only way to assert the stall-versus-slow distinction, the
 * whole reason this transport exists.
 */
class FakeXhr {
  static last: FakeXhr | undefined;
  /** Drives the synchronous-throw paths that leaked timers. */
  static throwOnSend: Error | undefined;
  static throwOnOpen: Error | undefined;

  status = 0;
  responseText = "";
  aborted = false;
  sentBody: string | null = null;
  readonly requestHeaders: Record<string, string> = {};
  private responseHeaderBlock = "";

  upload: {
    onprogress: ((event: ProgressEvent) => void) | null;
    onload: (() => void) | null;
  } = { onprogress: null, onload: null };

  onload: (() => void) | null = null;
  onerror: (() => void) | null = null;
  onabort: (() => void) | null = null;

  open(_method: string, _url: string, _async: boolean): void {
    if (FakeXhr.throwOnOpen) throw FakeXhr.throwOnOpen;
  }

  setRequestHeader(name: string, value: string): void {
    this.requestHeaders[name] = value;
  }

  send(body: string): void {
    FakeXhr.last = this;
    if (FakeXhr.throwOnSend) throw FakeXhr.throwOnSend;
    this.sentBody = body;
  }

  abort(): void {
    this.aborted = true;
  }

  getAllResponseHeaders(): string {
    return this.responseHeaderBlock;
  }

  // -- test drivers --

  emitProgress(loaded: number, total: number): void {
    this.upload.onprogress?.({
      loaded,
      total,
      lengthComputable: total > 0,
    } as ProgressEvent);
  }

  finishUpload(): void {
    this.upload.onload?.();
  }

  respond(status: number, body: unknown, headers: Record<string, string> = {}): void {
    this.status = status;
    this.responseText = typeof body === "string" ? body : JSON.stringify(body);
    this.responseHeaderBlock = Object.entries(headers)
      .map(([name, value]) => `${name}: ${value}`)
      .join("\r\n");
    this.onload?.();
  }

  failNetwork(): void {
    this.onerror?.();
  }
}

function installFakeXhr(): void {
  FakeXhr.last = undefined;
  (globalThis as Record<string, unknown>).XMLHttpRequest =
    FakeXhr as unknown as typeof XMLHttpRequest;
}

function uninstallFakeXhr(): void {
  delete (globalThis as Record<string, unknown>).XMLHttpRequest;
  FakeXhr.last = undefined;
}

/** Let the microtask queue drain so `send()` has run and `last` is set. */
async function settled(): Promise<void> {
  await Promise.resolve();
  await Promise.resolve();
}

afterEach(() => {
  uninstallFakeXhr();
  vi.useRealTimers();
});

describe("postJson: the slow-versus-stalled distinction", () => {
  it("does not abort an upload that is still moving, however long it takes", async () => {
    vi.useFakeTimers();
    installFakeXhr();

    const pending = postJson("https://example.test/v", { a: 1 }, { stallMs: 1000 });
    await settled();
    const xhr = FakeXhr.last!;

    // Ten seconds of upload, ten times the stall budget, but a progress
    // event every 900ms. This is the mobile case that used to fail: healthy,
    // slow, and previously killed by a fixed deadline.
    for (let sent = 0; sent < 10; sent++) {
      vi.advanceTimersByTime(900);
      xhr.emitProgress(sent * 1000, 10000);
    }
    expect(xhr.aborted).toBe(false);

    xhr.finishUpload();
    xhr.respond(200, { valid: true });
    const response = await pending;
    expect(response.status).toBe(200);
  });

  it("aborts an upload that goes quiet for longer than the stall budget", async () => {
    vi.useFakeTimers();
    installFakeXhr();

    const pending = postJson("https://example.test/v", { a: 1 }, { stallMs: 1000 });
    await settled();
    const xhr = FakeXhr.last!;

    xhr.emitProgress(500, 10000);
    vi.advanceTimersByTime(1001);

    await expect(pending).rejects.toMatchObject({ kind: "stalled" });
    expect(xhr.aborted).toBe(true);
  });

  it("stops watching for stalls once the body is fully sent", async () => {
    vi.useFakeTimers();
    installFakeXhr();

    const pending = postJson("https://example.test/v", { a: 1 }, { stallMs: 1000 });
    await settled();
    const xhr = FakeXhr.last!;

    xhr.emitProgress(10000, 10000);
    xhr.finishUpload();

    // The server is transcribing now. Silence here is expected work, not a
    // dead connection, so the stall clock must not be running.
    vi.advanceTimersByTime(60000);
    expect(xhr.aborted).toBe(false);

    xhr.respond(200, { valid: true });
    await expect(pending).resolves.toMatchObject({ status: 200 });
  });

  it("stops watching for stalls on byte count alone, as React Native requires", async () => {
    // RN dispatches `progress` on the upload object and never `load`, so a
    // transport that waits for `upload.onload` kills a healthy request one
    // stall budget after the last byte. The byte count is the portable signal.
    vi.useFakeTimers();
    installFakeXhr();

    const pending = postJson("https://example.test/v", { a: 1 }, { stallMs: 1000 });
    await settled();
    const xhr = FakeXhr.last!;

    xhr.emitProgress(10000, 10000);
    // Deliberately no finishUpload(): RN would never call it.
    vi.advanceTimersByTime(60000);
    expect(xhr.aborted).toBe(false);

    xhr.respond(200, { valid: true });
    await expect(pending).resolves.toMatchObject({ status: 200 });
  });

  it("treats progress events that move no bytes as a stall", async () => {
    // A connection can keep emitting events with an unchanged byte count.
    // Re-arming on every event regardless would keep that alive indefinitely.
    vi.useFakeTimers();
    installFakeXhr();

    const pending = postJson("https://example.test/v", { a: 1 }, { stallMs: 1000 });
    await settled();
    const xhr = FakeXhr.last!;

    for (let tick = 0; tick < 5; tick++) {
      vi.advanceTimersByTime(300);
      xhr.emitProgress(500, 10000);
    }

    await expect(pending).rejects.toMatchObject({ kind: "stalled" });
  });

  it("still gives up at the deadline", async () => {
    vi.useFakeTimers();
    installFakeXhr();

    const pending = postJson("https://example.test/v", { a: 1 }, { deadlineMs: 5000 });
    await settled();
    const xhr = FakeXhr.last!;

    xhr.emitProgress(1, 10000);
    vi.advanceTimersByTime(5001);

    await expect(pending).rejects.toMatchObject({ kind: "deadline" });
    expect(xhr.aborted).toBe(true);
  });
});

describe("postJson: outcomes", () => {
  it("returns a non-2xx rather than throwing, so the status survives", async () => {
    installFakeXhr();
    const pending = postJson("https://example.test/v", {});
    await settled();
    FakeXhr.last!.respond(413, { error: "too big", reason: "payload_too_large" });

    const response = await pending;
    expect(response.status).toBe(413);
    expect(response.body).toMatchObject({ reason: "payload_too_large" });
  });

  it("throws a network failure distinctly from a timeout", async () => {
    installFakeXhr();
    const pending = postJson("https://example.test/v", {});
    await settled();
    FakeXhr.last!.failNetwork();

    await expect(pending).rejects.toBeInstanceOf(TransportError);
    await expect(pending).rejects.toMatchObject({ kind: "network" });
  });

  it("honours a caller's abort signal", async () => {
    installFakeXhr();
    const controller = new AbortController();
    const pending = postJson("https://example.test/v", {}, { signal: controller.signal });
    await settled();
    controller.abort();

    await expect(pending).rejects.toMatchObject({ kind: "aborted" });
  });

  it("rejects immediately when the signal is already aborted", async () => {
    installFakeXhr();
    const controller = new AbortController();
    controller.abort();

    await expect(
      postJson("https://example.test/v", {}, { signal: controller.signal }),
    ).rejects.toMatchObject({ kind: "aborted" });
  });

  it("reads response headers case-insensitively", async () => {
    installFakeXhr();
    const pending = postJson("https://example.test/v", {});
    await settled();
    FakeXhr.last!.respond(429, {}, { "Retry-After": "42" });

    const response = await pending;
    expect(response.header("retry-after")).toBe("42");
    expect(response.header("Retry-After")).toBe("42");
    expect(response.header("nope")).toBeNull();
  });

  it("treats an unparseable body as empty rather than an error", async () => {
    installFakeXhr();
    const pending = postJson("https://example.test/v", {});
    await settled();
    // What a gateway returns when it never reached our server.
    FakeXhr.last!.respond(502, "<html>Bad Gateway</html>");

    const response = await pending;
    expect(response.status).toBe(502);
    expect(response.body).toEqual({});
  });

  it("sends JSON and preserves caller headers", async () => {
    installFakeXhr();
    const pending = postJson(
      "https://example.test/v",
      { wallet_id: "abc" },
      { headers: { "X-API-Key": "k" } },
    );
    await settled();
    const xhr = FakeXhr.last!;

    expect(JSON.parse(xhr.sentBody!)).toEqual({ wallet_id: "abc" });
    expect(xhr.requestHeaders["X-API-Key"]).toBe("k");
    expect(xhr.requestHeaders["Content-Type"]).toBe("application/json");

    xhr.respond(200, {});
    await pending;
  });

  it("reports upload progress to the caller", async () => {
    installFakeXhr();
    const seen: Array<[number, number]> = [];
    const pending = postJson(
      "https://example.test/v",
      {},
      { onUploadProgress: (loaded, total) => seen.push([loaded, total]) },
    );
    await settled();
    const xhr = FakeXhr.last!;

    xhr.emitProgress(100, 400);
    xhr.emitProgress(400, 400);
    // A body of unknown length reports 0 rather than a bogus total.
    xhr.emitProgress(400, 0);

    expect(seen).toEqual([
      [100, 400],
      [400, 400],
      [400, 0],
    ]);

    xhr.respond(200, {});
    await pending;
  });

  it("reports a synchronous send failure as a TransportError and leaks no timers", async () => {
    vi.useFakeTimers();
    installFakeXhr();
    FakeXhr.throwOnSend = new Error("Request has not been opened");

    const pending = postJson(
      "https://example.test/v",
      { a: 1 },
      { stallMs: 1000, deadlineMs: 5000 },
    );

    await expect(pending).rejects.toBeInstanceOf(TransportError);
    // Both timers must be gone. A leak here holds the serialized body, which
    // for a real capture is roughly 850 KB, for the whole deadline.
    expect(vi.getTimerCount()).toBe(0);
    FakeXhr.throwOnSend = undefined;
  });

  it("reports a malformed URL as a TransportError rather than a raw throw", async () => {
    installFakeXhr();
    FakeXhr.throwOnOpen = new Error("Failed to execute 'open'");

    await expect(postJson("not a url", {})).rejects.toBeInstanceOf(TransportError);
    FakeXhr.throwOnOpen = undefined;
  });

  it("falls back to fetch where there is no XMLHttpRequest", async () => {
    // Node's situation, and the reason the fallback exists at all.
    expect((globalThis as Record<string, unknown>).XMLHttpRequest).toBeUndefined();

    const fetchMock = vi.fn(
      async () =>
        new Response(JSON.stringify({ valid: true }), {
          status: 200,
          headers: { "content-type": "application/json" },
        }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const response = await postJson("https://example.test/v", { a: 1 });
    expect(response.status).toBe(200);
    expect(response.body).toMatchObject({ valid: true });
    expect(fetchMock).toHaveBeenCalledOnce();

    vi.unstubAllGlobals();
  });
});

describe("reason taxonomy", () => {
  it("classifies a cooldown as wait, not as a permanent failure", () => {
    // The distinction the old boolean lost: these are not "not retryable",
    // they are "not retryable yet", and the countdown is what tells a user
    // which.
    expect(reasonDisposition("rate_limited")).toBe("wait");
    expect(reasonDisposition("ip_rate_limited")).toBe("wait");
    expect(reasonDisposition("cross_wallet_cooldown")).toBe("wait");
  });

  it("never offers a retry for an oversized payload", () => {
    // A retry re-sends an identical body and earns an identical rejection.
    expect(reasonDisposition("payload_too_large")).toBe("fatal");
    expect(RETRYABLE_REASONS.has("payload_too_large")).toBe(false);
  });

  it("offers a retry for capture-quality rejections and transport faults", () => {
    for (const reason of [
      "variance_floor",
      "entropy_bounds",
      "temporal_coupling_low",
      "phrase_content_mismatch",
      "captcha_required",
      "validation_unavailable",
      "validation_timeout",
    ] as const) {
      expect(reasonDisposition(reason)).toBe("retry");
      expect(RETRYABLE_REASONS.has(reason)).toBe(true);
    }
  });

  it("fails closed on an unknown or absent reason", () => {
    // A newer server must not be able to hand an older client a retry it
    // does not understand.
    expect(reasonDisposition("something_invented_later")).toBe("fatal");
    expect(reasonDisposition(undefined)).toBe("fatal");
    expect(isVerificationReason("something_invented_later")).toBe(false);
    expect(isVerificationReason(undefined)).toBe(false);
    expect(isVerificationReason(42)).toBe(false);
  });

  it("does not mistake an inherited property for a reason", () => {
    // `reason` comes straight off a server body, so every string is reachable
    // input. An `in` check against the disposition table matched the whole
    // prototype chain: `reasonDisposition("toString")` returned
    // `Object.prototype.toString`, a function, from a function declared to
    // return one of three string literals.
    for (const inherited of [
      "toString",
      "constructor",
      "valueOf",
      "hasOwnProperty",
      "__proto__",
      "isPrototypeOf",
      "propertyIsEnumerable",
    ]) {
      expect(isVerificationReason(inherited)).toBe(false);
      expect(reasonDisposition(inherited)).toBe("fatal");
    }
  });

  it("only ever returns one of the three dispositions", () => {
    // The bug above produced a value outside the union without TypeScript
    // seeing it, because the unsound narrowing happened at runtime. Assert
    // the return type holds for hostile input as well as valid input.
    const probes = [
      ...RETRYABLE_REASONS,
      ...COOLDOWN_REASONS,
      "toString",
      "__proto__",
      "",
      "payload_too_large",
      "not_a_reason",
    ];
    for (const probe of probes) {
      expect(["retry", "wait", "fatal"]).toContain(reasonDisposition(probe));
    }
  });

  it("keeps the exported sets derived from one table, not hand-listed", () => {
    // Regression guard on the drift this module exists to end: the sets are
    // built from DISPOSITIONS, so membership and classification cannot
    // disagree.
    for (const reason of RETRYABLE_REASONS) {
      expect(reasonDisposition(reason)).toBe("retry");
    }
    for (const reason of COOLDOWN_REASONS) {
      expect(reasonDisposition(reason)).toBe("wait");
    }
    expect(RETRYABLE_REASONS.size).toBeGreaterThan(0);
    expect(COOLDOWN_REASONS.size).toBeGreaterThan(0);
  });

  it("covers every reason the executor and validator actually emit", () => {
    // Sourced from `executor-node::AppError::into_response` and
    // `entros-validation::ReasonCode::safe_label`. A reason added on either
    // side without landing here would silently classify as fatal.
    for (const reason of [
      "variance_floor",
      "entropy_bounds",
      "temporal_coupling_low",
      "phrase_content_mismatch",
      "captcha_required",
      "rate_limited",
      "ip_rate_limited",
      "cross_wallet_cooldown",
      "payload_too_large",
    ] as const) {
      expect(isVerificationReason(reason)).toBe(true);
    }
  });
});
