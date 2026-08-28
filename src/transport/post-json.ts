/**
 * The one place the SDK sends a large request body.
 *
 * `/validate-features` carries base64 audio, so on a phone it is hundreds of
 * kilobytes against an uplink measured in hundreds of kilobits. A verification
 * failed in production because a fixed 15-second abort covered the upload as
 * well as the server's work, and a perfectly healthy 9.4-second upload ran out
 * of budget. The user was told the verification service was unreachable. It
 * was not: their connection was slow, and nothing in the client could tell the
 * difference.
 *
 * That is the distinction this module exists to make. Upload progress resets
 * the stall clock. The caller deadline still bounds total elapsed time.
 *
 * Telling those apart needs upload progress, and upload progress needs
 * `XMLHttpRequest`. A streaming `fetch` body would do it too, but only with
 * `duplex: "half"`, which is Chromium-only, and the bug is on iOS. So XHR is
 * the transport wherever it exists, which is every browser and React Native.
 * Node has no XHR, so `fetch` covers it; that path cannot see progress and
 * falls back to the deadline alone. Nothing in production runs it.
 */

/** Why a request ended without a response. */
export type TransportFailureKind =
  /** No upload progress for `stallMs`. The connection went quiet. */
  | "stalled"
  /** Ran past `deadlineMs`. Past this point the attempt cannot succeed. */
  | "deadline"
  /** The caller's `signal` fired. */
  | "aborted"
  /** DNS, TLS, CORS, offline. The request never reached a server. */
  | "network";

export class TransportError extends Error {
  readonly kind: TransportFailureKind;

  constructor(kind: TransportFailureKind, message: string) {
    super(message);
    this.name = "TransportError";
    this.kind = kind;
  }
}

export interface PostJsonOptions {
  headers?: Record<string, string>;
  /**
   * Called as the body goes out. `total` is 0 when the length is not
   * computable. Never called on the `fetch` fallback.
   */
  onUploadProgress?: (loaded: number, total: number) => void;
  /**
   * Abort when the upload makes no progress for this long. Ignored on the
   * `fetch` fallback, which cannot observe progress.
   */
  stallMs?: number;
  /**
   * Abort outright after this many milliseconds, whatever is happening.
   *
   * Callers should derive this from the server's own challenge validity
   * rather than inventing a number: an upload landing after the challenge
   * expires is refused anyway, so that is the point past which waiting is
   * pointless.
   */
  deadlineMs?: number;
  /** Host cancellation. */
  signal?: AbortSignal;
}

export interface PostJsonResponse {
  status: number;
  /** Parsed JSON body, or `{}` when absent, empty, or not JSON. */
  body: Record<string, unknown>;
  /**
   * Case-insensitive response header lookup.
   *
   * Cross-origin this only sees headers the server lists in
   * `Access-Control-Expose-Headers`, which is why callers should prefer a
   * value carried in the JSON body when the server offers both. The executor
   * sends `retry_after` in the body precisely so a browser can read it.
   */
  header(name: string): string | null;
}

function parseJson(text: string): Record<string, unknown> {
  if (!text) return {};
  try {
    const parsed: unknown = JSON.parse(text);
    return parsed !== null && typeof parsed === "object"
      ? (parsed as Record<string, unknown>)
      : {};
  } catch {
    // A proxy or gateway error page rather than our server. The status still
    // classifies it, so an unparseable body is not itself an error.
    return {};
  }
}

/** Parse the raw block `XMLHttpRequest.getAllResponseHeaders()` returns. */
function parseHeaderBlock(raw: string): Map<string, string> {
  const headers = new Map<string, string>();
  for (const line of raw.split("\r\n")) {
    const separator = line.indexOf(":");
    if (separator <= 0) continue;
    headers.set(
      line.slice(0, separator).trim().toLowerCase(),
      line.slice(separator + 1).trim(),
    );
  }
  return headers;
}

function postViaXhr(
  url: string,
  body: string,
  options: PostJsonOptions,
): Promise<PostJsonResponse> {
  const { headers = {}, onUploadProgress, stallMs, deadlineMs, signal } = options;

  return new Promise<PostJsonResponse>((resolve, reject) => {
    let xhr: XMLHttpRequest;
    try {
      xhr = new XMLHttpRequest();
    } catch (err) {
      reject(
        new TransportError(
          "network",
          err instanceof Error ? err.message : String(err),
        ),
      );
      return;
    }
    let stallTimer: ReturnType<typeof setTimeout> | undefined;
    let deadlineTimer: ReturnType<typeof setTimeout> | undefined;
    let settled = false;
    let uploadComplete = false;
    let lastLoaded = -1;

    const cleanup = () => {
      if (stallTimer !== undefined) clearTimeout(stallTimer);
      if (deadlineTimer !== undefined) clearTimeout(deadlineTimer);
      signal?.removeEventListener("abort", onHostAbort);
    };

    const fail = (kind: TransportFailureKind, message: string) => {
      if (settled) return;
      settled = true;
      cleanup();
      // Abort after marking settled so the resulting `onerror` is ignored.
      try {
        xhr.abort();
      } catch {
        // A settled XHR throws on abort in some engines. Nothing to do.
      }
      reject(new TransportError(kind, message));
    };

    function onHostAbort() {
      fail("aborted", "Verification was cancelled.");
    }

    // Reset on real movement only. This is the whole design: the clock
    // measures silence, not duration, so a slow upload survives and a dead
    // one does not. Re-arming on every event regardless of the byte count
    // would keep a connection alive that has emitted a hundred progress
    // events without sending a byte.
    const armStallTimer = () => {
      if (!stallMs || stallMs <= 0 || uploadComplete) return;
      if (stallTimer !== undefined) clearTimeout(stallTimer);
      stallTimer = setTimeout(
        () => fail("stalled", `Upload stalled for ${stallMs}ms.`),
        stallMs,
      );
    };

    // The body is out. Stop watching for stalls. The server is thinking now,
    // and a long silence while it transcribes is expected, not a fault. The
    // deadline still applies.
    const markUploadComplete = () => {
      uploadComplete = true;
      if (stallTimer !== undefined) {
        clearTimeout(stallTimer);
        stallTimer = undefined;
      }
    };

    // `open` and `setRequestHeader` throw synchronously on a malformed URL or
    // a forbidden header name. Routed through `fail` so the caller always
    // receives a `TransportError`, which is what this module documents and
    // what `pulse.ts` branches on.
    try {
      xhr.open("POST", url, true);
      xhr.setRequestHeader("Content-Type", "application/json");
      for (const [name, value] of Object.entries(headers)) {
        if (name.toLowerCase() === "content-type") continue;
        xhr.setRequestHeader(name, value);
      }
    } catch (err) {
      fail("network", err instanceof Error ? err.message : String(err));
      return;
    }

    xhr.upload.onprogress = (event: ProgressEvent) => {
      // Native HTTP retries can restart the count. Any changed count is progress.
      if (event.loaded !== lastLoaded) {
        lastLoaded = event.loaded;
        armStallTimer();
      }
      // React Native dispatches `progress` on the upload object and nothing
      // else: `load` fires on the request, never on `upload`. Waiting for
      // `upload.onload` there would leave the stall clock running through the
      // server's entire thinking time and kill a healthy request a stall
      // budget after the last byte went out. The byte count says the same
      // thing and every engine reports it.
      if (event.lengthComputable && event.total > 0 && event.loaded >= event.total) {
        markUploadComplete();
      }
      try {
        onUploadProgress?.(event.loaded, event.lengthComputable ? event.total : 0);
      } catch {
        // A progress observer cannot change the request outcome.
      }
    };

    xhr.upload.onload = markUploadComplete;

    xhr.onload = () => {
      if (settled) return;
      settled = true;
      cleanup();
      const headerMap = parseHeaderBlock(xhr.getAllResponseHeaders());
      resolve({
        status: xhr.status,
        body: parseJson(xhr.responseText),
        header: (name: string) => headerMap.get(name.toLowerCase()) ?? null,
      });
    };

    xhr.onerror = () => fail("network", "Could not reach the verification service.");
    xhr.onabort = () => fail("aborted", "Verification was cancelled.");

    if (signal) {
      if (signal.aborted) {
        onHostAbort();
        return;
      }
      signal.addEventListener("abort", onHostAbort, { once: true });
    }

    if (deadlineMs && deadlineMs > 0) {
      deadlineTimer = setTimeout(
        () => fail("deadline", `Request exceeded its ${deadlineMs}ms deadline.`),
        deadlineMs,
      );
    }

    armStallTimer();
    // A synchronous throw here would reject the promise while leaving both
    // timers armed and the abort listener attached, holding the serialized
    // body alive for as long as the deadline.
    try {
      xhr.send(body);
    } catch (err) {
      fail("network", err instanceof Error ? err.message : String(err));
    }
  });
}

async function postViaFetch(
  url: string,
  body: string,
  options: PostJsonOptions,
): Promise<PostJsonResponse> {
  const { headers = {}, deadlineMs, signal } = options;
  const controller = new AbortController();
  let deadlineTimer: ReturnType<typeof setTimeout> | undefined;
  let timedOut = false;

  const onHostAbort = () => controller.abort();
  if (signal) {
    if (signal.aborted) {
      throw new TransportError("aborted", "Verification was cancelled.");
    }
    signal.addEventListener("abort", onHostAbort, { once: true });
  }
  if (deadlineMs && deadlineMs > 0) {
    deadlineTimer = setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, deadlineMs);
  }

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...headers },
      body,
      signal: controller.signal,
    });
    const text = await response.text();
    return {
      status: response.status,
      body: parseJson(text),
      header: (name: string) => response.headers.get(name),
    };
  } catch (err) {
    const aborted = err instanceof Error && err.name === "AbortError";
    if (aborted && timedOut) {
      throw new TransportError("deadline", `Request exceeded its ${deadlineMs}ms deadline.`);
    }
    if (aborted) {
      throw new TransportError("aborted", "Verification was cancelled.");
    }
    throw new TransportError(
      "network",
      err instanceof Error ? err.message : String(err),
    );
  } finally {
    if (deadlineTimer !== undefined) clearTimeout(deadlineTimer);
    signal?.removeEventListener("abort", onHostAbort);
  }
}

/**
 * POST a JSON body and resolve with the response, whatever its status.
 *
 * A non-2xx is a normal return, not a throw. The status is the information the
 * caller needs, and collapsing 413, 429 and 500 into one exception is how they
 * became indistinguishable in the first place. Only a request that never
 * produced a response throws, and it throws a {@link TransportError} saying
 * which way it failed.
 */
export async function postJson(
  url: string,
  payload: unknown,
  options: PostJsonOptions = {},
): Promise<PostJsonResponse> {
  const body = JSON.stringify(payload);
  return typeof XMLHttpRequest === "undefined"
    ? postViaFetch(url, body, options)
    : postViaXhr(url, body, options);
}
