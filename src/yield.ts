/**
 * Cooperative yield to the host environment's event loop.
 *
 * Heavy synchronous work in feature extraction (F0 detection, HNR
 * autocorrelation, LPC formant analysis, Meyda spectral) blocks the
 * main thread for tens to hundreds of milliseconds on a single tight
 * loop. The verify UI sets `processingStage = "Extracting features..."`
 * but the spinner can't repaint while the thread is busy — leading to
 * the visible "stuck" stage the user reports.
 *
 * Calling `await yieldToMainThread()` between heavy stages hands control
 * back to the browser long enough to flush a paint frame, then resumes.
 * `MessageChannel` is the lowest-overhead path that still rounds through
 * a macrotask (microtasks don't yield to paint). Falls back to
 * `setTimeout(fn, 0)` for non-DOM environments (Node tests, React
 * Native) and a no-op when neither is available.
 */
export function yieldToMainThread(): Promise<void> {
  return new Promise<void>((resolve) => {
    if (typeof MessageChannel !== "undefined") {
      const channel = new MessageChannel();
      channel.port1.onmessage = () => {
        channel.port1.close();
        resolve();
      };
      channel.port2.postMessage(null);
      return;
    }
    if (typeof setTimeout !== "undefined") {
      setTimeout(resolve, 0);
      return;
    }
    resolve();
  });
}
