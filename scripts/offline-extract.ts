/**
 * Offline audio feature extractor.
 *
 * Decodes a WAV file (or directory of WAV files) and runs each through
 * the SDK's runtime `extractSpeakerFeatures` to produce 170-dim feature
 * vectors with bit-identical distribution to a browser-captured session.
 *
 * Pipeline mirrors the runtime capture path:
 *   WAV bytes
 *     → wavefile decode → Float32Array PCM (16kHz mono)
 *     → normalizeCaptureRMS (matches SDK capture-time gain)
 *     → AudioCapture { samples, sampleRate, duration }
 *     → extractSpeakerFeatures (existing extractor, treated as black box)
 *     → 170-dim feature vector
 *
 * Internal calibration tool — not exported, not in the published runtime,
 * not in the npm package surface (lives under `scripts/`, excluded from
 * the `files: ["dist"]` allowlist in package.json).
 *
 * Usage:
 *   npx tsx scripts/offline-extract.ts <wav-file>
 *   npx tsx scripts/offline-extract.ts --dir <directory> [--out features.jsonl]
 *   npx tsx scripts/offline-extract.ts --list <paths-file> [--out features.jsonl]
 *
 * `--list` reads newline-delimited absolute paths from a file. Used by the
 * Phase B2 corpus prep shell script to split a stratified file list across
 * N parallel Node processes (each instance handles one chunk).
 *
 * Format notes:
 *   - Input is WAV only. LibriSpeech (FLAC-native) must be pre-converted
 *     to 16kHz mono WAV (one-shot ffmpeg step in the corpus prep pipeline).
 *   - Sample rate auto-resamples to 16kHz if the WAV is at a different rate.
 *   - Stereo WAVs are downmixed to the left channel; the SDK is mono.
 *   - Output is JSONL when --out is used (one object per line, append-safe),
 *     or pretty-printed JSON for single-file mode (easier to eyeball).
 */

import {
  closeSync,
  openSync,
  readdirSync,
  readFileSync,
  statSync,
  writeSync,
} from "node:fs";
import { join, relative } from "node:path";

import { WaveFile } from "wavefile";

import { extractSpeakerFeatures } from "../src/extraction/speaker";
import { normalizeCaptureRMS } from "../src/sensor/audio";
import type { AudioCapture } from "../src/sensor/types";

const SDK_SAMPLE_RATE = 16_000;

interface ExtractResult {
  path: string;
  duration: number;
  features: number[];
}

function loadWavAsCapture(path: string): AudioCapture {
  const buffer = readFileSync(path);
  const wav = new WaveFile(buffer);

  // Resample to the SDK's expected rate if the WAV is at a different one.
  // LibriSpeech-clean-100 is 16kHz natively so this is usually a no-op,
  // but we keep the guard for other corpora (e.g., VoxCeleb at 16kHz,
  // Common Voice at variable rates).
  if (wav.fmt.sampleRate !== SDK_SAMPLE_RATE) {
    wav.toSampleRate(SDK_SAMPLE_RATE);
  }

  // `getSamples(false, Float32Array)` returns Float32Array in [-1, 1].
  // For mono: a single array. For stereo: [left, right]. We take the
  // first (left) channel only — the SDK is mono and downmixing requires
  // a calibration the offline pipeline doesn't have access to.
  const raw = wav.getSamples(false, Float32Array);
  const monoSamples: Float32Array = Array.isArray(raw) ? raw[0]! : raw;

  // Apply the SAME RMS gain the SDK applies during capture so the offline
  // distribution matches the runtime distribution. Without this, files
  // recorded at different volumes would produce different amplitude
  // features, contaminating the calibration corpus with mic-gain variance
  // that the SDK already removes from real captures.
  const normalizedSamples = normalizeCaptureRMS(monoSamples);

  return {
    samples: normalizedSamples,
    sampleRate: SDK_SAMPLE_RATE,
    duration: normalizedSamples.length / SDK_SAMPLE_RATE,
  };
}

async function extractOne(path: string): Promise<ExtractResult> {
  const capture = loadWavAsCapture(path);
  const features = await extractSpeakerFeatures(capture);
  return { path, duration: capture.duration, features };
}

function* walkWavFiles(dir: string): Generator<string> {
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    const st = statSync(full);
    if (st.isDirectory()) {
      yield* walkWavFiles(full);
    } else if (st.isFile() && entry.toLowerCase().endsWith(".wav")) {
      yield full;
    }
  }
}

function parseArgs(argv: string[]): {
  dir: string | null;
  list: string | null;
  out: string | null;
  file: string | null;
} {
  let dir: string | null = null;
  let list: string | null = null;
  let out: string | null = null;
  let file: string | null = null;
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--dir") {
      dir = argv[++i] ?? null;
    } else if (a === "--list") {
      list = argv[++i] ?? null;
    } else if (a === "--out") {
      out = argv[++i] ?? null;
    } else if (a && !a.startsWith("--")) {
      file = a;
    }
  }
  return { dir, list, out, file };
}

async function runBatch(
  paths: Iterable<string>,
  outPath: string | null,
  pathRelativeTo: string | null,
): Promise<void> {
  const outFd = outPath !== null ? openSync(outPath, "w") : null;
  let count = 0;
  let errors = 0;
  for (const path of paths) {
    try {
      const result = await extractOne(path);
      const line = JSON.stringify({
        ...result,
        path: pathRelativeTo !== null ? relative(pathRelativeTo, path) : path,
      });
      if (outFd !== null) {
        writeSync(outFd, line + "\n");
      } else {
        console.log(line);
      }
      count += 1;
      if (count % 100 === 0) {
        process.stderr.write(`[${count} extracted] ${path}\n`);
      }
    } catch (err) {
      errors += 1;
      const msg = err instanceof Error ? err.message : String(err);
      process.stderr.write(`[error] ${path}: ${msg}\n`);
    }
  }
  if (outFd !== null) closeSync(outFd);
  process.stderr.write(`Done. ${count} extracted, ${errors} errors.\n`);
}

async function main(): Promise<void> {
  const { dir, list, out, file } = parseArgs(process.argv.slice(2));

  if (dir) {
    // Walk a directory tree recursively. Paths in the JSONL are relative
    // to `dir` so output stays portable across machines.
    await runBatch(walkWavFiles(dir), out, dir);
    return;
  }

  if (list) {
    // Read newline-delimited paths. Used by the corpus-prep shell to
    // fan a stratified file list across N parallel tsx processes.
    const paths = readFileSync(list, "utf8")
      .split("\n")
      .map((s) => s.trim())
      .filter((s) => s.length > 0);
    await runBatch(paths, out, null);
    return;
  }

  if (file) {
    // Single-file mode: pretty-printed JSON for human inspection.
    const result = await extractOne(file);
    console.log(JSON.stringify(result, null, 2));
    return;
  }

  process.stderr.write(
    "Usage:\n" +
      "  npx tsx scripts/offline-extract.ts <wav-file>\n" +
      "  npx tsx scripts/offline-extract.ts --dir <directory> [--out features.jsonl]\n" +
      "  npx tsx scripts/offline-extract.ts --list <paths-file> [--out features.jsonl]\n",
  );
  process.exit(1);
}

main().catch((err: unknown) => {
  const msg = err instanceof Error ? err.stack ?? err.message : String(err);
  process.stderr.write(`${msg}\n`);
  process.exit(1);
});
