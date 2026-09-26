import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { decodePcm16, encodePcm16 } from "../src/sensor/encode";
import { COARSE_PATH_POINTS, OUTLINE_MARGIN, scorePath, toCoarsePath } from "../src/paired/coarse-path";
import { analysisSignal, MAX_ROUND_SAMPLES, roundWindow, TRIM_TAIL_SAMPLES } from "../src/paired/segment";
import { createRoundTracker, FRAME_SAMPLES, WAYPOINT_REACH } from "../src/paired/tracker";
import {
  attemptBindingDigest,
  attestationDigest,
  audioDigest,
  challengeDigest,
  commitRequestDigest,
  DOMAINS,
  encode,
  encodeCoarsePath,
  encodePathTarget,
  evidenceManifest,
  finalDigest,
  PairedEncodingError,
  pathDigest,
  roundCommitment,
  sessionCommitment,
  toHex,
} from "../src/paired/transcript";

const VECTORS_SHA256 = "cb88f752aed0e29a0f2321e85a2ff3006e3c1f65a933d2c729c069574445e63d";
const vectorsText = readFileSync(resolve(__dirname, "fixtures/paired-round-vectors.json"), "utf8");
const vectors = JSON.parse(vectorsText);

const bytes = (hex: string) => Uint8Array.from(Buffer.from(hex, "hex"));
const points = (pairs: number[][]) => pairs.map(([x, y]) => ({ x: x!, y: y! }));

describe("paired-round vectors", () => {
  it("is the pinned generation", () => {
    expect(createHash("sha256").update(vectorsText).digest("hex")).toBe(VECTORS_SHA256);
  });

  it("uses the generator's constants", () => {
    expect(vectors.constants.maxRoundSamples).toBe(MAX_ROUND_SAMPLES);
    expect(vectors.constants.trimTailSamples).toBe(TRIM_TAIL_SAMPLES);
    expect(vectors.tracker.frameSamples).toBe(FRAME_SAMPLES);
    expect(vectors.tracker.waypointReach).toBe(WAYPOINT_REACH);
    expect(vectors.tracker.outlineMargin).toBe(OUTLINE_MARGIN);
    expect(vectors.tracker.coarsePathPoints).toBe(COARSE_PATH_POINTS);
  });

  it("uses the generator's domains", () => {
    const decoder = new TextDecoder();
    expect(decoder.decode(DOMAINS.session)).toBe(vectors.domains.session);
    expect(decoder.decode(DOMAINS.round)).toBe(vectors.domains.round);
    expect(decoder.decode(DOMAINS.final)).toBe(vectors.domains.final);
    expect(decoder.decode(DOMAINS.attestation)).toBe(vectors.domains.attestation);
  });

  it("reproduces every trace-tier session", () => {
    for (const session of vectors.sessions.filter((entry: { tier: string }) => entry.tier === "trace")) {
      const nonce = bytes(session.sessionNonceHex);
      const attempt = attemptBindingDigest(
        bytes(session.serverAttemptIdHex),
        bytes(session.originalChallengeNonceHex),
      );
      expect(toHex(attempt)).toBe(session.attemptBindingDigestHex);
      let previous = sessionCommitment(nonce, attempt, session.rounds, session.sessionExpiryUnixMs);
      expect(toHex(previous)).toBe(session.sessionCommitmentHex);
      const manifest = [];
      for (const round of session.roundEntries) {
        const target = encodePathTarget(points(round.pathTargetWaypoints));
        expect(toHex(target)).toBe(round.pathTargetHex);
        const path = encodeCoarsePath(points(round.coarsePathPoints));
        expect(toHex(path)).toBe(round.coarsePathHex);
        const challenge = challengeDigest(nonce, round.index, bytes(round.roundNonceHex), round.word, target);
        expect(toHex(challenge)).toBe(round.challengeDigestHex);
        const segment = bytes(round.audioSegmentHex);
        const audio = audioDigest(nonce, round.index, challenge, round.audioFormat, segment);
        const pathHash = pathDigest(nonce, round.index, challenge, path);
        expect(toHex(audio)).toBe(round.audioDigestHex);
        expect(toHex(pathHash)).toBe(round.pathDigestHex);
        const fields = {
          sessionNonce: nonce,
          roundIndex: round.index,
          roundNonce: bytes(round.roundNonceHex),
          challenge,
          previous,
          audioFormat: round.audioFormat,
          audioByteLength: segment.length,
          pathPointCount: round.pathPointCount,
        };
        const current = roundCommitment({ ...fields, audio, path: pathHash });
        expect(toHex(current)).toBe(round.commitmentHex);
        expect(toHex(commitRequestDigest({ ...fields, current }))).toBe(round.requestDigestHex);
        manifest.push({ audio, path: pathHash });
        previous = current;
      }
      const evidence = evidenceManifest(manifest);
      expect(toHex(evidence)).toBe(session.evidenceManifestHex);
      expect(toHex(finalDigest(nonce, previous, session.rounds, evidence))).toBe(session.finalDigestHex);
    }
  });

  it("rejects every invalid encoding the generator lists", () => {
    for (const entry of vectors.invalidEncodings) {
      if (entry.encoding === "tier") continue;
      const build = () =>
        entry.encoding === "pathTarget"
          ? encodePathTarget(points(entry.waypoints))
          : encodeCoarsePath(points(entry.points));
      expect(build, entry.name).toThrow(PairedEncodingError);
      try {
        build();
      } catch (error) {
        expect((error as PairedEncodingError).reason, entry.name).toBe(entry.reason);
      }
    }
  });

  it("separates fields that naive concatenation merges", () => {
    const encoder = new TextEncoder();
    const separation = vectors.separation;
    expect(toHex(encode(separation.leftFields.map((text: string) => encoder.encode(text))))).toBe(
      separation.encodedLeftHex,
    );
    expect(toHex(encode(separation.rightFields.map((text: string) => encoder.encode(text))))).toBe(
      separation.encodedRightHex,
    );
  });

  it("reproduces every commitment recompute case", () => {
    for (const entry of vectors.commitRecompute) {
      const commitment = roundCommitment({
        sessionNonce: bytes(entry.sessionNonceHex),
        roundIndex: entry.roundIndex,
        roundNonce: bytes(entry.roundNonceHex),
        challenge: bytes(entry.challengeDigestHex),
        previous: bytes(entry.previousCommitmentHex),
        audioFormat: entry.audioFormat,
        audioByteLength: entry.audioByteLength,
        audio: bytes(entry.audioDigestHex),
        pathPointCount: entry.pathPointCount,
        path: bytes(entry.pathDigestHex),
      });
      expect(toHex(commitment), entry.name).toBe(entry.expectedCommitmentHex);
    }
  });

  it("reproduces every round window", () => {
    for (const entry of vectors.roundWindows) {
      const window = roundWindow(
        entry.roundStart,
        entry.roundEnd,
        entry.voicedRuns.map(([start, end]: number[]) => ({ start: start!, end: end! })),
      );
      expect([window.start, window.end], entry.name).toEqual([entry.windowStart, entry.windowEnd]);
    }
  });

  it("encodes and decodes PCM16 by the shared rule", () => {
    const encoded = encodePcm16(Float32Array.from(vectors.pcm16.samples));
    expect(toHex(encoded)).toBe(vectors.pcm16.pcm16Hex);
    expect(Array.from(decodePcm16(encoded))).toEqual(vectors.pcm16.decoded);
  });

  it("levels the joined segments into the shared analysis signal", () => {
    for (const entry of vectors.analysisSignal) {
      const signal = analysisSignal(entry.segmentsPcm16Hex.map(bytes));
      expect(signal.length).toBe(entry.sampleCount);
      const hash = createHash("sha256")
        .update(Buffer.from(signal.buffer, signal.byteOffset, signal.byteLength))
        .digest("hex");
      expect(hash, entry.name).toBe(entry.signalF32LeSha256Hex);
    }
  });

  it("reaches every tracker decision the generator reaches", () => {
    for (const entry of vectors.trackerSequences) {
      const tracker = createRoundTracker();
      for (const [level, count] of entry.priorLevels) {
        for (let index = 0; index < count; index++) tracker.observe(level);
      }
      tracker.begin(points(entry.waypoints), entry.traceRequired);
      const reaches = new Map<number, { x: number; y: number }[]>();
      for (const [frame, x, y] of entry.reaches) {
        reaches.set(frame, [...(reaches.get(frame) ?? []), { x, y }]);
      }
      const decisions: string[] = [];
      let frame = 0;
      for (const [level, count] of entry.levels) {
        for (let index = 0; index < count; index++) {
          for (const point of reaches.get(frame) ?? []) tracker.reach(point);
          decisions.push(tracker.frame(level));
          frame++;
        }
      }
      const runLength: [string, number][] = [];
      for (const decision of decisions) {
        const last = runLength[runLength.length - 1];
        if (last && last[0] === decision) last[1]++;
        else runLength.push([decision, 1]);
      }
      expect(runLength, entry.name).toEqual(entry.decisions);
      const completed = decisions.indexOf("complete");
      expect(completed === -1 ? null : completed, entry.name).toBe(entry.completedAtFrame);
      expect(tracker.runs(), entry.name).toEqual(entry.finalRuns);
    }
  });

  it("commits the outline the generator commits", () => {
    for (const entry of vectors.coarsePaths) {
      const trace = entry.trace.map(([x, y, t]: number[]) => ({ x: x!, y: y!, t: t! }));
      const [width, height] = entry.surface;
      expect(toCoarsePath(trace, { width, height }), entry.name).toEqual(points(entry.outline));
    }
  });

  it("scores every outline the way the server does", () => {
    for (const entry of vectors.pathScoring) {
      expect(scorePath(points(entry.waypoints), points(entry.outline)), entry.name).toEqual({
        reached: entry.reached,
        inOrder: entry.inOrder,
      });
    }
  });

  it("reproduces every attestation digest", () => {
    for (const entry of vectors.attestation) {
      const digest = attestationDigest(
        entry.protocolVersion,
        bytes(entry.sessionNonceHex),
        bytes(entry.attemptBindingDigestHex),
        bytes(entry.finalDigestHex),
        entry.projectionVersion,
      );
      expect(toHex(digest), entry.name).toBe(entry.requestHash);
    }
  });
});
