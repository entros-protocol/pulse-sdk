import {afterEach, describe, expect, it, vi} from "vitest";
import {startPairedRecorder} from "../src/paired/recorder";

vi.mock("../src/sensor/audio", () => ({
  audioCaptureConstraints: () => ({}), detectVirtualAudioInput: async () => false,
  readVoiceIsolationApplied: () => null,
}));

afterEach(() => vi.unstubAllGlobals());

function device() {
  const processor: {onaudioprocess: ((event: AudioProcessingEvent) => void) | null; connect():void; disconnect():void} = {
    onaudioprocess: null, connect() {}, disconnect() {},
  };
  const stop = vi.fn();
  vi.stubGlobal("navigator", {mediaDevices:{getUserMedia:async () => ({getTracks:() => [{stop}]})}});
  vi.stubGlobal("AudioContext", class {
    sampleRate = 16000;
    destination = {};
    async resume() {}
    async close() {}
    createMediaStreamSource() {return {connect() {},disconnect() {}};}
    createScriptProcessor() {return processor;}
  });
  return {stop, emit(samples: Float32Array) {
    processor.onaudioprocess?.({inputBuffer:{getChannelData:() => samples}} as unknown as AudioProcessingEvent);
  }};
}

describe("paired recorder buffer boundaries", () => {
  it("retains the unfinished frame when asked to release through a future reveal", async () => {
    const mic = device();
    const levels: number[] = [];
    const recorder = await startPairedRecorder((level) => levels.push(level));
    mic.emit(new Float32Array(1000).fill(0.25));
    recorder.releaseBefore(10000);
    mic.emit(new Float32Array(1000).fill(0.25));
    expect(levels).toHaveLength(2);
    expect(levels[1]).toBe(0.25);
    expect(recorder.slice(800,1600)).toEqual(new Float32Array(800).fill(0.25));
    await recorder.stop();
  });

  it("publishes each frame position before its callback can release audio", async () => {
    const mic = device();
    const positions: number[] = [];
    const recorder = await startPairedRecorder((_level,end) => {
      positions.push(recorder.framedSamples());
      recorder.releaseBefore(end);
    });
    mic.emit(new Float32Array(2000).fill(0.25));
    expect(positions).toEqual([800,1600]);
    await recorder.stop();
  });
});
