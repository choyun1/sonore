// The audio thread's part of the live page: copy each buffer of the
// microphone (128 samples, first channel) straight to the analysis worker,
// through the port the page hands over, so the page's own thread (busy
// drawing) is not in the way. Nothing else runs here, so a slow analysis can
// delay the picture but never the audio.
class Capture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.worker = null;
    this.port.onmessage = ({ data }) => { this.worker = data; };
  }

  process(inputs) {
    const channel = inputs[0][0];
    if (channel && this.worker) {
      const samples = channel.slice(0);
      // currentFrame numbers the samples, so the worker can tell a gap from a continuous sound.
      this.worker.postMessage({ type: "audio", samples, frame: currentFrame }, [samples.buffer]);
    }
    return true;
  }
}
registerProcessor("capture", Capture);
