import {
  BasicPitch,
  noteFramesToTime,
  addPitchBendsToNoteEvents,
  outputToNotesPoly,
} from "@spotify/basic-pitch";

const $ = (id) => document.getElementById(id);

// Open-string MIDI notes, listed in the order they are drawn from bottom line to top line.
const TUNINGS = {
  guitar: { labels: ["E", "A", "D", "G", "B", "e"], open: [40, 45, 50, 55, 59, 64], maxFret: 17, program: 25 },
  bass: { labels: ["E", "A", "D", "G"], open: [28, 33, 38, 43], maxFret: 17, program: 33 },
  ukulele: { labels: ["G", "C", "E", "A"], open: [67, 60, 64, 69], maxFret: 12, program: 24 },
};

const NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const SHAPES = {
  "": [0, 4, 7],
  m: [0, 3, 7],
  7: [0, 4, 7, 10],
  maj7: [0, 4, 7, 11],
  m7: [0, 3, 7, 10],
};

let raw = null; // model output, kept so sliders can re-run without re-analysing
let transpose = 0; // semitones, applied to tabs, chords and the MIDI file
let currentName = "song";

// ---------- audio + model ----------

async function decodeMono(file) {
  const ctx = new AudioContext({ sampleRate: 22050 });
  const buf = await ctx.decodeAudioData(await file.arrayBuffer());
  const mono = new Float32Array(buf.length);
  for (let c = 0; c < buf.numberOfChannels; c++) {
    const data = buf.getChannelData(c);
    for (let i = 0; i < data.length; i++) mono[i] += data[i] / buf.numberOfChannels;
  }
  ctx.close();
  return mono;
}

async function runModel(file) {
  const audio = await decodeMono(file);
  const frames = [];
  const onsets = [];
  const contours = [];
  const bp = new BasicPitch(import.meta.env.BASE_URL + "model/model.json");
  await bp.evaluateModel(
    audio,
    (f, o, c) => {
      frames.push(...f);
      onsets.push(...o);
      contours.push(...c);
    },
    (p) => {
      $("bar").value = Math.round(p * 100);
    }
  );
  return { frames, onsets, contours };
}

function getNotes() {
  const s = Number($("sens").value);
  const onset = 0.8 - 0.06 * s; // sensitivity 5 -> 0.5, 10 -> 0.2
  const frame = onset * 0.6;
  const events = outputToNotesPoly(raw.frames, raw.onsets, onset, frame, 5);
  return noteFramesToTime(addPitchBendsToNoteEvents(raw.contours, events))
    .sort((a, b) => a.startTimeSeconds - b.startTimeSeconds)
    .map((n) => ({ ...n, pitchMidi: n.pitchMidi + transpose }));
}

// ---------- notes -> tab ----------

function toTab(notes, tuning) {
  let lastFret = 0;
  const out = [];
  for (const n of notes) {
    const options = [];
    tuning.open.forEach((open, string) => {
      const fret = n.pitchMidi - open;
      if (fret >= 0 && fret <= tuning.maxFret) options.push({ string, fret });
    });
    if (!options.length) continue; // outside this instrument's range
    options.sort((a, b) => Math.abs(a.fret - lastFret) - Math.abs(b.fret - lastFret));
    lastFret = options[0].fret;
    out.push({ time: n.startTimeSeconds, ...options[0] });
  }
  return out;
}

function fmtTime(t) {
  const m = Math.floor(t / 60);
  const s = Math.floor(t % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

function renderTab(positions, tuning, step = 0.25, perRow = 20) {
  if (!positions.length) return "No notes found in this instrument's range. Try raising the sensitivity.";
  const last = positions.reduce((m, p) => Math.max(m, p.time), 0);
  const cols = Math.floor(last / step) + 1;
  const strings = tuning.open.length;
  const grid = Array.from({ length: strings }, () => Array(cols).fill("---"));

  for (const p of positions) {
    const col = Math.floor(p.time / step);
    if (grid[p.string][col] === "---") grid[p.string][col] = String(p.fret).padEnd(3, "-");
  }

  const rows = [];
  for (let start = 0; start < cols; start += perRow) {
    const lines = [`${fmtTime(start * step)}`];
    for (let s = strings - 1; s >= 0; s--) {
      lines.push(`${tuning.labels[s]}|${grid[s].slice(start, start + perRow).join("")}|`);
    }
    rows.push(lines.join("\n"));
  }
  return rows.join("\n\n");
}

// ---------- notes -> chords ----------

function detectChord(windowNotes) {
  const weights = new Array(12).fill(0);
  for (const n of windowNotes) weights[n.pitchMidi % 12] += n.durationSeconds * n.amplitude;
  const total = weights.reduce((a, b) => a + b, 0);
  let best = { score: 0, name: null };
  for (let root = 0; root < 12; root++) {
    for (const [suffix, shape] of Object.entries(SHAPES)) {
      const inChord = shape.reduce((sum, i) => sum + weights[(root + i) % 12], 0);
      const score = inChord - (total - inChord) * 0.5;
      if (score > best.score) best = { score, name: NAMES[root] + suffix };
    }
  }
  return best.name;
}

function toChords(notes, windowSec) {
  if (!notes.length) return [];
  const end = notes.reduce((m, n) => Math.max(m, n.startTimeSeconds), 0);
  const out = [];
  for (let t = 0; t <= end; t += windowSec) {
    const inWindow = notes.filter((n) => n.startTimeSeconds >= t && n.startTimeSeconds < t + windowSec);
    const chord = inWindow.length ? detectChord(inWindow) : null;
    if (chord && chord !== out[out.length - 1]?.chord) out.push({ time: t, chord });
  }
  return out;
}

function renderChords(list) {
  const box = $("chords");
  box.replaceChildren();
  if (!list.length) {
    box.textContent = "No chords found. Try raising the sensitivity.";
    return;
  }
  for (const c of list) {
    const chip = document.createElement("div");
    chip.className = "chip";
    const name = document.createElement("b");
    name.textContent = c.chord;
    const time = document.createElement("small");
    time.textContent = fmtTime(c.time);
    chip.append(name, time);
    box.append(chip);
  }
}

// ---------- MIDI export ----------

function vlq(value) {
  let v = value;
  const bytes = [v & 0x7f];
  while ((v >>= 7) > 0) bytes.unshift((v & 0x7f) | 0x80);
  return bytes;
}

function buildMidi(notes, program) {
  const ppq = 480;
  const usPerBeat = 500000; // 120 BPM, so one second is 960 ticks
  const ticks = (s) => Math.round(s * 960);

  const events = [];
  for (const n of notes) {
    const pitch = Math.max(0, Math.min(127, Math.round(n.pitchMidi)));
    const velocity = Math.max(1, Math.min(127, Math.round(n.amplitude * 127)));
    const on = ticks(n.startTimeSeconds);
    const off = Math.max(on + 1, ticks(n.startTimeSeconds + n.durationSeconds));
    events.push({ tick: on, order: 1, data: [0x90, pitch, velocity] });
    events.push({ tick: off, order: 0, data: [0x80, pitch, 0] });
  }
  events.sort((a, b) => a.tick - b.tick || a.order - b.order);

  const track = [
    0x00, 0xff, 0x51, 0x03, (usPerBeat >> 16) & 255, (usPerBeat >> 8) & 255, usPerBeat & 255,
    0x00, 0xc0, program,
  ];
  let prev = 0;
  for (const e of events) {
    track.push(...vlq(e.tick - prev), ...e.data);
    prev = e.tick;
  }
  track.push(0x00, 0xff, 0x2f, 0x00);

  const len = track.length;
  const header = [0x4d, 0x54, 0x68, 0x64, 0, 0, 0, 6, 0, 0, 0, 1, ppq >> 8, ppq & 255];
  const trackHead = [0x4d, 0x54, 0x72, 0x6b, (len >> 24) & 255, (len >> 16) & 255, (len >> 8) & 255, len & 255];
  return new Uint8Array([...header, ...trackHead, ...track]);
}

function downloadMidi() {
  if (!raw) return;
  const notes = getNotes();
  if (!notes.length) {
    setStatus("No notes to export. Try raising the sensitivity.", true);
    return;
  }
  const program = TUNINGS[$("instrument").value].program;
  const blob = new Blob([buildMidi(notes, program)], { type: "audio/midi" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `${currentName}.mid`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

// ---------- UI ----------

function render() {
  if (!raw) return;
  const notes = getNotes();
  const mode = $("mode").value;
  const winSec = Number($("win").value);
  $("winOut").textContent = `${winSec}s`;
  $("tOut").textContent = transpose > 0 ? `+${transpose}` : String(transpose);

  $("chordsBox").hidden = mode === "tabs";
  $("tabBox").hidden = mode === "chords";

  if (mode !== "tabs") renderChords(toChords(notes, winSec));
  if (mode !== "chords") {
    const tuning = TUNINGS[$("instrument").value];
    $("tab").textContent = renderTab(toTab(notes, tuning), tuning);
  }
}

function setStatus(text, isError = false) {
  const el = $("status");
  el.textContent = text;
  el.className = isError ? "error" : "";
}

async function handleFile(file) {
  if (!file) return;
  raw = null;
  transpose = 0;
  currentName = file.name.replace(/\.[^.]+$/, "") || "song";
  $("controls").hidden = true;
  $("chordsBox").hidden = true;
  $("tabBox").hidden = true;

  const player = $("player");
  player.src = URL.createObjectURL(file);
  player.hidden = false;

  $("bar").value = 0;
  $("bar").hidden = false;
  setStatus("Analysing the audio. Longer songs can take a few minutes.");

  try {
    raw = await runModel(file);
    setStatus(`Done: ${file.name}`);
    $("controls").hidden = false;
    render();
  } catch (err) {
    console.error(err);
    setStatus("Couldn't read that file. Try an MP3 or WAV.", true);
  } finally {
    $("bar").hidden = true;
  }
}

$("file").addEventListener("change", (e) => handleFile(e.target.files[0]));

const drop = $("drop");
drop.addEventListener("dragover", (e) => {
  e.preventDefault();
  drop.classList.add("over");
});
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  drop.classList.remove("over");
  handleFile(e.dataTransfer.files[0]);
});

for (const id of ["instrument", "mode", "sens", "win"]) {
  $(id).addEventListener("input", render);
}

function shift(by) {
  transpose = Math.max(-12, Math.min(12, transpose + by));
  render();
}
$("tDown").addEventListener("click", () => shift(-1));
$("tUp").addEventListener("click", () => shift(1));
$("tReset").addEventListener("click", () => shift(-transpose));
$("midi").addEventListener("click", downloadMidi);
