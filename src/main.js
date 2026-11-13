import * as tf from "@tensorflow/tfjs";
import { BasicPitch, noteFramesToTime, outputToNotesPoly } from "@spotify/basic-pitch";

const $ = (id) => document.getElementById(id);

// Open-string MIDI notes, listed in the order they are drawn from bottom line to top line.
const TUNINGS = {
  guitar: { labels: ["E", "A", "D", "G", "B", "e"], open: [40, 45, 50, 55, 59, 64], maxFret: 17, program: 25, range: [40, 88] },
  bass: { labels: ["E", "A", "D", "G"], open: [28, 33, 38, 43], maxFret: 17, program: 33, range: [28, 67] },
  ukulele: { labels: ["G", "C", "E", "A"], open: [67, 60, 64, 69], maxFret: 12, program: 24, range: [60, 84] },
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
let currentFile = null; // kept so "Analyze again" can reuse it
let analysisOffset = 0; // seconds into the song where the analysed section starts

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
  const full = await decodeMono(file);
  const startSec = Math.max(0, Number($("startAt").value) || 0);
  const lenSec = Number($("lenSel").value); // 0 means the whole song
  const from = Math.floor(startSec * 22050);
  if (from >= full.length - 22050) throw new Error("range");
  const to = lenSec > 0 ? Math.min(full.length, from + lenSec * 22050) : full.length;
  const audio = full.subarray(from, to);
  analysisOffset = from / 22050;
  await tf.ready();
  const gpu = tf.getBackend() === "webgl";
  console.log("TensorFlow backend:", tf.getBackend());
  setStatus(
    gpu
      ? "Analysing the audio using your graphics chip."
      : "Analysing the audio on the CPU, which is slower. Enabling hardware acceleration in your browser can help."
  );

  // The pitch contours are about three times bigger than everything else and the app never uses them,
  // so they are not collected at all.
  const frames = [];
  const onsets = [];
  const bp = new BasicPitch(import.meta.env.BASE_URL + "model/model.json");
  await bp.evaluateModel(
    audio,
    (f, o) => {
      frames.push(...f);
      onsets.push(...o);
    },
    (p) => {
      $("bar").value = Math.round(p * 100);
    }
  );
  return { frames, onsets };
}

let baseNotes = null; // detected notes for the current sensitivity, reused by every other control
let baseSens = null;

function getNotes(part = $("part").value) {
  const s = Number($("sens").value);
  if (!baseNotes || baseSens !== s) {
    baseNotes = detectNotes(s);
    baseSens = s;
  }
  return cleanNotes(baseNotes, part).map((n) => ({ ...n, pitchMidi: n.pitchMidi + transpose }));
}

function detectNotes(s) {
  const onset = 0.8 - 0.06 * s; // sensitivity 5 -> 0.5, 10 -> 0.2
  const frame = onset * 0.6;
  // Pitch bends are not used by this app, and the "melodia" gap-filling pass (switched off by the
  // final false below) is the slowest part on long songs, so both are skipped.
  const events = outputToNotesPoly(raw.frames, raw.onsets, onset, frame, 5, true, null, null, false);
  const notes = noteFramesToTime(events).sort((a, b) => a.startTimeSeconds - b.startTimeSeconds);
  return notes;
}

// Drops notes outside the instrument's range, optionally joins back-to-back repeats
// of the same pitch, then removes notes shorter than the minimum length.
function cleanNotes(notes, part = "all") {
  let [lo, hi] = TUNINGS[$("instrument").value].range;
  if ($("useRange").checked) {
    lo = Number($("lowNote").value);
    hi = Number($("highNote").value);
  }
  const minLen = Number($("minLen").value);
  const inRange = notes.filter((n) => n.pitchMidi >= lo && n.pitchMidi <= hi);

  let result = inRange;
  if ($("merge").checked) {
    const lastByPitch = new Map();
    result = [];
    for (const n of inRange) {
      const prev = lastByPitch.get(n.pitchMidi);
      const prevEnd = prev ? prev.startTimeSeconds + prev.durationSeconds : 0;
      if (prev && n.startTimeSeconds - prevEnd < 0.08) {
        const end = Math.max(prevEnd, n.startTimeSeconds + n.durationSeconds);
        prev.durationSeconds = end - prev.startTimeSeconds;
        prev.amplitude = Math.max(prev.amplitude, n.amplitude);
      } else {
        const copy = { ...n };
        result.push(copy);
        lastByPitch.set(n.pitchMidi, copy);
      }
    }
  }
  return pickPart(result.filter((n) => n.durationSeconds >= minLen), part);
}

// ---------- pulling out a part ----------
// This sorts notes by pitch and by how many sound together. It cannot tell instruments apart,
// so a voice and a guitar playing the same pitches look identical to it.

const PART_RANGE = { melody: [52, 96], bass: [24, 60], chords: [40, 76] }; // MIDI notes
const PART_INFO = {
  all: "",
  melody: "Melody shows the highest note at each moment, from about E3 up. It can't tell a voice from a lead guitar or a synth.",
  bass: "Bass line shows the lowest note at each moment, up to about C4.",
  chords: "Chords and rhythm shows moments where three or more notes sound together.",
};

function pickPart(notes, part) {
  if (part === "all" || !notes.length) return notes;
  const [lo, hi] = PART_RANGE[part];
  const pool = notes.filter((n) => n.pitchMidi >= lo && n.pitchMidi <= hi);
  if (!pool.length) return pool;

  const FRAME = 0.05; // seconds
  const end = pool.reduce((m, n) => Math.max(m, n.startTimeSeconds + n.durationSeconds), 0);
  const size = Math.ceil(end / FRAME) + 1;
  const top = new Array(size).fill(-Infinity);
  const bottom = new Array(size).fill(Infinity);
  const count = new Array(size).fill(0);
  const span = (n) => {
    const a = Math.floor(n.startTimeSeconds / FRAME);
    const b = Math.max(a, Math.ceil((n.startTimeSeconds + n.durationSeconds) / FRAME) - 1);
    return [a, b];
  };

  for (const n of pool) {
    const [a, b] = span(n);
    for (let f = a; f <= b; f++) {
      if (n.pitchMidi > top[f]) top[f] = n.pitchMidi;
      if (n.pitchMidi < bottom[f]) bottom[f] = n.pitchMidi;
      count[f]++;
    }
  }

  // Keep a note if it is the top (or bottom) note, or part of a group of 3+, for at least half its length.
  return pool.filter((n) => {
    const [a, b] = span(n);
    let hits = 0;
    for (let f = a; f <= b; f++) {
      if (part === "melody" && n.pitchMidi === top[f]) hits++;
      else if (part === "bass" && n.pitchMidi === bottom[f]) hits++;
      else if (part === "chords" && count[f] >= 3) hits++;
    }
    return hits / (b - a + 1) >= 0.5;
  });
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

// ---------- tab editing ----------

const TAB_STEP = 0.25; // seconds per tab column
const PER_ROW = 20; // columns per row of tab
const editStore = new Map(); // one set of edits per instrument and part
const getEdits = () => {
  const key = `${$("instrument").value}:${$("part").value}`;
  if (!editStore.has(key)) editStore.set(key, new Map());
  return editStore.get(key);
};
const hasEdits = () => getEdits().size > 0;
let selected = null; // { col, string } of the spot being edited
let lastTab = null; // { grid, tuning, cols } for the tab currently on screen
let digitBuffer = "";
let digitTime = 0;

const cellText = (fret) => (fret === null ? "" : String(fret)).padEnd(3, "-");

// Builds the grid of frets (null = empty) from the detected notes, then puts the user's edits on top.
function buildGrid(positions, tuning) {
  const strings = tuning.open.length;
  const edits = getEdits();
  let cols = positions.reduce((m, p) => Math.max(m, Math.floor(p.time / TAB_STEP) + 1), 0);
  for (const key of edits.keys()) cols = Math.max(cols, Number(key.split(":")[0]) + 1);
  cols = Math.max(1, Math.ceil(cols / PER_ROW)) * PER_ROW; // pad so the last row has empty spots to click
  const grid = Array.from({ length: strings }, () => Array(cols).fill(null));
  for (const p of positions) {
    const col = Math.floor(p.time / TAB_STEP);
    if (grid[p.string][col] === null) grid[p.string][col] = p.fret;
  }
  for (const [key, fret] of edits) {
    const [col, s] = key.split(":").map(Number);
    if (s < strings) grid[s][col] = fret; // a fret of null means the note was removed
  }
  return { grid, cols };
}

function renderTab(positions, tuning) {
  const box = $("tab");
  box.replaceChildren();
  if (!positions.length && !hasEdits()) {
    lastTab = null;
    box.textContent = "No notes found in this instrument's range. Try raising the sensitivity.";
    updateEditBar();
    return;
  }
  const { grid, cols } = buildGrid(positions, tuning);
  lastTab = { grid, tuning, cols };
  if (selected && (selected.string >= tuning.open.length || selected.col >= cols)) selected = null;

  const edits = getEdits();
  for (let start = 0; start < cols; start += PER_ROW) {
    box.append(`${fmtTime(start * TAB_STEP + analysisOffset)}\n`);
    for (let s = tuning.open.length - 1; s >= 0; s--) {
      box.append(`${tuning.labels[s]}|`);
      for (let c = start; c < start + PER_ROW; c++) {
        const cell = document.createElement("span");
        cell.className = "cell";
        cell.dataset.col = c;
        cell.dataset.string = s;
        cell.textContent = cellText(grid[s][c]);
        if (edits.has(`${c}:${s}`)) cell.classList.add("edited");
        if (selected && selected.col === c && selected.string === s) cell.classList.add("selected");
        box.append(cell);
      }
      box.append("|\n");
    }
    box.append("\n");
  }
  updateEditBar();
}

function cellEl(col, s) {
  return $("tab").querySelector(`.cell[data-col="${col}"][data-string="${s}"]`);
}

function selectCell(col, s, scroll = false) {
  if (!lastTab) return;
  if (selected) cellEl(selected.col, selected.string)?.classList.remove("selected");
  selected = { col, string: s };
  const el = cellEl(col, s);
  el?.classList.add("selected");
  if (scroll) el?.scrollIntoView({ block: "nearest", inline: "nearest" });
  updateEditBar();
}

// fret is a number, or null to remove the note
function setFret(fret) {
  if (!selected || !lastTab) return;
  const { col, string } = selected;
  const value = fret === null ? null : Math.max(0, Math.min(24, Math.round(fret)));
  getEdits().set(`${col}:${string}`, value);
  lastTab.grid[string][col] = value;
  const el = cellEl(col, string);
  if (el) {
    el.textContent = cellText(value);
    el.classList.add("edited");
  }
  updateEditBar();
}

function updateEditBar() {
  const active = Boolean(selected && lastTab);
  for (const id of ["fretIn", "fretMinus", "fretPlus", "fretClear"]) $(id).hidden = !active;
  $("resetEdits").hidden = !hasEdits();
  const info = $("selInfo");
  if (!active) {
    info.textContent = "Click a spot in the tab to edit it.";
    return;
  }
  const { col, string } = selected;
  const fret = lastTab.grid[string][col];
  info.textContent =
    `${lastTab.tuning.labels[string]} string at ${fmtTime(col * TAB_STEP + analysisOffset)}: ` +
    (fret === null ? "empty" : `fret ${fret}`);
  $("fretIn").value = fret === null ? "" : fret;
}

function tabAsText() {
  if (!lastTab) return "";
  const { grid, tuning, cols } = lastTab;
  const rows = [];
  for (let start = 0; start < cols; start += PER_ROW) {
    const lines = [fmtTime(start * TAB_STEP + analysisOffset)];
    for (let s = tuning.open.length - 1; s >= 0; s--) {
      const cells = grid[s].slice(start, start + PER_ROW).map((f) => cellText(f));
      lines.push(`${tuning.labels[s]}|${cells.join("")}|`);
    }
    rows.push(lines.join("\n"));
  }
  return rows.join("\n\n");
}

function downloadTab() {
  const text = tabAsText();
  if (!text) return;
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type: "text/plain" }));
  a.download = `${currentName}-tab.txt`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

// When the tab has been edited, the MIDI file is built from the tab itself so the two agree.
// Timing is rounded to the tab's columns.
function notesFromTab() {
  const tuning = TUNINGS[$("instrument").value];
  const { grid, cols } = buildGrid(toTab(getNotes(), tuning), tuning);
  const out = [];
  grid.forEach((line, s) => {
    for (let c = 0; c < cols; c++) {
      if (line[c] === null) continue;
      let next = c + 1;
      while (next < cols && line[next] === null) next++;
      out.push({
        startTimeSeconds: c * TAB_STEP,
        durationSeconds: Math.min(1, (next - c) * TAB_STEP),
        pitchMidi: tuning.open[s] + line[c],
        amplitude: 0.7,
      });
    }
  });
  return out.sort((a, b) => a.startTimeSeconds - b.startTimeSeconds);
}

// ---------- notes -> chords ----------

function detectChord(windowNotes, scale = null) {
  const weights = new Array(12).fill(0);
  for (const n of windowNotes) weights[n.pitchMidi % 12] += n.durationSeconds * n.amplitude;
  const total = weights.reduce((a, b) => a + b, 0);
  const withSevenths = $("sevenths").checked;
  if (weights.filter((v) => v > 0).length < 2) return null; // one note is not enough to name a chord
  let best = { score: 0, name: null };
  for (let root = 0; root < 12; root++) {
    for (const [suffix, shape] of Object.entries(SHAPES)) {
      if (shape.length > 3 && !withSevenths) continue;
      const inChord = shape.reduce((sum, i) => sum + weights[(root + i) % 12], 0);
      // A 7th chord always scores at least as high as its plain triad, because it only adds one more
      // note and any stray sound in the mix lands on that note. The penalty means the extra note has
      // to be clearly present before the 7th chord wins.
      const complexity = shape.length > 3 ? total * 0.2 : 0;
      // Chords built only from the song's key get a head start. Others can still win, but the notes
      // have to be clearly present, so real borrowed chords are not blocked outright.
      const outOfKey = scale && !shape.every((i) => scale.has((root + i) % 12)) ? total * 0.25 : 0;
      const score = inChord - (total - inChord) * 0.5 - complexity - outOfKey;
      if (score > best.score) best = { score, name: NAMES[root] + suffix };
    }
  }
  return best.name;
}

// Krumhansl-Schmuckler key finding: compares how much each note is used against typical
// major and minor key profiles and picks the best of the 24 keys.
const MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88];
const MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.6, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17];

function estimateKey(notes) {
  if (!notes.length) return null;
  const w = new Array(12).fill(0);
  for (const n of notes) w[((n.pitchMidi % 12) + 12) % 12] += n.durationSeconds * n.amplitude;

  const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
  const mw = mean(w);
  let best = null;
  for (const minor of [false, true]) {
    const profile = minor ? MINOR_PROFILE : MAJOR_PROFILE;
    for (let tonic = 0; tonic < 12; tonic++) {
      const ys = w.map((_, pc) => profile[(pc - tonic + 12) % 12]);
      const my = mean(ys);
      let num = 0, dx = 0, dy = 0;
      for (let i = 0; i < 12; i++) {
        num += (w[i] - mw) * (ys[i] - my);
        dx += (w[i] - mw) ** 2;
        dy += (ys[i] - my) ** 2;
      }
      const score = num / Math.sqrt(dx * dy || 1);
      if (!best || score > best.score) best = { score, tonic, minor };
    }
  }
  // A minor key uses the same seven notes as the major key three semitones above it.
  const scaleRoot = best.minor ? (best.tonic + 3) % 12 : best.tonic;
  return {
    scale: new Set([0, 2, 4, 5, 7, 9, 11].map((i) => (scaleRoot + i) % 12)),
    label: `${NAMES[best.tonic]} ${best.minor ? "minor" : "major"}`,
  };
}

function toChords(notes, windowSec, scale = null) {
  if (!notes.length) return [];
  const end = notes.reduce((m, n) => Math.max(m, n.startTimeSeconds), 0);
  const out = [];
  for (let t = 0; t <= end; t += windowSec) {
    const inWindow = notes.filter((n) => n.startTimeSeconds >= t && n.startTimeSeconds < t + windowSec);
    const chord = inWindow.length ? detectChord(inWindow, scale) : null;
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
    time.textContent = fmtTime(c.time + analysisOffset);
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
  const notes = hasEdits() ? notesFromTab() : getNotes();
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

const noteName = (m) => `${NAMES[m % 12]}${Math.floor(m / 12) - 1}`;

function render() {
  if (!raw) return;
  const useRange = $("useRange").checked;
  $("rangeBox").hidden = !useRange;
  if (!useRange) {
    // keep the sliders ready at this instrument's normal range for when they are switched on
    const [lo, hi] = TUNINGS[$("instrument").value].range;
    $("lowNote").value = lo;
    $("highNote").value = hi;
  }
  $("lowOut").textContent = noteName(Number($("lowNote").value));
  $("highOut").textContent = noteName(Number($("highNote").value));
  const notes = getNotes();
  const mode = $("mode").value;
  const winSec = Number($("win").value);
  $("winOut").textContent = `${winSec}s`;
  $("lenOut").textContent = `${Number($("minLen").value).toFixed(2)}s`;
  $("tOut").textContent = transpose > 0 ? `+${transpose}` : String(transpose);

  $("chordsBox").hidden = mode === "tabs";
  $("tabBox").hidden = mode === "chords";

  if (mode !== "tabs") {
    const chordNotes = getNotes("all"); // chords always come from the whole mix, not the picked part
    const key = $("inKey").checked ? estimateKey(chordNotes) : null;
    $("keyInfo").textContent = key ? `Detected key: ${key.label}. Chords from this key are favored.` : "";
    renderChords(toChords(chordNotes, winSec, key ? key.scale : null));
  }
  if (mode !== "chords") {
    $("partInfo").textContent = PART_INFO[$("part").value];
    const tuning = TUNINGS[$("instrument").value];
    renderTab(toTab(notes, tuning), tuning);
  }
}

function setStatus(text, isError = false) {
  const el = $("status");
  el.textContent = text;
  el.className = isError ? "error" : "";
}

async function handleFile(file) {
  if (!file) return;
  currentFile = file;
  editStore.clear();
  selected = null;
  raw = null;
  baseNotes = null;
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
    setStatus("Finding the notes…");
    await new Promise((r) => setTimeout(r, 30)); // lets the page repaint before the heavy step
    $("controls").hidden = false;
    render();
    setStatus(`Done: ${file.name}`);
  } catch (err) {
    console.error(err);
    setStatus(
      err.message === "range"
        ? "That start time is past the end of the song. Lower the start time and try again."
        : "Couldn't read that file. Try an MP3 or WAV.",
      true
    );
  } finally {
    $("bar").hidden = true;
  }
}

$("file").addEventListener("change", (e) => handleFile(e.target.files[0]));
$("reanalyze").addEventListener("click", () => {
  if (currentFile) handleFile(currentFile);
});

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

let renderTimer = 0;
function scheduleRender(delay = 60, busy = false) {
  clearTimeout(renderTimer);
  renderTimer = setTimeout(async () => {
    if (busy) {
      setStatus("Updating…");
      await new Promise((r) => setTimeout(r, 30)); // lets the page repaint first
    }
    render();
    if (busy && currentFile) setStatus(`Done: ${currentFile.name}`);
  }, delay);
}

for (const id of ["instrument", "mode", "sens", "win", "minLen", "merge", "useRange", "sevenths", "inKey", "part"]) {
  // sensitivity re-detects every note, so wait until the slider stops moving
  $(id).addEventListener("input", () => scheduleRender(id === "sens" ? 350 : 60, id === "sens"));
}

function shift(by) {
  transpose = Math.max(-12, Math.min(12, transpose + by));
  render();
}
$("tDown").addEventListener("click", () => shift(-1));
$("tUp").addEventListener("click", () => shift(1));
$("tReset").addEventListener("click", () => shift(-transpose));
$("midi").addEventListener("click", downloadMidi);

$("lowNote").addEventListener("input", () => {
  if (Number($("lowNote").value) > Number($("highNote").value)) $("highNote").value = $("lowNote").value;
  scheduleRender();
});
$("highNote").addEventListener("input", () => {
  if (Number($("highNote").value) < Number($("lowNote").value)) $("lowNote").value = $("highNote").value;
  scheduleRender();
});

// ---------- tab editing: events ----------

$("tab").addEventListener("click", (e) => {
  const cell = e.target.closest(".cell");
  if (!cell) return;
  selectCell(Number(cell.dataset.col), Number(cell.dataset.string));
  $("tab").focus({ preventScroll: true });
});

$("tab").addEventListener("keydown", (e) => {
  if (!selected || !lastTab) return;
  const { col, string } = selected;
  const strings = lastTab.tuning.open.length;
  const fret = lastTab.grid[string][col];
  const move = (c, s) => {
    e.preventDefault();
    selectCell(Math.max(0, Math.min(lastTab.cols - 1, c)), Math.max(0, Math.min(strings - 1, s)), true);
  };
  if (e.key === "ArrowLeft") move(col - 1, string);
  else if (e.key === "ArrowRight") move(col + 1, string);
  else if (e.key === "ArrowUp") move(col, string + 1);
  else if (e.key === "ArrowDown") move(col, string - 1);
  else if (e.key === "Delete" || e.key === "Backspace") {
    e.preventDefault();
    setFret(null);
  } else if (e.key === "+" || e.key === "=") {
    e.preventDefault();
    setFret(fret === null ? 0 : fret + 1);
  } else if (e.key === "-") {
    e.preventDefault();
    if (fret !== null) setFret(fret - 1);
  } else if (/^[0-9]$/.test(e.key)) {
    e.preventDefault();
    const now = Date.now();
    digitBuffer = now - digitTime < 800 && digitBuffer.length < 2 ? digitBuffer + e.key : e.key;
    digitTime = now;
    setFret(Number(digitBuffer));
  }
});

$("fretIn").addEventListener("input", (e) => {
  if (e.target.value !== "") setFret(Number(e.target.value));
});
$("fretPlus").addEventListener("click", () => {
  if (!selected || !lastTab) return;
  const fret = lastTab.grid[selected.string][selected.col];
  setFret(fret === null ? 0 : fret + 1);
});
$("fretMinus").addEventListener("click", () => {
  if (!selected || !lastTab) return;
  const fret = lastTab.grid[selected.string][selected.col];
  if (fret !== null) setFret(fret - 1);
});
$("fretClear").addEventListener("click", () => setFret(null));
$("resetEdits").addEventListener("click", () => {
  getEdits().clear();
  render();
});
$("dlTab").addEventListener("click", downloadTab);

// Picking the bass line switches to the bass instrument, and leaving it switches back to guitar.
$("part").addEventListener("input", () => {
  if ($("part").value === "bass") $("instrument").value = "bass";
  else if ($("instrument").value === "bass") $("instrument").value = "guitar";
});
