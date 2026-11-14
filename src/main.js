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
  setStatus("Reading the chords from the audio…");
  await new Promise((r) => setTimeout(r, 30)); // lets the page repaint first
  const chroma = computeChroma(audio);
  return { frames, onsets, chroma };
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

// Chooses a string and fret for every note.
// Notes that start in the same tab column sound together, so each needs its own string and the whole
// shape has to fit under one hand. Then the best run of shapes through the whole song is picked,
// so the hand moves as little as possible.
const FINGERINGS_KEPT = 14; // best shapes kept per column

function toTab(notes, tuning) {
  const strings = tuning.open.length;
  const byColumn = new Map();
  for (const n of notes) {
    const col = Math.floor(n.startTimeSeconds / TAB_STEP);
    if (!byColumn.has(col)) byColumn.set(col, []);
    byColumn.get(col).push(n);
  }

  // 1. For each column, list the playable shapes, best first.
  const steps = [];
  for (const col of [...byColumn.keys()].sort((a, b) => a - b)) {
    const seen = new Set();
    let group = byColumn
      .get(col)
      .sort((a, b) => (b.amplitude || 0) - (a.amplitude || 0))
      .filter((n) => !seen.has(n.pitchMidi) && seen.add(n.pitchMidi))
      .map((note) => ({
        note,
        options: tuning.open
          .map((open, string) => ({ string, fret: note.pitchMidi - open }))
          .filter((o) => o.fret >= 0 && o.fret <= tuning.maxFret),
      }))
      .filter((item) => item.options.length) // notes this instrument can't reach are left out
      .slice(0, strings);
    let candidates = [];
    while (group.length) {
      candidates = fingerings(group, tuning.open);
      if (candidates.length) break;
      group = group.slice(0, -1); // they won't fit on separate strings, so drop the quietest note
    }
    if (candidates.length) steps.push({ col, group, candidates });
  }
  if (!steps.length) return [];

  // 2. Pick one shape per column so that the total cost, including hand movement, is lowest.
  const total = steps.map((st) => st.candidates.map((c) => c.cost));
  const back = steps.map((st) => new Array(st.candidates.length).fill(0));
  for (let t = 1; t < steps.length; t++) {
    const gap = (steps[t].col - steps[t - 1].col) * TAB_STEP;
    steps[t].candidates.forEach((cur, j) => {
      let bestCost = Infinity;
      let bestI = 0;
      steps[t - 1].candidates.forEach((prev, i) => {
        const c = total[t - 1][i] + transitionCost(prev, cur, gap);
        if (c < bestCost) {
          bestCost = c;
          bestI = i;
        }
      });
      total[t][j] += bestCost;
      back[t][j] = bestI;
    });
  }
  let pick = 0;
  const last = steps.length - 1;
  total[last].forEach((c, j) => {
    if (c < total[last][pick]) pick = j;
  });

  const out = [];
  for (let t = last; t >= 0; t--) {
    const shape = steps[t].candidates[pick];
    steps[t].group.forEach((item, i) => {
      out.push({ time: item.note.startTimeSeconds, string: shape.picks[i].string, fret: shape.picks[i].fret });
    });
    pick = back[t][pick];
  }
  return out.sort((a, b) => a.time - b.time || a.string - b.string);
}

// Every way to put the notes on different strings, cheapest first.
function fingerings(group, open) {
  const picks = [];
  const used = new Set();
  const found = [];
  const visit = (i) => {
    if (i === group.length) {
      found.push(scoreFingering(group, picks, open));
      return;
    }
    for (const o of group[i].options) {
      if (used.has(o.string)) continue;
      used.add(o.string);
      picks.push(o);
      visit(i + 1);
      picks.pop();
      used.delete(o.string);
    }
  };
  visit(0);
  found.sort((a, b) => a.cost - b.cost);
  return found.slice(0, FINGERINGS_KEPT);
}

function scoreFingering(group, picks, open) {
  const fretted = picks.filter((p) => p.fret > 0).map((p) => p.fret);
  const span = fretted.length > 1 ? Math.max(...fretted) - Math.min(...fretted) : 0;
  let cost = Math.max(0, span - 3) * 1.2 + (span > 5 ? 2 : 0); // a hand covers about four frets
  for (const p of picks) {
    if (p.fret === 0) cost -= 0.1; // open strings are easy
    else cost += p.fret * 0.03 + Math.max(0, p.fret - 12) * 0.15; // low frets are easier than very high ones
  }
  // a higher note shouldn't sit on a lower-sounding string than a lower note
  for (let a = 0; a < picks.length; a++) {
    for (let b = a + 1; b < picks.length; b++) {
      const pitchDiff = group[a].note.pitchMidi - group[b].note.pitchMidi;
      const stringDiff = open[picks[a].string] - open[picks[b].string];
      if (pitchDiff * stringDiff < 0) cost += 2;
    }
  }
  return {
    picks: picks.map((p) => ({ ...p })),
    cost,
    hand: fretted.length ? fretted.reduce((sum, f) => sum + f, 0) / fretted.length : null,
    at: new Map(group.map((g, i) => [g.note.pitchMidi, `${picks[i].string}:${picks[i].fret}`])),
  };
}

// Cost of moving from one shape to the next: moving the hand, and playing the same note in a different place.
function transitionCost(a, b, gapSeconds) {
  let cost = 0;
  if (a.hand !== null && b.hand !== null) {
    cost += Math.abs(a.hand - b.hand) * 0.35 * (gapSeconds > 1 ? 0.4 : 1); // after a rest the hand can reposition
  }
  for (const [pitch, place] of b.at) {
    const before = a.at.get(pitch);
    if (before !== undefined && before !== place) cost += 0.8;
  }
  return cost;
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
      const complexity = shape.length > 3 ? total * 0.3 : 0;
      // Chords built only from the song's key get a head start. Others can still win, but the notes
      // have to be clearly present, so real borrowed chords are not blocked outright.
      const outOfKey = scale && !shape.every((i) => scale.has((root + i) % 12)) ? total * 0.25 : 0;
      const score = inChord - (total - inChord) * 0.5 - complexity - outOfKey;
      if (score > best.score) best = { score, name: NAMES[root] + suffix };
    }
  }
  // If nothing fits well (a quiet or sparse moment), name no chord and let the previous one carry on.
  return best.score >= total * 0.4 ? best.name : null;
}

// Krumhansl-Schmuckler key finding: compares how much each note is used against typical
// major and minor key profiles and picks the best of the 24 keys.
const MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88];
const MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.6, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17];

function estimateKey(notes) {
  if (!notes.length) return null;
  const w = new Array(12).fill(0);
  for (const n of notes) w[((n.pitchMidi % 12) + 12) % 12] += n.durationSeconds * n.amplitude;
  return keyFromWeights(w);
}

function keyFromWeights(w) {
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
    tonic: best.tonic,
    minor: best.minor,
  };
}

// ---------- chords from the audio itself ----------
// Instead of listening for single notes, this measures how much of each of the 12 note names
// (C, C#, D ...) is present in the sound over time, then matches that to chord shapes.

function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wr = Math.cos(ang);
    const wi = Math.sin(ang);
    const half = len / 2;
    for (let i = 0; i < n; i += len) {
      let cr = 1;
      let ci = 0;
      for (let k = 0; k < half; k++) {
        const a = i + k;
        const b = a + half;
        const vr = re[b] * cr - im[b] * ci;
        const vi = re[b] * ci + im[b] * cr;
        re[b] = re[a] - vr;
        im[b] = im[a] - vi;
        re[a] += vr;
        im[a] += vi;
        const nr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = nr;
      }
    }
  }
}

function computeChroma(audio, sr = 22050) {
  const N = 8192;
  const hop = 4096;
  const half = N / 2;
  const hann = new Float64Array(N);
  for (let i = 0; i < N; i++) hann[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (N - 1));

  // Which note name each frequency bin belongs to, and which bins count as "bass".
  const pcOfBin = new Int8Array(half).fill(-1);
  const isBass = new Uint8Array(half);
  for (let k = 1; k < half; k++) {
    const f = (k * sr) / N;
    if (f < 65 || f > 1800) continue;
    pcOfBin[k] = (((Math.round(69 + 12 * Math.log2(f / 440)) % 12) + 12) % 12);
    isBass[k] = f <= 260 ? 1 : 0;
  }

  const frames = [];
  const bass = [];
  const energy = [];
  const re = new Float64Array(N);
  const im = new Float64Array(N);
  for (let start = 0; start + N <= audio.length; start += hop) {
    for (let i = 0; i < N; i++) {
      re[i] = audio[start + i] * hann[i];
      im[i] = 0;
    }
    fft(re, im);
    const c = new Float32Array(12);
    const b = new Float32Array(12);
    let e = 0;
    for (let k = 1; k < half; k++) {
      const pc = pcOfBin[k];
      if (pc < 0) continue;
      const m = Math.pow(Math.hypot(re[k], im[k]), 0.6); // compress so loud notes don't drown the rest
      c[pc] += m;
      if (isBass[k]) b[pc] += m;
      e += m;
    }
    frames.push(c);
    bass.push(b);
    energy.push(e);
  }
  return { frames, bass, energy, hop: hop / sr, offset: N / 2 / sr };
}

const CHANGE_COST = 0.15; // how much a chord change has to be worth, so the result doesn't flicker

function chordsFromAudio(chroma, winSec) {
  const shift = (v) => Array.from(v, (_, i) => v[(((i - transpose) % 12) + 12) % 12]);
  const withSevenths = $("sevenths").checked;
  const useKey = $("inKey").checked;
  const argmax = (v) => v.reduce((best, x, i) => (x > v[best] ? i : best), 0);

  // average the chroma inside each chord window
  const wins = [];
  const total = chroma.frames.length * chroma.hop;
  for (let t0 = 0; t0 < total; t0 += winSec) {
    const from = Math.max(0, Math.ceil((t0 - chroma.offset) / chroma.hop));
    const to = Math.min(chroma.frames.length, Math.ceil((t0 + winSec - chroma.offset) / chroma.hop));
    if (to <= from) continue;
    const c = new Float64Array(12);
    const b = new Float64Array(12);
    let e = 0;
    for (let i = from; i < to; i++) {
      for (let k = 0; k < 12; k++) {
        c[k] += chroma.frames[i][k];
        b[k] += chroma.bass[i][k];
      }
      e += chroma.energy[i];
    }
    // remove the flat background (drums, noise) by subtracting the median of the 12 values
    const sorted = Array.from(c).sort((x, y) => x - y);
    const floor = (sorted[5] + sorted[6]) / 2;
    const cleaned = c.map((v) => Math.max(0, v - floor));
    wins.push({ time: t0, c: shift(cleaned), b: shift(b), energy: e / (to - from) });
  }

  // drop near-silent windows
  const energies = wins.map((w) => w.energy).sort((x, y) => x - y);
  const median = energies.length ? energies[Math.floor(energies.length / 2)] : 0;
  const active = wins.filter((w) => w.energy >= median * 0.2 && w.c.some((v) => v > 0));
  if (!active.length) return { chords: [], key: null };

  // song key from the whole recording
  const sum = new Array(12).fill(0);
  for (const w of active) for (let k = 0; k < 12; k++) sum[k] += w.c[k];
  const key = keyFromWeights(sum);
  // Chords that usually belong in a song in this key, written as (steps above the tonic + M or m).
  // This includes the common borrowed ones (flat seven, minor four, minor five in a major key),
  // so real chords such as a minor v are not pushed out just because they leave the scale.
  const allowed = new Set(
    key.minor
      ? ["0m", "3M", "5m", "7m", "7M", "8M", "10M"]
      : ["0M", "2m", "4m", "5M", "7M", "9m", "10M", "5m", "7m", "3M", "8M"]
  );

  // every chord we are willing to name
  const states = [];
  for (let root = 0; root < 12; root++) {
    for (const [suffix, shape] of Object.entries(SHAPES)) {
      if (shape.length > 3 && !withSevenths) continue;
      const tones = shape.map((i) => (root + i) % 12);
      states.push({
        name: NAMES[root] + suffix,
        root,
        tones,
        norm: Math.sqrt(tones.length),
        inKey: !useKey || allowed.has(`${(root - key.tonic + 12) % 12}${shape[1] === 3 ? "m" : "M"}`),
        extra: shape.length > 3,
      });
    }
  }

  const emit = (w, st) => {
    const cn = Math.hypot(...w.c) || 1;
    let dot = 0;
    for (const t of st.tones) dot += w.c[t];
    let score = dot / (cn * st.norm); // how closely the sound matches this chord shape
    if (st.root === argmax(w.b)) score += 0.06; // the lowest notes agree on the root
    if (!st.inKey) score -= 0.1;
    if (st.extra) score -= 0.06;
    return score;
  };

  // pick the best path through time, so one odd window doesn't flip the chord
  const T = active.length;
  const S = states.length;
  const back = Array.from({ length: T }, () => new Int16Array(S));
  let prev = Float64Array.from(states, (st) => emit(active[0], st));
  for (let t = 1; t < T; t++) {
    let bestPrev = 0;
    for (let s = 1; s < S; s++) if (prev[s] > prev[bestPrev]) bestPrev = s;
    const cur = new Float64Array(S);
    for (let s = 0; s < S; s++) {
      const stay = prev[s];
      const jump = prev[bestPrev] - CHANGE_COST;
      const from = stay >= jump ? s : bestPrev;
      cur[s] = Math.max(stay, jump) + emit(active[t], states[s]);
      back[t][s] = from;
    }
    prev = cur;
  }
  let s = 0;
  for (let i = 1; i < S; i++) if (prev[i] > prev[s]) s = i;
  const path = new Array(T);
  for (let t = T - 1; t >= 0; t--) {
    path[t] = s;
    s = back[t][s];
  }

  const chords = [];
  active.forEach((w, t) => {
    const name = states[path[t]].name;
    if (name !== chords[chords.length - 1]?.chord) chords.push({ time: w.time, chord: name });
  });
  return { chords, key: useKey ? key : null };
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
    let chords;
    let key;
    if ($("chordMethod").value === "audio" && raw.chroma) {
      ({ chords, key } = chordsFromAudio(raw.chroma, winSec));
    } else {
      const chordNotes = getNotes("all"); // chords always come from the whole mix, not the picked part
      key = $("inKey").checked ? estimateKey(chordNotes) : null;
      chords = toChords(chordNotes, winSec, key ? key.scale : null);
    }
    $("keyInfo").textContent = key ? `Detected key: ${key.label}. Chords from this key are favored.` : "";
    renderChords(chords);
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

for (const id of ["instrument", "mode", "sens", "win", "minLen", "merge", "useRange", "sevenths", "inKey", "part", "chordMethod"]) {
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
