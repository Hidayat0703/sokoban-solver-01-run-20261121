// Sokoban solver: push-optimal per-room A* with deadlock pruning.
//
// Strategy: a level is a hallway with 1-4 independent rooms off it (boxes never
// leave their room, and the level optimum P* is the sum of the rooms' optima).
// Each room is solved push-optimally on its own with A* over pushes (a state is
// the boxes plus the player's reachable region, named by its lowest square),
// then the room solutions are joined by the player's walks along the hallway.
// The heuristic is the cheapest assignment of boxes to goals by per-box
// push-distance (a lower bound: one push moves one box by one square), so the
// first goal state popped is push-optimal. Dead squares (no push path to any
// goal) and frozen 2x2 blocks prune dead states. All budgets are work caps,
// never the clock, so two runs write identical bytes.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Work budgets (state expansions), never the clock: deterministic on every run.
const ROOM_EXPANSIONS = 250_000;
const WHOLE_EXPANSIONS = 400_000;

interface Dir { dx: number; dy: number; step: string; push: string }
const DIRS: Dir[] = [
  { dx: 0, dy: -1, step: 'u', push: 'U' },
  { dx: 0, dy: 1, step: 'd', push: 'D' },
  { dx: -1, dy: 0, step: 'l', push: 'L' },
  { dx: 1, dy: 0, step: 'r', push: 'R' },
];

interface Level {
  w: number; h: number;
  wall: Uint8Array; // 1 = wall
  goals: number[]; // sorted
  boxes: number[]; // sorted
  player: number;
}

function parse(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  boxes.sort((a, b) => a - b);
  goals.sort((a, b) => a - b);
  return { w, h, wall, goals, boxes, player };
}

/**
 * Squares from which a box alone can still be pushed onto some goal, found by
 * pulling boxes back from the goals: a box on y can be pushed to x = y + d
 * when the player can stand on y - d, so y is live when x is live and both y
 * and y - d are open floor.
 */
function liveSquares(level: Level): Uint8Array {
  const { w, wall, goals } = level;
  const live = new Uint8Array(wall.length);
  const queue = [...goals];
  for (const g of goals) live[g] = 1;
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi]!;
    for (const dir of DIRS) {
      const d = dir.dy * w + dir.dx;
      const y = x - d;
      if (y < 0 || y >= wall.length) continue;
      if (!wall[y] && !wall[y - d] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

/**
 * Per-goal push distances: dist[g][s] is the fewest pushes moving one box
 * alone from s onto goal g (other boxes ignored), or -1 when unreachable.
 * Same pull-back walk as liveSquares, run once per goal, keeping distances.
 */
function pushDistances(level: Level): Int32Array[] {
  const { w, wall, goals } = level;
  return goals.map((g) => {
    const dist = new Int32Array(wall.length).fill(-1);
    dist[g] = 0;
    const queue = [g];
    for (let qi = 0; qi < queue.length; qi++) {
      const x = queue[qi]!;
      for (const dir of DIRS) {
        const d = dir.dy * w + dir.dx;
        const y = x - d;
        if (y < 0 || y >= wall.length) continue;
        if (!wall[y] && !wall[y - d] && dist[y] < 0) {
          dist[y] = dist[x]! + 1;
          queue.push(y);
        }
      }
    }
    return dist;
  });
}

/** All orderings of n goal indexes (n <= 4), in fixed lexicographic order. */
function permutations(n: number): number[][] {
  const out: number[][] = [];
  const cur: number[] = [];
  const used = new Array<boolean>(n).fill(false);
  const rec = (): void => {
    if (cur.length === n) { out.push([...cur]); return; }
    for (let i = 0; i < n; i++) {
      if (used[i]) continue;
      used[i] = true;
      cur.push(i);
      rec();
      cur.pop();
      used[i] = false;
    }
  };
  rec();
  return out;
}

/** Cheapest assignment of boxes to goals by push distance; Infinity if any box is stranded. */
function assignmentBound(boxes: number[], dists: Int32Array[], perms: number[][]): number {
  let best = Infinity;
  for (const p of perms) {
    let sum = 0;
    for (let i = 0; i < boxes.length; i++) {
      const d = dists[p[i]!]![boxes[i]!]!;
      if (d < 0) { sum = Infinity; break; }
      sum += d;
    }
    if (sum < best) best = sum;
  }
  return best;
}

interface Ctx {
  w: number;
  wall: Uint8Array;
  goalMask: Uint8Array;
  live: Uint8Array;
  dists: Int32Array[];
  perms: number[][];
  size: number;
  seen: Int32Array; // the expanded state's region (membership + lowest square)
  mark: Int32Array; // a child state's region, kept apart so it never clobbers `seen`
  stack: Int32Array;
  seenStamp: number;
  markStamp: number;
}

/**
 * The player's reachable region from `from` (walls and `occupied` in the way):
 * marks it in `arr` with a fresh stamp and returns its lowest square, the
 * region's canonical name.
 */
function regionLow(
  ctx: Ctx, arr: Int32Array, stamp: number, occupied: Uint8Array, from: number,
): number {
  const { w, wall, stack } = ctx;
  let low = from;
  let n = 0;
  stack[n++] = from;
  arr[from] = stamp;
  while (n > 0) {
    const p = stack[--n]!;
    if (p < low) low = p;
    for (const dir of DIRS) {
      const q = p + dir.dy * w + dir.dx;
      if (q < 0 || q >= wall.length) continue;
      if (!wall[q] && !occupied[q] && arr[q] !== stamp) {
        arr[q] = stamp;
        stack[n++] = q;
      }
    }
  }
  return low;
}

/** Whether the box just pushed to `t` sits in a frozen 2x2 block of walls and boxes with a box off its goal. */
function frozenBlock(ctx: Ctx, occupied: Uint8Array, t: number): boolean {
  const { w, wall, goalMask } = ctx;
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (a < 0 || d >= wall.length) continue;
    if (!(wall[a] || occupied[a]) || !(wall[b] || occupied[b]) ||
        !(wall[c] || occupied[c]) || !(wall[d] || occupied[d])) continue;
    if ((occupied[a] && !goalMask[a]) || (occupied[b] && !goalMask[b]) ||
        (occupied[c] && !goalMask[c]) || (occupied[d] && !goalMask[d])) return true;
  }
  return false;
}

/** Shortest walk from `from` to `to` around walls and boxes, in lower-case LURD (first direction first on ties). */
function walkTo(ctx: Ctx, occupied: Uint8Array, from: number, to: number): string | null {
  const { w, wall } = ctx;
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how = new Array<string>(wall.length);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi]!;
    if (p === to) break;
    for (const dir of DIRS) {
      const q = p + dir.dy * w + dir.dx;
      if (q < 0 || q >= wall.length) continue;
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = dir.step;
      queue.push(q);
    }
  }
  if (prev[to] === -1) return null;
  let out = '';
  for (let p = to; p !== from; p = prev[p]!) out = how[p]! + out;
  return out;
}

interface Solution { pushes: number; moves: string; player: number }

/**
 * Push-optimal A* for one (sub-)level: states are (sorted boxes, player's
 * region low); edges are single pushes; the heuristic is the assignment bound.
 * Returns null when the expansion cap is hit or no solution exists.
 */
function solvePush(level: Level, maxExpansions: number): Solution | null {
  const { w, wall, goals, boxes: startBoxes, player: startPlayer } = level;
  const size = wall.length;
  const goalMask = new Uint8Array(size);
  for (const g of goals) goalMask[g] = 1;
  const live = liveSquares(level);
  if (startBoxes.some((b) => !live[b])) return null;
  const dists = pushDistances(level);
  const perms = permutations(goals.length);
  const ctx: Ctx = {
    w, wall, goalMask, live, dists, perms, size,
    seen: new Int32Array(size), mark: new Int32Array(size),
    stack: new Int32Array(size), seenStamp: 0, markStamp: 0,
  };
  const solved = (bs: number[]): boolean => bs.every((b) => goalMask[b]);
  if (solved(startBoxes)) return { pushes: 0, moves: '', player: startPlayer };

  const occupied = new Uint8Array(size);
  const keyOf = (bs: number[], low: number): number => {
    let k = low;
    for (const b of bs) k = k * size + b;
    return k;
  };

  // State store: boxes, player square after the push that made it, parent,
  // the pushed box's square before the push, the push direction, and its key.
  const stBoxes: number[][] = [startBoxes];
  const stPlayer: number[] = [startPlayer];
  const stParent: number[] = [-1];
  const stPushBox: number[] = [-1];
  const stPushDir: number[] = [-1];
  const stG: number[] = [0];
  const stKey: number[] = [0];
  const best = new Map<number, number>(); // key -> state index with lowest g

  for (const b of startBoxes) occupied[b] = 1;
  const startLow = regionLow(ctx, ctx.seen, ++ctx.seenStamp, occupied, startPlayer);
  for (const b of startBoxes) occupied[b] = 0;
  stKey[0] = keyOf(startBoxes, startLow);
  best.set(stKey[0]!, 0);

  // Binary heap on (f, h, insertion sequence): fully deterministic ordering.
  type Entry = [number, number, number, number]; // f, h, seq, state index
  const heap: Entry[] = [];
  let seq = 0;
  const less = (a: Entry, b: Entry): boolean =>
    a[0] < b[0] || (a[0] === b[0] && (a[1] < b[1] || (a[1] === b[1] && a[2] < b[2])));
  const heapPush = (e: Entry): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i]!, heap[p]!)) break;
      [heap[i], heap[p]] = [heap[p]!, heap[i]!];
      i = p;
    }
  };
  const heapPop = (): Entry => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        [heap[i], heap[c]] = [heap[c]!, heap[i]!];
        i = c;
      }
    }
    return top;
  };

  const h0 = assignmentBound(startBoxes, dists, perms);
  if (h0 === Infinity) return null;
  heapPush([h0, h0, seq++, 0]);

  let expanded = 0;
  let found = -1;
  while (heap.length > 0 && expanded < maxExpansions) {
    const [, , , si] = heapPop();
    if (best.get(stKey[si]!) !== si) continue; // superseded by a cheaper path to the same key
    const boxes = stBoxes[si]!;
    if (solved(boxes)) { found = si; break; }
    expanded++;
    const g = stG[si]! + 1;
    for (const b of boxes) occupied[b] = 1;
    regionLow(ctx, ctx.seen, ++ctx.seenStamp, occupied, stPlayer[si]!);
    const here = ctx.seenStamp;
    for (let bi = 0; bi < boxes.length; bi++) {
      const b = boxes[bi]!;
      for (let di = 0; di < DIRS.length; di++) {
        const dir = DIRS[di]!;
        const d = dir.dy * w + dir.dx;
        const from = b - d; // where the player must stand to push
        const t = b + d; // where the box lands
        if (t < 0 || t >= size || from < 0 || from >= size) continue;
        if (ctx.seen[from] !== here) continue; // push square out of reach
        if (wall[t] || occupied[t] || !live[t]) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        const frozen = frozenBlock(ctx, occupied, t);
        let key = -1;
        let next: number[] | null = null;
        let h = Infinity;
        if (!frozen) {
          next = boxes.slice();
          next[bi] = t;
          next.sort((x, y) => x - y);
          h = assignmentBound(next, dists, perms);
          if (h !== Infinity) {
            const low = regionLow(ctx, ctx.mark, ++ctx.markStamp, occupied, b);
            key = keyOf(next, low);
          }
        }
        occupied[t] = 0;
        occupied[b] = 1;
        if (frozen || h === Infinity || key < 0) continue;
        const prev = best.get(key);
        if (prev !== undefined && stG[prev]! <= g) continue;
        const ni = stBoxes.length;
        best.set(key, ni);
        stBoxes.push(next!);
        stPlayer.push(b);
        stParent.push(si);
        stPushBox.push(b);
        stPushDir.push(di);
        stG.push(g);
        stKey.push(key);
        heapPush([g + h, h, seq++, ni]);
      }
    }
    for (const b of boxes) occupied[b] = 0;
  }
  if (found < 0) return null;

  // Rebuild the move string: for each push on the path, walk to the push
  // square (around the boxes as they stand before the push), then push.
  const path: number[] = [];
  for (let s2 = found; s2 > 0; s2 = stParent[s2]!) path.push(s2);
  path.reverse();
  let moves = '';
  let player = startPlayer;
  let boxes = startBoxes;
  for (const s2 of path) {
    const nextBoxes = stBoxes[s2]!;
    const dir = DIRS[stPushDir[s2]!]!;
    const d = dir.dy * w + dir.dx;
    const from = stPushBox[s2]!;
    for (const b of boxes) occupied[b] = 1;
    const walk = walkTo(ctx, occupied, player, from - d);
    for (const b of boxes) occupied[b] = 0;
    if (walk === null) return null; // should not happen: the push square was reachable
    moves += walk + dir.push;
    player = from;
    boxes = nextBoxes;
  }
  return { pushes: path.length, moves, player };
}

interface Room { squares: number[]; boxes: number[]; goals: number[]; done: boolean }

/**
 * Split a level into rooms: cut the hallway row's dead squares (a box there
 * reaches no goal) out of the floor; each remaining part holding boxes is a
 * room. Null when the level is no set of rooms.
 */
function roomsOf(level: Level): Room[] | null {
  const { w, wall, goals, boxes, player } = level;
  const live = liveSquares(level);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) if (!wall[i] && !live[i]) cut[i] = 1;
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms: Room[] = [];
  for (let s2 = 0; s2 < wall.length; s2++) {
    if (wall[s2] || cut[s2] || part[s2] >= 0) continue;
    const squares = [s2];
    part[s2] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (const dir of DIRS) {
        const q = squares[i]! + dir.dy * w + dir.dx;
        if (q < 0 || q >= wall.length) continue;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ squares, boxes: [], goals: [], done: false });
  }
  for (const b of boxes) rooms[part[b]!]!.boxes.push(b);
  for (const g of goals) if (part[g]! >= 0) rooms[part[g]!]!.goals.push(g);
  if (goals.some((g) => part[g]! < 0)) return null;
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/** Solve a level room by room; null when any room's search gives up. */
function solveByRooms(level: Level): Solution | null {
  const rooms = roomsOf(level);
  if (rooms === null) return null;
  let player = level.player;
  let pushes = 0;
  let moves = '';
  for (const room of rooms) {
    const wall = level.wall.slice();
    for (const other of rooms) {
      if (other === room) continue;
      // Other rooms' boxes stand as walls: on their goals once solved, on
      // their starting squares while unsolved.
      for (const b of other.done ? other.goals : other.boxes) wall[b] = 1;
    }
    const sub: Level = {
      w: level.w, h: level.h, wall,
      goals: room.goals.slice().sort((a, b) => a - b),
      boxes: room.boxes.slice().sort((a, b) => a - b),
      player,
    };
    const solved = solvePush(sub, ROOM_EXPANSIONS);
    if (solved === null) return null;
    pushes += solved.pushes;
    moves += solved.moves;
    player = solved.player;
    room.done = true;
  }
  return { pushes, moves, player };
}

/** A level's moves: room by room when possible, else one search over the whole level. */
function solveLevel(text: string): string {
  const level = parse(text);
  const byRooms = solveByRooms(level);
  if (byRooms !== null) return byRooms.moves;
  const whole = solvePush(level, WHOLE_EXPANSIONS);
  return whole !== null ? whole.moves : '';
}

for (const id of readdirSync(I).sort()) {
  const moves = solveLevel(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
