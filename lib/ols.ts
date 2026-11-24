// ols.ts — the radar's small mathematics core: descriptive statistics and an ordinary
// least squares fit through the normal equations. Pure functions, no dependencies,
// no clock and no randomness, so a fit is a function of its rows alone.
//
// The solver is Gaussian elimination with partial pivoting on the normal equations
// (X'X)b = X'y. The design matrices this program builds are tiny (an intercept plus a
// handful of log-scale terms over ~100 rows) and well conditioned, so the normal
// equations are the right tool; the self-test in the build notes checks the solver
// against three hand-computed systems before any real number is trusted.

/** Why a fit cannot be made: a singular system or a shape that is not a regression. */
export class OlsError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'OlsError';
  }
}

/** The arithmetic mean; 0 on an empty list (callers never pass one). */
export function mean(xs: readonly number[]): number {
  if (xs.length === 0) return 0;
  let s = 0;
  for (const x of xs) s += x;
  return s / xs.length;
}

/** A copy of the list, sorted ascending: quantiles never mutate their input. */
function sorted(xs: readonly number[]): number[] {
  return [...xs].sort((a, b) => a - b);
}

/** The median: the middle value, or the mean of the two middle values. */
export function median(xs: readonly number[]): number {
  return quantile(xs, 0.5);
}

/**
 * The q-quantile with linear interpolation between closest ranks (the type-7 rule):
 * the value at position q*(n-1) in the sorted list, blended between its neighbours.
 */
export function quantile(xs: readonly number[], q: number): number {
  if (xs.length === 0) return 0;
  const s = sorted(xs);
  if (s.length === 1) return s[0]!;
  const pos = Math.min(1, Math.max(0, q)) * (s.length - 1);
  const lo = Math.floor(pos);
  const hi = Math.min(lo + 1, s.length - 1);
  const frac = pos - lo;
  return s[lo]! + (s[hi]! - s[lo]!) * frac;
}

/** Solve A x = b for a square A by Gaussian elimination with partial pivoting. */
export function solveLinear(a: readonly (readonly number[])[], b: readonly number[]): number[] {
  const n = b.length;
  if (a.length !== n || a.some((row) => row.length !== n)) throw new OlsError('solveLinear: the system is not square');
  // An augmented working copy; the caller's arrays are never touched.
  const m = a.map((row, i) => [...row, b[i]!]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(m[r]![col]!) > Math.abs(m[pivot]![col]!)) pivot = r;
    if (Math.abs(m[pivot]![col]!) < 1e-12) throw new OlsError('solveLinear: the system is singular');
    if (pivot !== col) [m[col], m[pivot]] = [m[pivot]!, m[col]!];
    const div = m[col]![col]!;
    for (let r = col + 1; r < n; r++) {
      const f = m[r]![col]! / div;
      if (f === 0) continue;
      for (let c = col; c <= n; c++) m[r]![c] = m[r]![c]! - f * m[col]![c]!;
    }
  }
  const x = new Array<number>(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = m[r]![n]!;
    for (let c = r + 1; c < n; c++) s -= m[r]![c]! * x[c]!;
    x[r] = s / m[r]![r]!;
  }
  return x;
}

/**
 * The least-squares coefficients of y ~ X by the normal equations. X is one row per
 * observation, one column per term (the caller includes an intercept column of ones).
 */
export function ols(x: readonly (readonly number[])[], y: readonly number[]): number[] {
  if (x.length === 0 || x.length !== y.length) throw new OlsError('ols: rows and targets disagree');
  const k = x[0]!.length;
  if (k === 0 || x.some((row) => row.length !== k)) throw new OlsError('ols: the design is ragged');
  const xtx: number[][] = Array.from({ length: k }, () => new Array<number>(k).fill(0));
  const xty = new Array<number>(k).fill(0);
  for (let i = 0; i < x.length; i++) {
    const row = x[i]!;
    for (let a = 0; a < k; a++) {
      xty[a] = xty[a]! + row[a]! * y[i]!;
      for (let b = 0; b < k; b++) xtx[a]![b] = xtx[a]![b]! + row[a]! * row[b]!;
    }
  }
  return solveLinear(xtx, xty);
}

/** The dot product of a coefficient vector with one design row. */
export function dot(beta: readonly number[], row: readonly number[]): number {
  let s = 0;
  for (let i = 0; i < beta.length; i++) s += beta[i]! * row[i]!;
  return s;
}

/** R-squared of predictions against actuals, in whatever space both are given in. */
export function rSquared(actual: readonly number[], predicted: readonly number[]): number {
  const m = mean(actual);
  let ssRes = 0;
  let ssTot = 0;
  for (let i = 0; i < actual.length; i++) {
    ssRes += (actual[i]! - predicted[i]!) ** 2;
    ssTot += (actual[i]! - m) ** 2;
  }
  return ssTot === 0 ? 0 : 1 - ssRes / ssTot;
}

/** The median absolute percentage error of dollar predictions against dollar actuals. */
export function medianApe(actual: readonly number[], predicted: readonly number[]): number {
  const apes = actual.map((a, i) => Math.abs(predicted[i]! - a) / a);
  return median(apes);
}

/** Round to a fixed number of decimals for reported figures (display only). */
export function roundTo(n: number, places: number): number {
  const f = 10 ** places;
  return Math.round(n * f) / f;
}
