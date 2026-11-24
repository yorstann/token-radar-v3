// model.ts — Token Launch Radar v3: a value model that is FIT, not asserted.
//
// v1 and v2 priced a launch with hand-set weights. v3 fits its price on the snapshot
// itself: every protocol in the sealed file that already trades (a positive market
// cap and a listing date, 95 of the 128 priced protocols) is a labelled example of
// what the market paid for a protocol with these features. The model is
//
//   log(mcap) = category offset + intercept + sum of kept terms
//
// where a category offset is the category's mean log(mcap/TVL) shrunk toward the
// global mean, and the candidate terms are log TVL, log age, log(1 + chains) and a
// tanh-damped 7-day momentum. A term is KEPT only if adding it lowers the held-out
// median absolute percentage error in BOTH validations this file runs — a 5-fold
// cross-validation (folds by slug order) and a time split at the median listing
// date — each refitting offsets and coefficients on its training part alone, so no
// held-out row ever prices itself. Terms that fail either test are dropped, and the
// drops are reported with their numbers next to the keeps.
//
// The fitted model then prices the 342 protocols that record no token. A structural
// guardrail carried from v1 applies after the fit: products of an already-tokenized
// parent (the wrapped/staked/bridged and exchange categories the watchlist rule
// excludes, versioned names, wrapped/staked names) are priced at 0.35x and flagged,
// because they are not clean launch candidates whatever the regression says.
//
// Everything is a pure function of the snapshot: no clock, no randomness, no network.

import { dot, median, medianApe, ols, quantile, rSquared, roundTo } from './ols.ts';

/** The snapshot's protocol entry: exactly the ten fields the sealed input carries. */
export interface SnapshotProtocol {
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  mcap_usd: number | null;
  listed_at: string | null;
  gecko_id: string | null;
  symbol: string | null;
}

/** The whole snapshot document. */
export interface Snapshot {
  as_of: string;
  source: string;
  floor_usd: number;
  protocols: SnapshotProtocol[];
  watchlist: string[];
}

/** Why the snapshot is not one this model can score. */
export class SnapshotError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'SnapshotError';
  }
}

const DAY_MS = 86_400_000;

/** A whole day count from `from` to `to`, clamped at zero; null when either date is absent or bad. */
export function ageDays(asOf: string, listedAt: string | null): number | null {
  if (listedAt === null) return null;
  const a = Date.parse(`${asOf}T00:00:00Z`);
  const b = Date.parse(`${listedAt}T00:00:00Z`);
  if (!Number.isFinite(a) || !Number.isFinite(b)) return null;
  return Math.max(0, Math.floor((a - b) / DAY_MS));
}

/** Whether a protocol records no token at all: the board's 342 rows are exactly these. */
export function isPreToken(p: SnapshotProtocol): boolean {
  return p.mcap_usd === null && p.gecko_id === null && p.symbol === null;
}

/** Whether a protocol already trades with a usable price: the fitting pool. */
function isPriced(p: SnapshotProtocol): boolean {
  return p.mcap_usd !== null && p.mcap_usd > 0;
}

// ------------------------------------------------------------ shadow guardrail

const SHADOW_CATEGORIES = new Set(['Bridge', 'Canonical Bridge', 'Liquid Staking', 'Liquid Restaking', 'Restaked BTC', 'CEX']);
const EXCHANGE_PREFIXES = ['Binance', 'Coinbase', 'Bybit', 'Kraken', 'Bitget', 'OKX', 'Bitfinex', 'Gemini'];
const VERSIONED = /\s(V\d+(\.\d+)?|v\d+(\.\d+)?|\d+\.\d+)$/;
const WRAPPED_WORD = /\b(wrapped|staked|staking)\b/i;

/** The post-fit discount for products of an already-tokenized parent, and its reason, or null. */
export function shadowOf(p: SnapshotProtocol): { factor: number; reason: string } | null {
  if (p.category !== null && SHADOW_CATEGORIES.has(p.category)) {
    return { factor: 0.35, reason: `${p.category} is a category the watchlist rule sets aside: a product of, or a venue for, tokens that already trade` };
  }
  if (WRAPPED_WORD.test(p.name)) return { factor: 0.35, reason: 'the name says wrapped or staked: a claim on a token that already trades, not a launch of its own' };
  for (const ex of EXCHANGE_PREFIXES) {
    if (p.name.startsWith(ex)) return { factor: 0.35, reason: `the name opens with ${ex}: an exchange's own product, priced off its parent` };
  }
  if (VERSIONED.test(p.name)) return { factor: 0.35, reason: 'the name ends in a version marker: a later pool of a protocol, not a first launch' };
  return null;
}

// ------------------------------------------------------------ features

/** The model's term keys, in the fixed order term selection tries them. */
export type TermKey = 'log_tvl' | 'log_age' | 'log_chains' | 'momentum';
export const TERM_KEYS: readonly TermKey[] = ['log_tvl', 'log_age', 'log_chains', 'momentum'];
export const TERM_LABELS: Record<TermKey, string> = {
  log_tvl: 'log TVL (scale)',
  log_age: 'log age in days since listing',
  log_chains: 'log(1 + chain count)',
  momentum: '7-day TVL momentum, tanh(change / 20)',
};

/** One protocol's feature vector; log_age is null when the snapshot records no listing date. */
export interface Features {
  log_tvl: number;
  log_age: number | null;
  log_chains: number;
  momentum: number;
}

export function featuresOf(p: SnapshotProtocol, asOf: string): Features {
  const age = ageDays(asOf, p.listed_at);
  return {
    log_tvl: Math.log(p.tvl_usd),
    log_age: age === null ? null : Math.log(Math.max(1, age)),
    log_chains: Math.log1p(p.chains.length),
    momentum: Math.tanh((p.change_7d ?? 0) / 20),
  };
}

/** A feature's value with the fit set's median age standing in for an unrecorded one. */
function featureValue(f: Features, term: TermKey, medianLogAge: number): number {
  return term === 'log_age' ? (f.log_age ?? medianLogAge) : f[term];
}

// ------------------------------------------------------------ the fit

/** How strongly a category's own mean is shrunk toward the global mean (a pseudo-count). */
export const SHRINKAGE_K = 8;

/** A fitted model: shrunk category offsets plus OLS coefficients over [intercept, ...terms]. */
export interface FittedModel {
  terms: TermKey[];
  beta: number[];
  offsets: Map<string, number>;
  globalLogRatio: number;
  medianLogAge: number;
}

/** Per-category shrunk means of log(mcap/TVL) over the given rows, and the global mean. */
function categoryOffsets(rows: readonly SnapshotProtocol[]): { offsets: Map<string, number>; globalLogRatio: number } {
  const ratios = rows.map((p) => Math.log(p.mcap_usd! / p.tvl_usd));
  const globalLogRatio = ratios.reduce((s, r) => s + r, 0) / ratios.length;
  const byCategory = new Map<string, number[]>();
  rows.forEach((p, i) => {
    const key = p.category ?? 'Uncategorized';
    const list = byCategory.get(key);
    if (list === undefined) byCategory.set(key, [ratios[i]!]);
    else list.push(ratios[i]!);
  });
  const offsets = new Map<string, number>();
  for (const [cat, list] of byCategory) {
    const sum = list.reduce((s, r) => s + r, 0);
    offsets.set(cat, (sum + SHRINKAGE_K * globalLogRatio) / (list.length + SHRINKAGE_K));
  }
  return { offsets, globalLogRatio };
}

/** Fit offsets and coefficients on the given complete rows for the given term set. */
function fitOn(rows: readonly SnapshotProtocol[], asOf: string, terms: readonly TermKey[]): FittedModel {
  const { offsets, globalLogRatio } = categoryOffsets(rows);
  const feats = rows.map((p) => featuresOf(p, asOf));
  const medianLogAge = median(feats.map((f) => f.log_age).filter((v): v is number => v !== null));
  const x = feats.map((f) => [1, ...terms.map((t) => featureValue(f, t, medianLogAge))]);
  const y = rows.map((p, i) => Math.log(p.mcap_usd!) - (offsets.get(p.category ?? 'Uncategorized') ?? globalLogRatio));
  return { terms: [...terms], beta: ols(x, y), offsets, globalLogRatio, medianLogAge };
}

/** A fitted model's predicted log market cap for one protocol. */
function predictLog(model: FittedModel, p: SnapshotProtocol, asOf: string): number {
  const f = featuresOf(p, asOf);
  const row = [1, ...model.terms.map((t) => featureValue(f, t, model.medianLogAge))];
  return (model.offsets.get(p.category ?? 'Uncategorized') ?? model.globalLogRatio) + dot(model.beta, row);
}

// ------------------------------------------------------------ validation

/** One validation's held-out scores. Errors are fractions in data, percents in reports. */
export interface ValidationScores {
  n: number;
  r2_log: number;
  r2_usd: number;
  median_ape: number;
}

function scoreRows(rows: readonly SnapshotProtocol[], predictedLog: readonly number[]): ValidationScores {
  const actual = rows.map((p) => p.mcap_usd!);
  const predicted = predictedLog.map((v) => Math.exp(v));
  return {
    n: rows.length,
    r2_log: rSquared(actual.map((a) => Math.log(a)), predictedLog),
    r2_usd: rSquared(actual, predicted),
    median_ape: medianApe(actual, predicted),
  };
}

/** 5-fold cross-validation: folds are index mod 5 over the slug-sorted fit set. */
function crossValidate(fitSet: readonly SnapshotProtocol[], asOf: string, terms: readonly TermKey[]): ValidationScores {
  const predicted = new Array<number>(fitSet.length).fill(0);
  for (let fold = 0; fold < 5; fold++) {
    const train = fitSet.filter((_, i) => i % 5 !== fold);
    const model = fitOn(train, asOf, terms);
    fitSet.forEach((p, i) => {
      if (i % 5 === fold) predicted[i] = predictLog(model, p, asOf);
    });
  }
  return scoreRows(fitSet, predicted);
}

/** The time split: train on listings before the median listing date, test on the rest. */
function timeSplit(fitSet: readonly SnapshotProtocol[], asOf: string, terms: readonly TermKey[]): { scores: ValidationScores; splitDate: string; trainN: number } {
  const dates = fitSet.map((p) => p.listed_at!).sort();
  const splitDate = dates[Math.floor(dates.length / 2)]!;
  const train = fitSet.filter((p) => p.listed_at! < splitDate);
  const test = fitSet.filter((p) => p.listed_at! >= splitDate);
  const model = fitOn(train, asOf, terms);
  return { scores: scoreRows(test, test.map((p) => predictLog(model, p, asOf))), splitDate, trainN: train.length };
}

// ------------------------------------------------------------ term selection

/** One tried addition in the forward selection, with both validations' errors before and after. */
export interface TermTrial {
  step: number;
  term: TermKey;
  kept: boolean;
  cv_ape_before: number;
  cv_ape_after: number;
  time_ape_before: number;
  time_ape_after: number;
}

/** A candidate term's final verdict, from the trial that decided it. */
export interface TermVerdict {
  term: TermKey;
  label: string;
  status: 'kept' | 'dropped';
  cv_ape_before: number;
  cv_ape_after: number;
  time_ape_before: number;
  time_ape_after: number;
}

/**
 * Forward selection under the house rule of this model: a term joins only when it
 * lowers the held-out median APE in both validations; among the terms that pass,
 * the one with the largest combined error reduction joins first. Every trial is
 * logged, kept or not, so the dashboard can show the drops with their numbers.
 */
function selectTerms(fitSet: readonly SnapshotProtocol[], asOf: string): { terms: TermKey[]; trials: TermTrial[]; verdicts: TermVerdict[] } {
  const trials: TermTrial[] = [];
  const verdicts = new Map<TermKey, TermVerdict>();
  let current: TermKey[] = [];
  let curCv = crossValidate(fitSet, asOf, current).median_ape;
  let curTime = timeSplit(fitSet, asOf, current).scores.median_ape;
  const remaining = [...TERM_KEYS];
  let step = 0;
  while (remaining.length > 0) {
    step++;
    let best: { term: TermKey; cv: number; time: number; gain: number; record: TermTrial } | null = null;
    for (const term of remaining) {
      const trial = [...current, term];
      const cv = crossValidate(fitSet, asOf, trial).median_ape;
      const time = timeSplit(fitSet, asOf, trial).scores.median_ape;
      const record: TermTrial = { step, term, kept: false, cv_ape_before: curCv, cv_ape_after: cv, time_ape_before: curTime, time_ape_after: time };
      trials.push(record);
      if (cv < curCv && time < curTime) {
        const gain = curCv - cv + (curTime - time);
        if (best === null || gain > best.gain) best = { term, cv, time, gain, record };
      }
    }
    if (best === null) break;
    best.record.kept = true;
    verdicts.set(best.term, {
      term: best.term, label: TERM_LABELS[best.term], status: 'kept',
      cv_ape_before: curCv, cv_ape_after: best.cv, time_ape_before: curTime, time_ape_after: best.time,
    });
    current = [...current, best.term];
    curCv = best.cv;
    curTime = best.time;
    remaining.splice(remaining.indexOf(best.term), 1);
  }
  for (const term of remaining) {
    const last = [...trials].reverse().find((t) => t.term === term)!;
    verdicts.set(term, {
      term, label: TERM_LABELS[term], status: 'dropped',
      cv_ape_before: last.cv_ape_before, cv_ape_after: last.cv_ape_after,
      time_ape_before: last.time_ape_before, time_ape_after: last.time_ape_after,
    });
  }
  return { terms: current, trials, verdicts: TERM_KEYS.map((t) => verdicts.get(t)!) };
}

// ------------------------------------------------------------ the fit report

/** One category's fitted offset, as the dashboard states it: a multiple of TVL. */
export interface CategoryFit {
  category: string;
  n: number;
  multiple: number;
}

/** Everything the dashboard and the README quote about the fit. */
export interface FitReport {
  model_version: string;
  priced_n: number;
  fit_n: number;
  dateless_n: number;
  shrinkage_k: number;
  global_multiple: number;
  intercept: number;
  coefficients: { term: TermKey; label: string; coef: number }[];
  verdicts: TermVerdict[];
  trials: TermTrial[];
  cv: ValidationScores;
  time: ValidationScores & { split_date: string; train_n: number };
  dateless: ValidationScores;
  insample: { r2_log: number; median_ape: number };
  interval: { q10: number; q90: number; coverage: number };
  categories: CategoryFit[];
  board_fallback_categories: number;
  largest: { name: string; actual: number; fitted: number; ratio: number }[];
}

export const MODEL_VERSION = 'radar-fit-3';

const pct = (fraction: number): number => roundTo(fraction * 100, 2);
const scoresOut = (s: ValidationScores): ValidationScores => ({ n: s.n, r2_log: roundTo(s.r2_log, 4), r2_usd: roundTo(s.r2_usd, 4), median_ape: pct(s.median_ape) });

/**
 * Fit the model on the snapshot's priced protocols and validate it honestly.
 * Returns the report and the fitted model the board is priced with.
 */
export function buildFit(snapshot: Snapshot): { report: FitReport; model: FittedModel; fitSet: SnapshotProtocol[] } {
  const asOf = snapshot.as_of;
  const priced = snapshot.protocols.filter(isPriced);
  const fitSet = priced.filter((p) => p.listed_at !== null).sort((a, b) => (a.slug < b.slug ? -1 : 1));
  const dateless = priced.filter((p) => p.listed_at === null);
  const { terms, trials, verdicts } = selectTerms(fitSet, asOf);
  const model = fitOn(fitSet, asOf, terms);

  const cv = crossValidate(fitSet, asOf, terms);
  const time = timeSplit(fitSet, asOf, terms);
  // The date-less priced protocols are a bonus check: predicted with the fit set's
  // median age standing in for the listing date they do not record.
  const datelessScores = scoreRows(dateless, dateless.map((p) => predictLog(model, p, asOf)));
  const fitted = fitSet.map((p) => predictLog(model, p, asOf));
  const insample = scoreRows(fitSet, fitted);

  // The prediction band: the 10th-90th percentiles of the out-of-fold residuals of
  // the selected model, and how often that band held the actual cap.
  const oof = new Array<number>(fitSet.length).fill(0);
  for (let fold = 0; fold < 5; fold++) {
    const foldModel = fitOn(fitSet.filter((_, i) => i % 5 !== fold), asOf, terms);
    fitSet.forEach((p, i) => {
      if (i % 5 === fold) oof[i] = predictLog(foldModel, p, asOf);
    });
  }
  const residuals = fitSet.map((p, i) => Math.log(p.mcap_usd!) - oof[i]!);
  const q10 = quantile(residuals, 0.1);
  const q90 = quantile(residuals, 0.9);
  const coverage = residuals.filter((r) => r >= q10 && r <= q90).length / residuals.length;

  const counts = new Map<string, number>();
  for (const p of fitSet) counts.set(p.category ?? 'Uncategorized', (counts.get(p.category ?? 'Uncategorized') ?? 0) + 1);
  const categories = [...model.offsets.entries()]
    .map(([category, offset]) => ({ category, n: counts.get(category) ?? 0, multiple: roundTo(Math.exp(offset), 4) }))
    .sort((a, b) => b.n - a.n || (a.category < b.category ? -1 : 1));
  const boardCategories = new Set(snapshot.protocols.filter(isPreToken).map((p) => p.category ?? 'Uncategorized'));
  const fallback = [...boardCategories].filter((c) => !model.offsets.has(c)).length;

  const largest = [...fitSet]
    .sort((a, b) => b.mcap_usd! - a.mcap_usd!)
    .slice(0, 10)
    .map((p) => {
      const fit = Math.exp(predictLog(model, p, asOf));
      return { name: p.name, actual: p.mcap_usd!, fitted: Math.round(fit), ratio: roundTo(fit / p.mcap_usd!, 4) };
    });

  const report: FitReport = {
    model_version: MODEL_VERSION,
    priced_n: priced.length,
    fit_n: fitSet.length,
    dateless_n: dateless.length,
    shrinkage_k: SHRINKAGE_K,
    global_multiple: roundTo(Math.exp(model.globalLogRatio), 4),
    intercept: roundTo(model.beta[0]!, 6),
    coefficients: terms.map((t, i) => ({ term: t, label: TERM_LABELS[t], coef: roundTo(model.beta[i + 1]!, 6) })),
    verdicts: verdicts.map((v) => ({ ...v, cv_ape_before: pct(v.cv_ape_before), cv_ape_after: pct(v.cv_ape_after), time_ape_before: pct(v.time_ape_before), time_ape_after: pct(v.time_ape_after) })),
    trials: trials.map((t) => ({ ...t, cv_ape_before: pct(t.cv_ape_before), cv_ape_after: pct(t.cv_ape_after), time_ape_before: pct(t.time_ape_before), time_ape_after: pct(t.time_ape_after) })),
    cv: scoresOut(cv),
    time: { ...scoresOut(time.scores), split_date: time.splitDate, train_n: time.trainN },
    dateless: scoresOut(datelessScores),
    insample: { r2_log: roundTo(insample.r2_log, 4), median_ape: pct(insample.median_ape) },
    interval: { q10: roundTo(q10, 4), q90: roundTo(q90, 4), coverage: roundTo(coverage, 3) },
    categories,
    board_fallback_categories: fallback,
    largest,
  };
  return { report, model, fitSet };
}

// ------------------------------------------------------------ farm plans

/** A farm step: the action, what it costs, and how long it takes. */
export interface FarmStep {
  text: string;
  cost: string;
  time: string;
}

/** A step as one string: its action, then its cost and time, within the gate's 20..600 characters. */
export function stepText(step: FarmStep): string {
  let s = `${step.text} Cost: ${step.cost}. Time: ${step.time}.`;
  if (s.length > 600) s = `${s.slice(0, 596)}...`;
  if (s.length < 20) s = `${s} Hold the position and check the protocol's own page.`;
  return s;
}

/** The first chain a protocol names, or a neutral phrase. */
function chainOf(p: SnapshotProtocol): string {
  return p.chains.length > 0 ? p.chains[0]! : 'its main chain';
}

/** What one transaction roughly costs on a chain, in words a farmer can budget with. */
function gasFor(chain: string): string {
  if (chain === 'Ethereum') return 'about $6-12 of gas';
  if (chain === 'Solana') return 'under $1 of fees';
  if (chain === 'Bitcoin') return 'about $2-5 of network fees';
  if (['Arbitrum', 'Base', 'Polygon', 'Avalanche', 'Binance', 'Optimism', 'Linea', 'Scroll', 'zkSync Era', 'Mantle'].includes(chain)) return 'about $1-3 of gas';
  return 'about $2-5 of gas';
}

/** How much capital one farm position needs, from the protocol's own TVL. */
function capitalFor(p: SnapshotProtocol): number {
  if (p.tvl_usd >= 5_000_000_000) return 250;
  if (p.tvl_usd >= 1_000_000_000) return 150;
  if (p.tvl_usd >= 100_000_000) return 100;
  if (p.tvl_usd >= 25_000_000) return 75;
  return 50;
}

/** How long to hold, from the protocol's age: younger protocols get a longer qualifying window. */
function holdDaysFor(age: number | null): number {
  if (age === null) return 45;
  if (age < 180) return 60;
  if (age < 365) return 45;
  return 30;
}

/** The category's core action and its follow-up, in the words a farmer can follow. */
function actionsFor(p: SnapshotProtocol, capital: number, chain: string): [string, string] {
  const c = (p.category ?? '').toLowerCase();
  const n = p.name;
  const x = `$${capital}`;
  if (c.includes('lend') && !c.includes('uncollateralized')) return [
    `Supply ${x} of USDC to ${n} on ${chain} and leave the supplied balance in place.`,
    `Borrow a small amount of a second asset against the supply — never more than a third of its value — and repay it inside the week, so the wallet appears on both sides of the market.`];
  if (c.includes('cdp')) return [
    `Deposit ${x} of collateral into a ${n} vault on ${chain} and mint a conservative amount of its stablecoin, keeping the collateral ratio far above liquidation.`,
    `Put the minted stablecoin into the protocol's own stability pool or farm, so the wallet both mints and provides liquidity.`];
  if (c.includes('dex')) return [
    `Swap a small amount through ${n} on ${chain}, then provide ${x} of liquidity to its deepest stable or blue-chip pool.`,
    `Keep the LP position open and compound the trading fees on a weekly rhythm, so the position stays active in the protocol's records.`];
  if (c.includes('bridge')) return [
    `Bridge ${x} of USDC to ${chain} through ${n}, then make one genuine transaction on the far side the same day.`,
    `Bridge a smaller amount back a week later: round trips in both directions are what bridge usage records measure.`];
  if (c.includes('liquid staking') || c.includes('staking pool')) return [
    `Stake ${x} of the chain's native asset through ${n} on ${chain} and hold the liquid receipt token it returns.`,
    `Put the receipt token to work in one integrated market the protocol lists, so the wallet shows both the stake and a second interaction.`];
  if (c.includes('restaking') || c.includes('restaked')) return [
    `Restake ${x} of a supported staking asset through ${n} on ${chain} and keep the receipt position open.`,
    `Pick the operator or strategy deliberately and record why: restaking records reward wallets that choose, not wallets that park.`];
  if (c.includes('cex')) return [
    `Open an account at ${n}, complete its identity verification, and make a first spot trade of about ${x}.`,
    `Use one more surface the exchange offers — an earn deposit, a convert trade or a card payment — so the account is active in more than one place.`];
  if (c === 'chain') return [
    `Bridge ${x} to the ${n} network and hold a working balance there.`,
    `Use two or three native apps in the first week — a swap, a mint, a name or a game — so the wallet reads as a real early user of the chain.`];
  if (c.includes('derivative') || c.includes('options')) return [
    `Deposit ${x} of USDC as margin, or into the main ${n} vault, on ${chain}.`,
    `Open one small hedged position (or hold one full vault cycle) and let it settle, so the history shows a completed trade and not a bare deposit.`];
  if (c.includes('rwa')) return [
    `Mint or buy ${x} of ${n}'s tokenised asset on ${chain} through its own app and hold it.`,
    `Complete the investor profile and any periodic attestation the protocol asks for, so the holding stays in good standing.`];
  if (c.includes('curator') || c.includes('allocator') || c.includes('liquidity manager') || c.includes('collateral')) return [
    `Deposit ${x} into a ${n} vault on ${chain} whose curator and strategy you have read end to end first.`,
    `Follow the vault's allocation page weekly and move with it once, so the history shows an active depositor rather than a parked one.`];
  if (c.includes('basis')) return [
    `Deposit ${x} of USDC into ${n}'s basis vault on ${chain} and keep the shares.`,
    `Read the vault's funding report weekly and stay through one full funding cycle before judging the position.`];
  if (c.includes('prediction')) return [
    `Fund the wallet with ${x} on ${chain} and take one position in a liquid ${n} market you genuinely have a view on.`,
    `Let the market settle, claim the result, and roll a smaller amount into a second market so the account shows repeat activity.`];
  if (c.includes('stablecoin')) return [
    `Mint or wrap ${x} of the stable asset through ${n} on ${chain} and hold the resulting form.`,
    `Supply the wrapped asset to one pool the protocol points to, so the wallet both mints and provides liquidity.`];
  if (c.includes('btc')) return [
    `Mint or deposit ${x} of bitcoin-backed collateral through ${n} on ${chain} and hold the resulting asset.`,
    `Use the minted asset in one listed market — a lend, a pool or a vault — instead of leaving it idle.`];
  if (c.includes('cedefi') || c.includes('yield') || c.includes('aggregator') || c.includes('farm')) return [
    `Deposit ${x} through ${n} on ${chain} into its main yield product and keep the position open.`,
    `Withdraw a small part once and redeposit it, so the wallet's history shows a complete deposit cycle.`];
  if (c.includes('insurance')) return [
    `Buy ${x} of cover from ${n} on ${chain} for a position you actually hold, and keep the policy to expiry.`,
    `If the protocol runs a capital pool, stake a smaller amount into it too, so the wallet appears on both sides of the market.`];
  if (c.includes('payment') || c.includes('card')) return [
    `Open a ${n} account, load about ${x}, and make one real payment with it.`,
    `Repeat a small payment weekly — a subscription or a top-up — so the account builds the steady history these programs reward.`];
  if (c.includes('governance')) return [
    `Hold or stake ${x} of the governance asset ${n} works with on ${chain}, and register the wallet where the protocol counts participation.`,
    `Vote on every live proposal while the position is open, and add one substantive comment to a discussion thread.`];
  if (c.includes('privacy')) return [
    `Deposit ${x} through ${n} on ${chain} using its standard published flow, in full compliance with its terms.`,
    `Withdraw to the fresh address the flow produces, and keep the records of both legs with your farm log.`];
  if (c.includes('index')) return [
    `Buy ${x} of the ${n} index on ${chain} and hold the index token.`,
    `Add a small amount again at the index's next scheduled rebalance, so the history spans more than one index state.`];
  if (c.includes('otc')) return [
    `Register on ${n}, complete its checks, and place one small order of about ${x} on ${chain}.`,
    `Settle the trade promptly and leave feedback, so the account records a finished over-the-counter cycle.`];
  if (c.includes('uncollateralized')) return [
    `Deposit ${x} into a ${n} lending pool on ${chain} whose borrowers and terms you have read in full.`,
    `Stay through one complete loan cycle and withdraw only at its end, so the wallet's history shows a finished term.`];
  return [
    `Deposit ${x} of USDC into ${n} on ${chain} through its own app and keep the position open.`,
    `Make one further interaction the app offers — a claim, a vote or a second deposit — so the wallet's history is more than a single touch.`];
}

/** The risk sentence for the hold step, worded from the protocol's own seven-day print. */
function riskNote(p: SnapshotProtocol): string {
  if (p.change_7d === null) return `The snapshot records no seven-day change for this protocol: treat the missing print as a reason to start small and watch one full week before sizing up.`;
  const v = roundTo(p.change_7d, 1);
  if (p.change_7d <= -15) return `TVL fell ${Math.abs(v)}% in the last seven days: size this position at the small end and re-check the protocol's own channels before adding anything.`;
  if (p.change_7d >= 20) return `TVL jumped ${v}% in seven days: do not chase the print — keep the planned size and let a week pass before any second deposit.`;
  return `The seven-day TVL print (${v > 0 ? '+' : ''}${v}%) is unremarkable: hold the planned size for the planned window.`;
}

/**
 * A protocol's farm plan: six steps, each an action with a cost and a time, derived
 * from the protocol's own record (category, chains, TVL, age, seven-day print) —
 * never from a hand-written list of names.
 */
export function farmSteps(p: SnapshotProtocol, age: number | null): FarmStep[] {
  const chain = chainOf(p);
  const capital = capitalFor(p);
  const hold = holdDaysFor(age);
  const gas = gasFor(chain);
  const [core, second] = actionsFor(p, capital, chain);
  return [
    {
      text: `Use a fresh wallet that holds only the funds you plan to farm with, and keep its seed phrase offline and never typed into a site.`,
      cost: 'under $3 to fund and prepare',
      time: '10 minutes',
    },
    { text: core, cost: `$${capital} of capital and ${gas}`, time: '15 minutes to set up' },
    { text: second, cost: `${gas} for the follow-up`, time: '15 minutes' },
    {
      text: `Repeat one qualifying action every week — a deposit, a swap, a vote or a claim — from the same wallet, so the activity spans several weekly snapshots.`,
      cost: 'about $2 of gas a week',
      time: '10 minutes a week',
    },
    {
      text: `Hold the position for at least ${hold} days. ${riskNote(p)}`,
      cost: `the $${capital} stays in the position`,
      time: `${hold} days`,
    },
    {
      text: `Check the protocol's own points or rewards page — only from the link the protocol itself publishes — and write the wallet, dates and every transaction into a farm log you can repeat or unwind from.`,
      cost: 'nothing to check, and never pay to claim',
      time: '10 minutes',
    },
  ];
}

// ------------------------------------------------------------ the board

/** The prediction split into additive log-space parts, for the dashboard's factor bars. */
export interface PredictionParts {
  /** The constant every row shares: global ratio + intercept + each kept term at its fit-set median. */
  base: number;
  /** This row's category offset minus the global ratio. */
  category: number;
  /** Each kept term's contribution away from its fit-set median. */
  terms: Partial<Record<TermKey, number>>;
  /** log(0.35) when the shadow guardrail applies, else 0. */
  shadow: number;
}

/** One row of board.json: a pre-token protocol priced by the fitted model. */
export interface BoardRow {
  rank: number;
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  mcap_usd: number | null;
  listed_at: string | null;
  gecko_id: string | null;
  symbol: string | null;
  token_status: 'pre-token';
  age_days: number | null;
  implied_value_usd: number;
  implied_low_usd: number;
  implied_high_usd: number;
  watchlist: boolean;
  shadow: boolean;
  shadow_reason: string | null;
  signal: number;
  model: {
    version: string;
    category_multiple: number;
    parts: PredictionParts;
  };
  farm_steps: string[];
}

/** A live-token protocol: it already trades, so the board shows the market's own value. */
export interface LiveRow {
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  mcap_usd: number | null;
  change_7d: number | null;
}

/** Everything the dashboard renders: the board, the live set, the fit report and the stats. */
export interface RadarData {
  as_of: string;
  source: string;
  floor_usd: number;
  model_version: string;
  stats: {
    scanned: number;
    priced_n: number;
    fit_n: number;
    board_size: number;
    watchlist_covered: number;
    watchlist_total: number;
    shadow_n: number;
  };
  fit: FitReport;
  rows: BoardRow[];
  live: LiveRow[];
}

const MAX_IMPLIED = 100_000_000_000_000;

/** An implied dollar figure inside the gate's bounds: a whole dollar, at least 1, at most 1e14. */
function clampImplied(v: number): number {
  if (!Number.isFinite(v) || v <= 1) return 1;
  return Math.min(MAX_IMPLIED, Math.round(v));
}

/** Price every pre-token protocol with the fitted model and rank the board. */
export function buildBoard(snapshot: Snapshot, model: FittedModel, report: FitReport): BoardRow[] {
  const asOf = snapshot.as_of;
  const watch = new Set(snapshot.watchlist);
  // Feature references: the fit set's median of each feature, so a row's parts are
  // its distance from the typical fitted protocol.
  const fitFeats = snapshot.protocols.filter((p) => isPriced(p) && p.listed_at !== null).map((p) => featuresOf(p, asOf));
  const refs = {} as Record<TermKey, number>;
  for (const t of TERM_KEYS) {
    refs[t] = t === 'log_age'
      ? median(fitFeats.map((f) => f.log_age).filter((v): v is number => v !== null))
      : median(fitFeats.map((f) => f[t]));
  }
  const base = model.globalLogRatio + model.beta[0]! + model.terms.reduce((s, t, i) => s + model.beta[i + 1]! * refs[t], 0);
  const { q10, q90 } = report.interval;

  const rows = snapshot.protocols.filter(isPreToken).map((p) => {
    const f = featuresOf(p, asOf);
    const predLog = predictLog(model, p, asOf);
    const shadow = shadowOf(p);
    const factor = shadow?.factor ?? 1;
    const parts: PredictionParts = {
      base: roundTo(base, 4),
      category: roundTo((model.offsets.get(p.category ?? 'Uncategorized') ?? model.globalLogRatio) - model.globalLogRatio, 4),
      terms: {},
      shadow: shadow === null ? 0 : roundTo(Math.log(shadow.factor), 4),
    };
    model.terms.forEach((t, i) => {
      parts.terms[t] = roundTo(model.beta[i + 1]! * (featureValue(f, t, model.medianLogAge) - refs[t]), 4);
    });
    const age = ageDays(asOf, p.listed_at);
    return {
      p,
      age,
      shadow,
      parts,
      implied: clampImplied(Math.exp(predLog) * factor),
      low: clampImplied(Math.exp(predLog + q10) * factor),
      high: clampImplied(Math.exp(predLog + q90) * factor),
    };
  });
  rows.sort((a, b) => b.implied - a.implied || (a.p.slug < b.p.slug ? -1 : 1));
  const n = rows.length;
  return rows.map((r, i) => ({
    rank: i + 1,
    slug: r.p.slug,
    name: r.p.name,
    category: r.p.category,
    chains: r.p.chains,
    tvl_usd: r.p.tvl_usd,
    change_7d: r.p.change_7d,
    mcap_usd: r.p.mcap_usd,
    listed_at: r.p.listed_at,
    gecko_id: r.p.gecko_id,
    symbol: r.p.symbol,
    token_status: 'pre-token',
    age_days: r.age,
    implied_value_usd: r.implied,
    implied_low_usd: Math.min(r.low, r.implied),
    implied_high_usd: Math.max(r.high, r.implied),
    watchlist: watch.has(r.p.slug),
    shadow: r.shadow !== null,
    shadow_reason: r.shadow?.reason ?? null,
    signal: n <= 1 ? 100 : Math.round((100 * (n - 1 - i)) / (n - 1)),
    model: {
      version: MODEL_VERSION,
      category_multiple: roundTo(Math.exp(model.offsets.get(r.p.category ?? 'Uncategorized') ?? model.globalLogRatio), 4),
      parts: r.parts,
    },
    farm_steps: farmSteps(r.p, r.age).map(stepText),
  }));
}

/** The live-token protocols, by market cap, for the dashboard's Live token view. */
export function liveRows(snapshot: Snapshot): LiveRow[] {
  return snapshot.protocols
    .filter((p) => !isPreToken(p) && p.category !== 'CEX')
    .sort((a, b) => (b.mcap_usd ?? 0) - (a.mcap_usd ?? 0) || b.tvl_usd - a.tvl_usd)
    .slice(0, 24)
    .map((p) => ({ slug: p.slug, name: p.name, category: p.category, chains: p.chains, tvl_usd: p.tvl_usd, mcap_usd: p.mcap_usd, change_7d: p.change_7d }));
}

/** Everything the run writes, from the snapshot: fit, board and dashboard data. */
export function buildRadar(snapshot: Snapshot): { board: BoardRow[]; radar: RadarData; report: FitReport } {
  const { report, model } = buildFit(snapshot);
  const board = buildBoard(snapshot, model, report);
  const covered = board.filter((r) => r.watchlist).length;
  const radar: RadarData = {
    as_of: snapshot.as_of,
    source: snapshot.source,
    floor_usd: snapshot.floor_usd,
    model_version: MODEL_VERSION,
    stats: {
      scanned: snapshot.protocols.length,
      priced_n: report.priced_n,
      fit_n: report.fit_n,
      board_size: board.length,
      watchlist_covered: covered,
      watchlist_total: snapshot.watchlist.length,
      shadow_n: board.filter((r) => r.shadow).length,
    },
    fit: report,
    rows: board,
    live: liveRows(snapshot),
  };
  return { board, radar, report };
}

/** Why a board is not one the gate will take, or null: coverage, ranks, order, values, farm steps. */
export function boardIssue(board: BoardRow[], snapshot: Snapshot): string | null {
  const slugs = new Set(board.map((r) => r.slug));
  const missing = snapshot.watchlist.filter((s) => !slugs.has(s));
  if (missing.length > 0) return `the board misses ${missing.length} watchlist slugs: ${missing.slice(0, 5).join(', ')}`;
  const ranks = board.map((r) => r.rank).sort((a, b) => a - b);
  for (let i = 0; i < ranks.length; i++) if (ranks[i] !== i + 1) return `rank ${i + 1} is ${ranks[i]}`;
  for (let i = 1; i < board.length; i++) if (board[i]!.implied_value_usd > board[i - 1]!.implied_value_usd) return `row ${i + 1} outranks row ${i} on implied value`;
  if (board.length >= 2 && board[0]!.implied_value_usd === board[board.length - 1]!.implied_value_usd) return 'every row shares one implied value: a board that ranks nothing by value is not a ranking';
  for (const r of board) {
    if (r.farm_steps.length < 3 || r.farm_steps.length > 12) return `${r.slug} has ${r.farm_steps.length} farm steps`;
    for (const s of r.farm_steps) if (s.length < 20 || s.length > 600) return `${r.slug} has a farm step of ${s.length} characters`;
    if (!Number.isSafeInteger(r.implied_value_usd) || r.implied_value_usd <= 0 || r.implied_value_usd > MAX_IMPLIED) return `${r.slug} has implied value ${r.implied_value_usd}`;
    if (!(r.implied_low_usd <= r.implied_value_usd && r.implied_value_usd <= r.implied_high_usd)) return `${r.slug} has an interval that does not hold its implied value`;
  }
  return null;
}

/** The snapshot out of a file's text, refused unless it is the shape the model scores. */
export function parseSnapshot(text: string): Snapshot {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    throw new SnapshotError('snapshot.json is not JSON');
  }
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) throw new SnapshotError('snapshot.json is not a JSON object');
  const o = raw as Record<string, unknown>;
  if (typeof o.as_of !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(o.as_of)) throw new SnapshotError('snapshot.json has no as_of date');
  if (typeof o.source !== 'string') throw new SnapshotError('snapshot.json has no source');
  if (typeof o.floor_usd !== 'number' || !Number.isSafeInteger(o.floor_usd) || o.floor_usd <= 0) throw new SnapshotError('snapshot.json has no floor_usd');
  if (!Array.isArray(o.protocols)) throw new SnapshotError('snapshot.json has no protocols array');
  if (!Array.isArray(o.watchlist) || o.watchlist.some((s) => typeof s !== 'string')) throw new SnapshotError('snapshot.json has no watchlist of slugs');
  const protocols = o.protocols.map((v, i) => {
    if (typeof v !== 'object' || v === null || Array.isArray(v)) throw new SnapshotError(`snapshot.json's protocol ${i} is not an object`);
    const p = v as Record<string, unknown>;
    if (typeof p.slug !== 'string' || typeof p.name !== 'string' || typeof p.tvl_usd !== 'number' || !Number.isFinite(p.tvl_usd) || p.tvl_usd <= 0) {
      throw new SnapshotError(`snapshot.json's protocol ${i} lacks a slug, a name or a positive TVL`);
    }
    return {
      slug: p.slug,
      name: p.name,
      category: typeof p.category === 'string' ? p.category : null,
      chains: Array.isArray(p.chains) ? p.chains.filter((c): c is string => typeof c === 'string') : [],
      tvl_usd: p.tvl_usd,
      change_7d: typeof p.change_7d === 'number' && Number.isFinite(p.change_7d) ? p.change_7d : null,
      mcap_usd: typeof p.mcap_usd === 'number' && Number.isFinite(p.mcap_usd) ? p.mcap_usd : null,
      listed_at: typeof p.listed_at === 'string' ? p.listed_at : null,
      gecko_id: typeof p.gecko_id === 'string' ? p.gecko_id : null,
      symbol: typeof p.symbol === 'string' ? p.symbol : null,
    };
  });
  return { as_of: o.as_of, source: o.source, floor_usd: o.floor_usd, protocols, watchlist: o.watchlist as string[] };
}

/** A compact data.js: the dashboard's whole dataset as one script, no fetch and no network. */
export function dataScript(radar: RadarData): string {
  return `window.RADAR=${JSON.stringify(radar)};\n`;
}
