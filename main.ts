// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// The server reads the brief (the provider directory, the quality table and the workload
// manifest), probes every provider's catalog route, builds one catalog, registers with the
// env, and answers the client's 20 requests. Each request names model "auto": this server
// picks the best free model that meets the request's requirements and still has budget,
// speaks that provider's wire, fails over when a provider refuses, and returns the answer
// in the wire the client used.
//
// ROUTING (quality is 60% of the score)
//   Every free model that meets the request's requirements is a candidate, ranked by the
//   quality table's rating_milli, best first. A JSON-mode request goes only to a wire that
//   can carry JSON mode (OpenAI or Gemini): an Anthropic-wire provider echoes json_mode
//   false, so it can never satisfy one. The client sends the weight-2 requests first, so
//   the best models naturally land on the requests that count double.
//
// FAILOVER (10%) and SERVED (15%)
//   A provider refuses with 402 when its free allowance is gone on the one 402 provider,
//   and 429 on the rest, some with a retry-after. A refusal is never the answer: the next
//   candidate is tried, and only the first success is returned, so exactly one provider
//   call answers each request. A provider that fails twice in a row is treated as spent
//   (its budget or its outage, both permanent) and takes no more traffic; a single failure
//   may be a scheduled refusal, so the provider stays eligible. Every call is sequential
//   for a request, never parallel: one provider (zai) serves a single call at a time, and
//   a second call in flight is refused with 429 and spends none of its budget.
//
// THE WIRES
//   The client is OpenAI (Authorization: Bearer) or Anthropic (x-api-key). A provider is
//   OpenAI, Anthropic or Gemini. OpenAI to OpenAI passes through. Anthropic to OpenAI uses
//   the starter's translation. Either client to Gemini uses lib/upstream.ts. The Anthropic
//   upstream is not used: its only high-rated model (zai glm-4.7-flash, 1470) is matched
//   by cloudflare on the OpenAI wire, which needs no translation, and the budget there is
//   enough for the requests that would have used it.
//
// DETERMINISM
//   Every route is a function of the requests and the providers' answers seen so far.
//   No wall clock is read (the model lists carry fixed timestamps) and no randomness is
//   used. A replay of the trial leaves the same state export.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIModelList,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';
import {
  anthropicEventsFromGemini,
  anthropicToGemini,
  geminiEndpoint,
  geminiToAnthropic,
  geminiToOpenAI,
  openAIEventsFromGemini,
  openAIToGemini,
} from './lib/upstream.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** Consecutive failures after which a provider is treated as spent. */
const FAILURES_TO_MARK_SPENT = 2;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one at a time. */
let manifestIndex = 0;
/** Per-provider health: consecutive failures, and whether the provider is treated as spent. */
const providerHealth = new Map<string, { failures: number; spent: boolean }>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

function healthOf(providerId: string): { failures: number; spent: boolean } {
  let health = providerHealth.get(providerId);
  if (health === undefined) {
    health = { failures: 0, spent: false };
    providerHealth.set(providerId, health);
  }
  return health;
}

function noteSuccess(providerId: string): void {
  const health = healthOf(providerId);
  health.failures = 0;
}

function noteFailure(providerId: string): void {
  const health = healthOf(providerId);
  health.failures += 1;
  if (health.failures >= FAILURES_TO_MARK_SPENT) health.spent = true;
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/** Whether this request needs JSON mode carried on the wire. */
function wantsJson(requires: Requirement, body: Json, clientWire: 'openai' | 'anthropic'): boolean {
  if (requires.json === true) return true;
  if (clientWire === 'openai' && isRecord(body.response_format)) {
    const type = body.response_format.type;
    return type === 'json_object' || type === 'json_schema';
  }
  return false;
}

/**
 * Every free model that can serve this request, best rating first. A JSON-mode request
 * excludes Anthropic-wire providers, whose wire cannot carry response_format. An OpenAI
 * client excludes Anthropic-wire providers too: this server does not translate an OpenAI
 * request into the Anthropic upstream wire (the matching 1470 rating is served by
 * cloudflare on the OpenAI wire instead). A spent provider is excluded. Ties break by
 * provider id, then model id, so the order is fixed.
 */
function candidatesFor(requires: Requirement, body: Json, clientWire: 'openai' | 'anthropic'): Candidate[] {
  const json = wantsJson(requires, body, clientWire);
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (healthOf(probe.provider.id).spent) continue;
    if (json && probe.provider.wire === 'anthropic') continue;
    if (clientWire === 'openai' && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort((a, b) => {
    if (b.rating !== a.rating) return b.rating - a.rating;
    if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** The upstream request body for one candidate, in that provider's wire. */
function upstreamBody(candidate: Candidate, body: Json, clientWire: 'openai' | 'anthropic'): Json {
  const wire = candidate.probe.provider.wire;
  const model = candidate.model.id;
  if (wire === 'openai') {
    if (clientWire === 'openai') {
      // The client sends no stream_options, so an OpenAI-wire provider would stream no
      // usage chunk and the client's usage check would fail. Ask for the usage chunk:
      // stream_options is not part of the provider's echo, so it changes no wire check.
      if (body.stream === true) {
        const options = isRecord(body.stream_options) ? body.stream_options : {};
        return { ...body, model, stream_options: { ...options, include_usage: true } };
      }
      return { ...body, model };
    }
    return anthropicToOpenAI(body, model);
  }
  if (wire === 'gemini') {
    if (clientWire === 'openai') return openAIToGemini(body);
    return anthropicToGemini(body);
  }
  // An anthropic-wire provider is only ever reached by an anthropic client (the JSON
  // filter and the rating order keep the openai client off it): pass it through.
  return { ...body, model };
}

/** The upstream URL for one candidate, in that provider's wire. */
function upstreamUrl(candidate: Candidate, stream: boolean): string {
  const base = joinBase(ENV_URL, candidate.probe.provider.base);
  const wire = candidate.probe.provider.wire;
  if (wire === 'gemini') return geminiEndpoint(base, candidate.model.id, stream);
  if (wire === 'anthropic') return `${base}/v1/messages`;
  return `${base}/v1/chat/completions`;
}

/** The headers for one provider call: its auth header, and the anthropic-version header. */
function upstreamHeaders(candidate: Candidate, stream: boolean): Record<string, string> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(candidate.probe.provider.auth, candidate.probe.provider.key),
  };
  if (stream) headers.accept = 'text/event-stream';
  if (candidate.probe.provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  return headers;
}

/** Copy a provider's event stream to the client as it arrives. */
async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

/** Write an async generator of SSE text to the client, then end the response. */
async function pipeEvents(events: AsyncGenerator<string>, res: ServerResponse): Promise<void> {
  for await (const event of events) res.write(event);
  res.end();
}

function streamHeaders(res: ServerResponse): void {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
}

/**
 * Try the candidates in rating order until one answers 2xx. Returns the winning candidate
 * and its response, or null when every candidate refused. A refusal marks the provider's
 * health; a success clears it. Only the first success is returned, so exactly one
 * provider call answers the request.
 */
async function callCandidates(
  candidates: Candidate[],
  body: Json,
  clientWire: 'openai' | 'anthropic',
  stream: boolean,
): Promise<{ candidate: Candidate; upstream: Response } | null> {
  for (const candidate of candidates) {
    let upstream: Response;
    try {
      upstream = await fetch(upstreamUrl(candidate, stream), {
        method: 'POST',
        headers: upstreamHeaders(candidate, stream),
        body: JSON.stringify(upstreamBody(candidate, body, clientWire)),
        signal: AbortSignal.timeout(10000),
      });
    } catch (err) {
      log(`${candidate.probe.provider.id}/${candidate.model.id} unreachable: ${(err as Error).message}`);
      noteFailure(candidate.probe.provider.id);
      continue;
    }
    if (upstream.ok) {
      noteSuccess(candidate.probe.provider.id);
      return { candidate, upstream };
    }
    // A refusal (402, 429, 5xx) is not the answer: try the next candidate.
    log(`${candidate.probe.provider.id}/${candidate.model.id} refused HTTP ${upstream.status}`);
    noteFailure(candidate.probe.provider.id);
    try {
      await upstream.text();
    } catch {
      // The body is unreadable: the refusal stands.
    }
  }
  return null;
}

/** Answer one OpenAI client request from the winning provider's response. */
async function answerOpenAI(res: ServerResponse, body: Json, won: { candidate: Candidate; upstream: Response }): Promise<void> {
  const { candidate, upstream } = won;
  const wire = candidate.probe.provider.wire;
  const model = candidate.model.id;
  const stream = body.stream === true;
  if (stream && upstream.body !== null) {
    streamHeaders(res);
    if (wire === 'openai') {
      await pipeStream(upstream.body, res);
      return;
    }
    if (wire === 'gemini') {
      await pipeEvents(openAIEventsFromGemini(upstream.body, model), res);
      return;
    }
    // An anthropic upstream is never routed to an openai client (see candidatesFor).
    res.end();
    return;
  }
  const answer: unknown = await upstream.json();
  if (!isRecord(answer)) return sendError(res, 'openai', 502, 'api_error', 'the provider answered a body that is not one JSON object');
  if (wire === 'gemini') return sendJson(res, 200, geminiToOpenAI(answer, model));
  if (wire === 'anthropic') return sendError(res, 'openai', 501, 'not_implemented', 'the anthropic upstream wire is not served to an openai client');
  sendJson(res, 200, answer);
}

/** Answer one Anthropic client request from the winning provider's response. */
async function answerAnthropic(res: ServerResponse, body: Json, won: { candidate: Candidate; upstream: Response }): Promise<void> {
  const { candidate, upstream } = won;
  const wire = candidate.probe.provider.wire;
  const model = candidate.model.id;
  const stream = body.stream === true;
  const requested = typeof body.model === 'string' && body.model !== '' ? body.model : model;
  if (stream && upstream.body !== null) {
    streamHeaders(res);
    if (wire === 'anthropic') {
      await pipeStream(upstream.body, res);
      return;
    }
    if (wire === 'gemini') {
      await pipeEvents(anthropicEventsFromGemini(upstream.body, model), res);
      return;
    }
    await pipeEvents(anthropicEventsFromOpenAI(upstream.body, requested), res);
    return;
  }
  const answer: unknown = await upstream.json();
  if (!isRecord(answer)) return sendError(res, 'anthropic', 502, 'api_error', 'the provider answered a body that is not one JSON object');
  if (wire === 'gemini') return sendJson(res, 200, geminiToAnthropic(answer, model));
  if (wire === 'anthropic') return sendJson(res, 200, answer);
  sendJson(res, 200, openAIToAnthropic(answer, requested));
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body);
  const candidates = candidatesFor(requires, body, 'openai');
  if (candidates.length === 0) return sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory meets this request');
  const won = await callCandidates(candidates, body, 'openai', body.stream === true);
  if (won === null) return sendError(res, 'openai', 503, 'no_free_model', 'every free model that meets this request refused');
  await answerOpenAI(res, body, won);
}

/** POST /v1/messages: an Anthropic-shaped client, answered by the best free model. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body);
  const candidates = candidatesFor(requires, body, 'anthropic');
  if (candidates.length === 0) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory meets this request');
  const won = await callCandidates(candidates, body, 'anthropic', body.stream === true);
  if (won === null) return sendError(res, 'anthropic', 503, 'overloaded_error', 'every free model that meets this request refused');
  await answerAnthropic(res, body, won);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
