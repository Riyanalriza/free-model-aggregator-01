// The Gemini upstream wire: request and response translation.
//
// The Gemini provider speaks the real Gemini API at /v1beta/models/<model>:generateContent
// (and :streamGenerateContent for a stream), with the key in x-goog-api-key. The client
// speaks OpenAI or Anthropic, so this file translates both ways. Everything here is a pure
// function of its arguments: no clock and no randomness, so a replay gives the same bytes.
//
// The provider's parser (the env's geminiParsed) reads exactly these fields, and the client
// compares the provider's echo of them with what it sent, so a translation must carry each
// one across faithfully or the request loses its wire checks:
//   systemInstruction.parts[].text        <- the system prompt
//   contents[].parts[].text               <- the user text (the last non-model content)
//   contents[].parts[].inlineData.data    <- an image, base64 bytes, or fileData.fileUri
//   tools[].functionDeclarations[].name   <- the tool names
//   generationConfig.maxOutputTokens      <- max_tokens
//   generationConfig.stopSequences        <- the stop sequences
//   generationConfig.responseMimeType     <- "application/json" for JSON mode
// The provider answers text only (the ANSWER echo), with finishReason STOP, and usage in
// usageMetadata. It never returns a tool call, so the response translation is text + usage
// + stop reason only.

import { isRecord, type Json } from './providers.ts';
import { sseEvent, stopReasonOf } from './wire.ts';

/** A Gemini finishReason as an OpenAI finish_reason. */
function openAIFinishOf(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'length';
  if (reason === 'SAFETY') return 'content_filter';
  if (reason === 'RECITATION') return 'content_filter';
  return 'stop';
}

/** A Gemini finishReason as an Anthropic stop_reason. */
function anthropicStopOf(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'max_tokens';
  if (reason === 'SAFETY') return 'refusal';
  if (reason === 'RECITATION') return 'refusal';
  return 'end_turn';
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** The parts of one OpenAI message's content as Gemini parts: text, and an image as inlineData. */
function geminiPartsOfOpenAI(content: unknown): Json[] {
  if (typeof content === 'string') return content === '' ? [] : [{ text: content }];
  if (!Array.isArray(content)) return [];
  const parts: Json[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') {
      parts.push({ text: block.text });
    } else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
      const url = block.image_url.url;
      const data = /^data:([^,]*);base64,(.*)$/s.exec(url);
      if (data !== null) {
        parts.push({ inlineData: { mimeType: data[1] === '' ? 'image/png' : data[1], data: data[2] } });
      } else {
        parts.push({ fileData: { fileUri: url } });
      }
    }
  }
  return parts;
}

/** The parts of one Anthropic message's content blocks as Gemini parts. */
function geminiPartsOfAnthropic(content: unknown): Json[] {
  if (typeof content === 'string') return content === '' ? [] : [{ text: content }];
  if (!Array.isArray(content)) return [];
  const parts: Json[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') {
      parts.push({ text: block.text });
    } else if (block.type === 'image' && isRecord(block.source)) {
      const source = block.source;
      if (source.type === 'base64' && typeof source.data === 'string') {
        parts.push({ inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } });
      } else if (source.type === 'url' && typeof source.url === 'string') {
        parts.push({ fileData: { fileUri: source.url } });
      }
    } else if (block.type === 'tool_result') {
      const text = textOf(block.content);
      parts.push({ text: text !== '' ? text : JSON.stringify(block.content ?? '') });
    }
  }
  return parts;
}

/** One OpenAI function tool as a Gemini functionDeclaration. */
function declarationOfOpenAITool(tool: Json): Json | null {
  if (!isRecord(tool.function) || typeof tool.function.name !== 'string') return null;
  return {
    name: tool.function.name,
    description: typeof tool.function.description === 'string' ? tool.function.description : '',
    parameters: isRecord(tool.function.parameters) ? tool.function.parameters : { type: 'object', properties: {} },
  };
}

/** One Anthropic tool as a Gemini functionDeclaration. */
function declarationOfAnthropicTool(tool: Json): Json | null {
  if (typeof tool.name !== 'string') return null;
  return {
    name: tool.name,
    description: typeof tool.description === 'string' ? tool.description : '',
    parameters: isRecord(tool.input_schema) ? tool.input_schema : { type: 'object', properties: {} },
  };
}

/** The generationConfig for a request: cap, stops, JSON mode, temperature and top_p. */
function generationConfigOf(opts: { maxTokens?: unknown; stops?: unknown; json?: boolean; temperature?: unknown; topP?: unknown }): Json {
  const config: Json = {};
  if (typeof opts.maxTokens === 'number') config.maxOutputTokens = opts.maxTokens;
  if (Array.isArray(opts.stops) && opts.stops.length > 0) config.stopSequences = opts.stops.filter((s): s is string => typeof s === 'string');
  if (opts.json === true) config.responseMimeType = 'application/json';
  if (typeof opts.temperature === 'number') config.temperature = opts.temperature;
  if (typeof opts.topP === 'number') config.topP = opts.topP;
  return config;
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent request. The system
 * messages become the systemInstruction, user messages become user contents (text and
 * images), assistant messages become model contents, tool messages become user contents
 * carrying the tool result text, and the function tools become functionDeclarations.
 */
export function openAIToGemini(body: Json): Json {
  const systemParts: Json[] = [];
  const contents: Json[] = [];
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    const parts = geminiPartsOfOpenAI(raw.content);
    if (raw.role === 'system') {
      systemParts.push(...parts);
      continue;
    }
    if (raw.role === 'assistant') {
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    // A user message, and a tool message whose result text the model must see.
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  const out: Json = { contents };
  if (systemParts.length > 0) out.systemInstruction = { parts: systemParts };
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw)) return [];
      const d = declarationOfOpenAITool(raw);
      return d === null ? [] : [d];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  const config = generationConfigOf({
    maxTokens: typeof body.max_tokens === 'number' ? body.max_tokens : body.max_completion_tokens,
    stops: body.stop,
    json: format === 'json_object' || format === 'json_schema',
    temperature: body.temperature,
    topP: body.top_p,
  });
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/**
 * An Anthropic Messages request as a Gemini generateContent request. The system field
 * becomes the systemInstruction, user content blocks become user contents (text and
 * images, tool results as text), assistant blocks become model contents, and the tools
 * become functionDeclarations. Anthropic has no response_format, so JSON mode is never
 * set from this wire.
 */
export function anthropicToGemini(body: Json): Json {
  const systemParts: Json[] = [];
  const systemText = textOf(body.system);
  if (systemText !== '') systemParts.push({ text: systemText });
  const contents: Json[] = [];
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    const parts = geminiPartsOfAnthropic(raw.content);
    if (parts.length === 0) continue;
    contents.push({ role: raw.role === 'assistant' ? 'model' : 'user', parts });
  }
  const out: Json = { contents };
  if (systemParts.length > 0) out.systemInstruction = { parts: systemParts };
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw)) return [];
      const d = declarationOfAnthropicTool(raw);
      return d === null ? [] : [d];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const config = generationConfigOf({
    maxTokens: body.max_tokens,
    stops: body.stop_sequences,
    json: false,
    temperature: body.temperature,
    topP: body.top_p,
  });
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** The text of a Gemini candidate's content parts, joined. */
function geminiTextOf(body: Json): string {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : null;
  if (first === null) return '';
  const content = isRecord(first.content) ? first.content : null;
  if (content === null || !Array.isArray(content.parts)) return '';
  return content.parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
}

/** The usage of a Gemini answer: prompt and candidate token counts. */
function geminiUsageOf(body: Json): { input: number; output: number } {
  const meta = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return { input: numberOrZero(meta.promptTokenCount), output: numberOrZero(meta.candidatesTokenCount) };
}

/** The finish reason of a Gemini answer's first candidate. */
function geminiFinishOf(body: Json): unknown {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : null;
  return first === null ? null : first.finishReason ?? null;
}

/** A Gemini generateContent answer as an OpenAI Chat Completions object. */
export function geminiToOpenAI(body: Json, model: string): Json {
  const usage = geminiUsageOf(body);
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: 1767225600,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: geminiTextOf(body) }, finish_reason: openAIFinishOf(geminiFinishOf(body)), logprobs: null }],
    usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
  };
}

/** A Gemini generateContent answer as an Anthropic Messages object. */
export function geminiToAnthropic(body: Json, model: string): Json {
  const usage = geminiUsageOf(body);
  const text = geminiTextOf(body);
  return {
    id: 'msg_sim',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text }],
    stop_reason: anthropicStopOf(geminiFinishOf(body)),
    stop_sequence: null,
    usage: { input_tokens: usage.input, output_tokens: usage.output },
  };
}

/** The endpoint for a Gemini call: generateContent, or streamGenerateContent for a stream. */
export function geminiEndpoint(base: string, model: string, stream: boolean): string {
  const path = stream ? ':streamGenerateContent' : ':generateContent';
  return `${base}/v1beta/models/${encodeURIComponent(model)}${path}`;
}

/** One SSE frame's data lines as a parsed JSON value, or null. */
function frameJson(frame: string): Json | null {
  for (const line of frame.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed.startsWith('data:')) continue;
    const payload = trimmed.slice(5).trim();
    if (payload === '' || payload === '[DONE]') return null;
    try {
      const value: unknown = JSON.parse(payload);
      return isRecord(value) ? value : null;
    } catch {
      return null;
    }
  }
  return null;
}

/**
 * A Gemini :streamGenerateContent SSE stream as OpenAI Chat Completions chunks: an opening
 * chunk with the assistant role, one content chunk a text piece, a finish chunk with the
 * finish_reason, a usage chunk when the provider sent usageMetadata, then [DONE]. `model`
 * is the model the client asked for, named back to it.
 */
export async function* openAIEventsFromGemini(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let opened = false;
  let finished = false;
  let usage: { input: number; output: number } | null = null;
  let finish: unknown = null;
  const chunk = (delta: Json, finishReason: string | null): string => {
    return `data: ${JSON.stringify({ id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: 1767225600, model, choices: [{ index: 0, delta, finish_reason: finishReason }] })}\n\n`;
  };
  const openEvents = (): string[] => {
    if (opened) return [];
    opened = true;
    return [chunk({ role: 'assistant', content: '' }, null)];
  };
  const closeEvents = (): string[] => {
    if (finished) return [];
    finished = true;
    const out: string[] = [chunk({}, openAIFinishOf(finish))];
    if (usage !== null) {
      out.push(
        `data: ${JSON.stringify({ id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: 1767225600, model, choices: [], usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output } })}\n\n`,
      );
    }
    out.push('data: [DONE]\n\n');
    return out;
  };
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        const parsed = frameJson(frame);
        if (parsed !== null) {
          const text = geminiTextOf(parsed);
          const meta = geminiUsageOf(parsed);
          if (meta.input !== 0 || meta.output !== 0) usage = meta;
          const f = geminiFinishOf(parsed);
          if (f !== null && f !== undefined) finish = f;
          if (text !== '') {
            for (const event of openEvents()) yield event;
            yield chunk({ content: text }, null);
          }
        }
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') {
      const parsed = frameJson(buffer);
      if (parsed !== null) {
        const text = geminiTextOf(parsed);
        const meta = geminiUsageOf(parsed);
        if (meta.input !== 0 || meta.output !== 0) usage = meta;
        const f = geminiFinishOf(parsed);
        if (f !== null && f !== undefined) finish = f;
        if (text !== '') {
          for (const event of openEvents()) yield event;
          yield chunk({ content: text }, null);
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
  for (const event of openEvents()) yield event;
  for (const event of closeEvents()) yield event;
}

/**
 * A Gemini :streamGenerateContent SSE stream as the Anthropic event sequence:
 * message_start, content_block_start, content_block_delta per text piece,
 * content_block_stop, message_delta with the stop reason and usage, message_stop.
 */
export async function* anthropicEventsFromGemini(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  let textIndex = -1;
  let inputTokens = 0;
  let outputTokens = 0;
  let finish: unknown = null;
  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: { id: 'msg_stream', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: inputTokens, output_tokens: 0 } },
      }),
    ];
  };
  const finishEvents = (): string[] => {
    if (finished) return [];
    finished = true;
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    out.push(
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: anthropicStopOf(finish), stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    );
    return out;
  };
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        const parsed = frameJson(frame);
        if (parsed !== null) {
          const text = geminiTextOf(parsed);
          const meta = geminiUsageOf(parsed);
          if (meta.input !== 0) inputTokens = meta.input;
          if (meta.output !== 0) outputTokens = meta.output;
          const f = geminiFinishOf(parsed);
          if (f !== null && f !== undefined) finish = f;
          if (text !== '') {
            for (const event of startEvents()) yield event;
            if (textIndex === -1) {
              textIndex = 0;
              yield sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } });
            }
            yield sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text } });
          }
        }
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') {
      const parsed = frameJson(buffer);
      if (parsed !== null) {
        const text = geminiTextOf(parsed);
        const meta = geminiUsageOf(parsed);
        if (meta.input !== 0) inputTokens = meta.input;
        if (meta.output !== 0) outputTokens = meta.output;
        const f = geminiFinishOf(parsed);
        if (f !== null && f !== undefined) finish = f;
        if (text !== '') {
          for (const event of startEvents()) yield event;
          if (textIndex === -1) {
            textIndex = 0;
            yield sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } });
          }
          yield sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text } });
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
  for (const event of startEvents()) yield event;
  for (const event of finishEvents()) yield event;
}

/** Re-export the OpenAI stop mapping for the main server's use. */
export { stopReasonOf };
