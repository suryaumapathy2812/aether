import { NextRequest, NextResponse } from "next/server";

export const runtime = "edge";

const ORCHESTRATOR_BASE_URL = (
  process.env.ORCHESTRATOR_BASE_URL || "http://localhost:4000"
).replace(/\/$/, "");

const PUBLIC_PREFIX = "/agent/v1";

async function proxy(request: NextRequest, params: { path?: string[] }) {
  const segments = params.path || [];
  const upstreamPath =
    segments.length > 0 ? `${PUBLIC_PREFIX}/${segments.join("/")}` : PUBLIC_PREFIX;
  const url = new URL(`${ORCHESTRATOR_BASE_URL}${upstreamPath}`);
  request.nextUrl.searchParams.forEach((value, key) => {
    url.searchParams.append(key, value);
  });

  const headers = new Headers();
  const contentType = request.headers.get("content-type");
  if (contentType) headers.set("content-type", contentType);
  const authorization = request.headers.get("authorization");
  if (authorization) headers.set("authorization", authorization);
  const cookie = request.headers.get("cookie");
  if (cookie) headers.set("cookie", cookie);
  const origin = request.headers.get("origin");
  if (origin) headers.set("origin", origin);
  const referer = request.headers.get("referer");
  if (referer) headers.set("referer", referer);
  const forwardedHost = request.headers.get("x-forwarded-host");
  const forwardedProto = request.headers.get("x-forwarded-proto");
  headers.set(
    "x-forwarded-host",
    forwardedHost || request.headers.get("host") || request.nextUrl.host,
  );
  headers.set(
    "x-forwarded-proto",
    (forwardedProto || request.nextUrl.protocol.replace(":", "")).split(",")[0].trim(),
  );

  const method = request.method.toUpperCase();
  const hasBody = method !== "GET" && method !== "HEAD";

  try {
    const response = await fetch(url.toString(), {
      method,
      headers,
      body: hasBody ? await request.text() : undefined,
      cache: "no-store",
      redirect: "manual",
    });

    const responseHeaders = new Headers();
    const upstreamType = response.headers.get("content-type");
    if (upstreamType) responseHeaders.set("content-type", upstreamType);
    const isSSE = upstreamType && upstreamType.toLowerCase().startsWith("text/event-stream");
    if (isSSE) {
      responseHeaders.set("cache-control", "no-cache, no-transform");
      responseHeaders.set("connection", "keep-alive");
      responseHeaders.set("x-accel-buffering", "no");
      responseHeaders.set("transfer-encoding", "chunked");
    }
    const location = response.headers.get("location");
    if (location) responseHeaders.set("location", location);
    if (!location && response.redirected && response.url) {
      responseHeaders.set("x-upstream-final-url", response.url);
    }
    const setCookie = response.headers.get("set-cookie");
    if (setCookie) responseHeaders.set("set-cookie", setCookie);

    if (isSSE && response.body) {
      const { readable, writable } = new TransformStream();
      const upstream = response.body;
      const writer = writable.getWriter();

      (async () => {
        const reader = upstream.getReader();
        try {
          while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            await writer.write(value);
          }
        } catch {
          // upstream closed
        } finally {
          try {
            writer.close();
          } catch {
            /* already closed */
          }
        }
      })();

      return new Response(readable, {
        status: response.status,
        statusText: response.statusText,
        headers: responseHeaders,
      });
    }

    return new Response(response.body, {
      status: response.status,
      statusText: response.statusText,
      headers: responseHeaders,
    });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "upstream connection failed";
    const code =
      err instanceof Error &&
      "cause" in err &&
      typeof err.cause === "object" &&
      err.cause !== null &&
      "code" in err.cause &&
      (err.cause as { code: string }).code === "ECONNREFUSED"
        ? "ECONNREFUSED"
        : "UPSTREAM_ERROR";

    console.error(`[agent/v1 proxy] ${method} ${url.toString()} failed: ${code} — ${message}`);

    return NextResponse.json(
      {
        error: "upstream_unavailable",
        detail: `Could not reach upstream at ${ORCHESTRATOR_BASE_URL}. Is the orchestrator running?`,
        code,
      },
      { status: 502 },
    );
  }
}

export async function GET(request: NextRequest, context: { params: Promise<{ path?: string[] }> }) {
  return proxy(request, await context.params);
}

export async function POST(
  request: NextRequest,
  context: { params: Promise<{ path?: string[] }> },
) {
  return proxy(request, await context.params);
}

export async function PUT(request: NextRequest, context: { params: Promise<{ path?: string[] }> }) {
  return proxy(request, await context.params);
}

export async function PATCH(
  request: NextRequest,
  context: { params: Promise<{ path?: string[] }> },
) {
  return proxy(request, await context.params);
}

export async function DELETE(
  request: NextRequest,
  context: { params: Promise<{ path?: string[] }> },
) {
  return proxy(request, await context.params);
}
