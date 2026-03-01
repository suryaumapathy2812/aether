import { NextRequest, NextResponse } from "next/server";

/**
 * Proxy route: forwards /api/go/* to the orchestrator upstream.
 *
 * Set AGENT_BASE_URL in your .env:
 *   - AGENT_BASE_URL=http://localhost:4000
 *
 * The proxy strips the /api/go prefix and forwards the rest as-is.
 */
const AGENT_BASE_URL = (
  process.env.AGENT_BASE_URL || "http://localhost:4000"
).replace(/\/$/, "");

async function proxy(request: NextRequest, params: { path?: string[] }) {
  const segments = params.path || [];
  const upstreamPath = segments.join("/");
  const url = new URL(`${AGENT_BASE_URL}/${upstreamPath}`);
  request.nextUrl.searchParams.forEach((value, key) => {
    url.searchParams.append(key, value);
  });

  const headers = new Headers();
  const contentType = request.headers.get("content-type");
  if (contentType) headers.set("content-type", contentType);
  const authorization = request.headers.get("authorization");
  if (authorization) headers.set("authorization", authorization);

  const method = request.method.toUpperCase();
  const hasBody = method !== "GET" && method !== "HEAD";

  try {
    const response = await fetch(url.toString(), {
      method,
      headers,
      body: hasBody ? await request.text() : undefined,
      cache: "no-store",
    });

    const responseHeaders = new Headers();
    const upstreamType = response.headers.get("content-type");
    if (upstreamType) responseHeaders.set("content-type", upstreamType);

    return new Response(response.body, {
      status: response.status,
      statusText: response.statusText,
      headers: responseHeaders,
    });
  } catch (err: unknown) {
    const message =
      err instanceof Error ? err.message : "upstream connection failed";
    const code =
      err instanceof Error && "cause" in err &&
      typeof err.cause === "object" && err.cause !== null &&
      "code" in err.cause && (err.cause as { code: string }).code === "ECONNREFUSED"
        ? "ECONNREFUSED"
        : "UPSTREAM_ERROR";

    console.error(
      `[api/go proxy] ${method} ${url.toString()} failed: ${code} — ${message}`
    );

    return NextResponse.json(
      {
        error: "upstream_unavailable",
        detail: `Could not reach upstream at ${AGENT_BASE_URL}. Is the orchestrator running?`,
        code,
      },
      { status: 502 }
    );
  }
}

export async function GET(request: NextRequest, context: { params: Promise<{ path?: string[] }> }) {
  return proxy(request, await context.params);
}

export async function POST(request: NextRequest, context: { params: Promise<{ path?: string[] }> }) {
  return proxy(request, await context.params);
}

export async function PUT(request: NextRequest, context: { params: Promise<{ path?: string[] }> }) {
  return proxy(request, await context.params);
}

export async function PATCH(request: NextRequest, context: { params: Promise<{ path?: string[] }> }) {
  return proxy(request, await context.params);
}

export async function DELETE(request: NextRequest, context: { params: Promise<{ path?: string[] }> }) {
  return proxy(request, await context.params);
}
