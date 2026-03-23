import { NextRequest, NextResponse } from "next/server";

export async function GET(request: NextRequest, context: { params: Promise<{ name: string }> }) {
  const { name } = await context.params;
  const pluginName = encodeURIComponent(name);
  const forwardedHost = request.headers.get("x-forwarded-host") || request.headers.get("host");
  const forwardedProtoRaw = request.headers.get("x-forwarded-proto");
  const forwardedProto = (forwardedProtoRaw || request.nextUrl.protocol.replace(":", ""))
    .split(",")[0]
    .trim();
  const origin = forwardedHost ? `${forwardedProto}://${forwardedHost}` : request.nextUrl.origin;
  const localTarget = new URL(`/agent/v1/plugins/${pluginName}/oauth/start`, origin);
  request.nextUrl.searchParams.forEach((value, key) => {
    localTarget.searchParams.append(key, value);
  });

  const fwdHeaders = new Headers();
  const cookie = request.headers.get("cookie");
  if (cookie) fwdHeaders.set("cookie", cookie);
  const authorization = request.headers.get("authorization");
  if (authorization) fwdHeaders.set("authorization", authorization);
  fwdHeaders.set("origin", origin);
  fwdHeaders.set("referer", origin + `/plugins/${pluginName}`);
  if (forwardedHost) fwdHeaders.set("x-forwarded-host", forwardedHost);
  if (forwardedProto) fwdHeaders.set("x-forwarded-proto", forwardedProto);

  let lastError = "Could not start connection. Please try again.";
  try {
    const res = await fetch(localTarget.toString(), {
      method: "GET",
      headers: fwdHeaders,
      cache: "no-store",
      redirect: "manual",
    });
    if (res.status >= 300 && res.status < 400) {
      const location = res.headers.get("location");
      if (location) {
        return NextResponse.redirect(location, 302);
      }
    }
    const upstreamFinalUrl = res.headers.get("x-upstream-final-url");
    if (upstreamFinalUrl) {
      return NextResponse.redirect(upstreamFinalUrl, 302);
    }
    const body = await res.json().catch(() => ({}));
    const msg =
      (typeof body?.error === "string" && body.error) ||
      (typeof body?.detail === "string" && body.detail) ||
      `${res.status} ${res.statusText}`;
    if (msg) {
      const normalized = msg.toLowerCase();
      if (
        normalized.includes("missing oauth client_id") ||
        normalized.includes("missing oauth client_secret")
      ) {
        lastError = "This connection needs setup details before it can be connected.";
      } else {
        lastError = "Could not start connection. Please try again.";
      }
    }
  } catch {
    lastError = "Could not start connection. Please try again.";
  }

  const fallback = new URL(`/plugins/${pluginName}`, origin);
  fallback.searchParams.set("error", lastError);
  return NextResponse.redirect(fallback.toString(), 302);
}
