"use client";

import { useEffect, useMemo } from "react";
import { useParams, useRouter, useSearchParams } from "next/navigation";
import { useSession } from "@/lib/auth-client";
import { fetchWithAuth } from "@/lib/api";

export default function PluginOAuthCallbackPage() {
  const router = useRouter();
  const params = useParams();
  const searchParams = useSearchParams();
  const { data: session, isPending } = useSession();
  const pluginName = params.name as string;

  const code = useMemo(() => searchParams.get("code") || "", [searchParams]);
  const state = useMemo(() => searchParams.get("state") || "", [searchParams]);
  const oauthError = useMemo(() => searchParams.get("error") || "", [searchParams]);

  useEffect(() => {
    if (isPending) return;
    if (!session) {
      router.replace("/");
      return;
    }

    if (oauthError) {
      router.replace(`/plugins/${pluginName}?error=${encodeURIComponent(oauthError)}`);
      return;
    }

    if (!code) {
      router.replace(`/plugins/${pluginName}?error=${encodeURIComponent("missing_code")}`);
      return;
    }

    const run = async () => {
      try {
        const res = await fetchWithAuth(`/agent/v1/plugins/${pluginName}/oauth/callback`, {
          method: "POST",
          body: JSON.stringify({ code, state }),
        });
        if (!res.ok) {
          const body = await res.json().catch(() => ({}));
          const msg =
            (typeof body?.error === "string" && body.error) ||
            (typeof body?.detail === "string" && body.detail) ||
            "oauth_callback_failed";
          router.replace(`/plugins/${pluginName}?error=${encodeURIComponent(msg)}`);
          return;
        }
        router.replace(`/plugins/${pluginName}?connected=true`);
      } catch (error: unknown) {
        const msg = error instanceof Error ? error.message : "oauth_callback_failed";
        router.replace(`/plugins/${pluginName}?error=${encodeURIComponent(msg)}`);
      }
    };

    void run();
  }, [session, isPending, router, pluginName, code, state, oauthError]);

  return (
    <main className="min-h-screen w-full flex items-center justify-center px-6">
      <p className="text-xs text-muted-foreground tracking-wider">Completing OAuth connection...</p>
    </main>
  );
}
