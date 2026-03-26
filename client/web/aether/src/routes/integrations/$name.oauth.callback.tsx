import { createFileRoute } from "@tanstack/react-router";
import { useEffect } from "react";
import { useNavigate, useParams } from "@tanstack/react-router";
import { useSession } from "#/lib/auth-client";
import { fetchWithAuth } from "#/lib/api";
import { z } from "zod";

const oauthCallbackSearchSchema = z.object({
  code: z.string().optional().catch(undefined),
  state: z.string().optional().catch(undefined),
  error: z.string().optional().catch(undefined),
});

export const Route = createFileRoute("/integrations/$name/oauth/callback")({
  validateSearch: oauthCallbackSearchSchema,
  component: PluginOAuthCallbackPage,
});

function PluginOAuthCallbackPage() {
  const navigate = useNavigate();
  const { name: integrationName } = useParams({ from: "/integrations/$name/oauth/callback" });
  const { code = "", state = "", error: oauthError = "" } = Route.useSearch();
  const { data: session, isPending } = useSession();

  useEffect(() => {
    if (isPending) return;
    if (!session) {
      navigate({ to: "/", replace: true });
      return;
    }

    if (oauthError) {
      navigate({
        to: "/integrations/$name",
        params: { name: integrationName },
        search: { error: oauthError },
        replace: true,
      });
      return;
    }

    if (!code) {
      navigate({
        to: "/integrations/$name",
        params: { name: integrationName },
        search: { error: "missing_code" },
        replace: true,
      });
      return;
    }

    const run = async () => {
      try {
        const res = await fetchWithAuth(`/agent/v1/integrations/${integrationName}/oauth/callback`, {
          method: "POST",
          body: JSON.stringify({ code, state }),
        });
        if (!res.ok) {
          const body = await res.json().catch(() => ({}));
          const msg =
            (typeof body?.error === "string" && body.error) ||
            (typeof body?.detail === "string" && body.detail) ||
            "oauth_callback_failed";
          navigate({
            to: "/integrations/$name",
            params: { name: integrationName },
            search: { error: msg },
            replace: true,
          });
          return;
        }
        navigate({
          to: "/integrations/$name",
          params: { name: integrationName },
          search: { connected: "true" },
          replace: true,
        });
      } catch (error: unknown) {
        const msg = error instanceof Error ? error.message : "oauth_callback_failed";
        navigate({
          to: "/integrations/$name",
          params: { name: integrationName },
          search: { error: msg },
          replace: true,
        });
      }
    };

    void run();
  }, [session, isPending, navigate, integrationName, code, state, oauthError]);

  return (
    <main className="min-h-screen w-full flex items-center justify-center px-6">
      <p className="text-xs text-muted-foreground tracking-wider">Completing OAuth connection...</p>
    </main>
  );
}
