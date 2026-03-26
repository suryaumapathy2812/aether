import { createFileRoute } from "@tanstack/react-router";

export const Route = createFileRoute("/api/skills/content")({
  server: {
    handlers: {
      GET: async ({ request }: { request: Request }) => {
        const { searchParams } = new URL(request.url);
        const source = searchParams.get("source");
        const skillId = searchParams.get("skillId");

        if (!source || !skillId) {
          return Response.json(
            { content: "", error: "source and skillId are required" },
            { status: 400 }
          );
        }

        const clean = source
          .replace(/^https?:\/\/github\.com\//, "")
          .replace(/^github\.com\//, "");

        const url = `https://raw.githubusercontent.com/${clean}/main/skills/${encodeURIComponent(skillId)}/SKILL.md`;

        try {
          const res = await fetch(url, {
            next: { revalidate: 3600 },
          });

          if (!res.ok) {
            return Response.json({ content: "" });
          }

          const content = await res.text();
          return Response.json({ content });
        } catch {
          return Response.json({ content: "" });
        }
      },
    },
  },
} as any);
