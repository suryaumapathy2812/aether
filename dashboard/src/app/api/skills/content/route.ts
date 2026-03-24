import { NextResponse } from "next/server";

export const revalidate = 3600; // 1 hour

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const source = searchParams.get("source");
  const skillId = searchParams.get("skillId");

  if (!source || !skillId) {
    return NextResponse.json(
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
      return NextResponse.json({ content: "" });
    }

    const content = await res.text();
    return NextResponse.json({ content });
  } catch {
    return NextResponse.json({ content: "" });
  }
}
