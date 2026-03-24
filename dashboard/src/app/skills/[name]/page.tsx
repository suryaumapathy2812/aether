"use client";

import { useEffect, useState } from "react";
import { useRouter, useParams } from "next/navigation";
import ContentShell from "@/components/ContentShell";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { useSession } from "@/lib/auth-client";
import {
  installSkill,
  listInstalledSkills,
  removeSkill,
  searchMarketplaceSkills,
  getSkillContent,
  fetchMarketplaceSkillContent,
} from "@/lib/api";
import { toast } from "sonner";
import { IconExternalLink } from "@tabler/icons-react";
import { MessageResponse } from "@/components/ai-elements/message";

interface SkillData {
  name: string;
  description: string;
  repoUrl: string;
  source: string;
  installs: number;
  isInstalled: boolean;
  skill_id?: string;
}

function toRepoUrl(source: string): string {
  if (!source) return "";
  if (source.startsWith("http")) return source;
  if (source.includes("/")) return `https://github.com/${source.replace(/^github\.com\//, "")}`;
  return `https://github.com/${source}`;
}

function formatRepoLabel(source: string): string {
  if (!source) return "";
  const clean = source.replace(/^https?:\/\//, "").replace(/^github\.com\//, "");
  return `github.com/${clean}`;
}

function parseFrontmatter(raw: string): { description: string; body: string } {
  const lines = raw.split("\n");
  if (lines[0]?.trim() !== "---") return { description: "", body: raw };

  let end = -1;
  for (let i = 1; i < lines.length; i++) {
    if (lines[i].trim() === "---") {
      end = i;
      break;
    }
  }
  if (end === -1) return { description: "", body: raw };

  const fmLines = lines.slice(1, end);
  let description = "";
  for (const line of fmLines) {
    const match = line.match(/^description:\s*(.+)/);
    if (match) {
      description = match[1].replace(/^["']|["']$/g, "");
      break;
    }
  }

  const body = lines.slice(end + 1).join("\n").trim();
  return { description, body };
}

export default function SkillDetailsPage() {
  const router = useRouter();
  const params = useParams();
  const skillName = decodeURIComponent(params.name as string);
  const { data: session, isPending } = useSession();

  const [skill, setSkill] = useState<SkillData | null>(null);
  const [content, setContent] = useState<string>("");
  const [loading, setLoading] = useState(true);
  const [loadingContent, setLoadingContent] = useState(false);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (isPending) return;
    if (!session) {
      router.push("/");
      return;
    }
    void loadSkill();
  }, [session, isPending, router, skillName]);

  async function loadSkill() {
    setLoading(true);
    setContent("");
    try {
      const [installedData, marketplaceData] = await Promise.all([
        listInstalledSkills(),
        searchMarketplaceSkills(skillName, 20),
      ]);

      const installed = (installedData.skills || []).find(
        (s) => s.name === skillName
      );
      const marketplace = (marketplaceData.skills || []).find(
        (s) => s.name === skillName
      );

      if (installed) {
        setSkill({
          name: installed.name,
          description: installed.description || "",
          repoUrl: "",
          source: installed.source || "unknown",
          installs: 0,
          isInstalled: true,
        });
        await loadInstalledContent(skillName);
      } else if (marketplace) {
        setSkill({
          name: marketplace.name,
          description: "",
          repoUrl: marketplace.source,
          source: marketplace.source,
          installs: marketplace.installs,
          isInstalled: false,
          skill_id: marketplace.skill_id,
        });
        await loadMarketplaceContent(marketplace.source, marketplace.skill_id);
      } else {
        setSkill(null);
      }
    } catch {
      toast.error("Failed to load skill details");
    } finally {
      setLoading(false);
    }
  }

  async function loadInstalledContent(name: string) {
    setLoadingContent(true);
    try {
      const data = await getSkillContent(name);
      setContent(data.content || "");
    } catch {
      // Content not available
    } finally {
      setLoadingContent(false);
    }
  }

  async function loadMarketplaceContent(source: string, skillId: string) {
    setLoadingContent(true);
    try {
      const data = await fetchMarketplaceSkillContent(source, skillId);
      setContent(data.content || "");
    } catch {
      // Content not available
    } finally {
      setLoadingContent(false);
    }
  }

  async function handleInstall() {
    if (!skill || !skill.skill_id) return;
    setBusy(true);
    try {
      await installSkill(skill.source, skill.skill_id);
      toast.success(`${skill.name} installed`);
      await loadSkill();
    } catch {
      toast.error("Install failed");
    } finally {
      setBusy(false);
    }
  }

  async function handleRemove() {
    if (!skill) return;
    setBusy(true);
    try {
      await removeSkill(skill.name);
      toast.success(`${skill.name} removed`);
      await loadSkill();
    } catch {
      toast.error("Remove failed");
    } finally {
      setBusy(false);
    }
  }

  if (isPending || !session) return null;

  const parsed = parseFrontmatter(content);
  const description = skill?.description || parsed.description;

  return (
    <ContentShell title={skillName} back="/skills">
      {loading ? (
        <p className="text-muted-foreground text-sm">Loading...</p>
      ) : !skill ? (
        <div className="py-12 text-center">
          <p className="text-muted-foreground text-sm">Skill not found.</p>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => router.push("/skills")}
            className="mt-3"
          >
            Browse skills
          </Button>
        </div>
      ) : (
        <div className="flex flex-col gap-6">
          {/* Status + Action */}
          <div className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-2">
              <Badge variant="secondary">
                {skill.isInstalled ? "Installed" : "Marketplace"}
              </Badge>
              {skill.installs > 0 && (
                <Badge variant="outline">
                  {skill.installs.toLocaleString()} installs
                </Badge>
              )}
            </div>

            {skill.isInstalled ? (
              <Button
                variant="outline"
                size="sm"
                onClick={handleRemove}
                disabled={busy}
              >
                {busy ? "..." : "Remove"}
              </Button>
            ) : (
              <Button
                size="sm"
                onClick={handleInstall}
                disabled={busy}
              >
                {busy ? "..." : "Install"}
              </Button>
            )}
          </div>

          {/* Source */}
          {skill.repoUrl && (
            <div>
              <a
                href={toRepoUrl(skill.repoUrl)}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1.5 text-sm text-foreground hover:text-foreground/70 transition-colors"
              >
                {formatRepoLabel(skill.repoUrl)}
                <IconExternalLink className="size-3.5 text-muted-foreground" />
              </a>
            </div>
          )}

          {/* Description */}
          {description && (
            <>
              <Separator />
              <div>
                <h3 className="text-sm font-medium text-foreground mb-2">Summary</h3>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  {description}
                </p>
              </div>
            </>
          )}

          {/* SKILL.md content */}
          {content && (
            <>
              <Separator />
              <div>
                <h3 className="text-sm font-medium text-foreground mb-4">SKILL.md</h3>
                <div className="text-sm leading-relaxed">
                  <MessageResponse>
                    {parsed.body}
                  </MessageResponse>
                </div>
              </div>
            </>
          )}

          {loadingContent && (
            <p className="text-sm text-muted-foreground">Loading content...</p>
          )}
        </div>
      )}
    </ContentShell>
  );
}
