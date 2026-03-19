"use client";

import ContentShell from "@/components/ContentShell";
import { Sparkles } from "lucide-react";

export default function SkillsPage() {
  return (
    <ContentShell title="Skills">
      <div className="flex flex-col items-center justify-center py-24 text-center">
        <Sparkles className="size-8 text-foreground/20 mb-4" />
        <p className="text-foreground/70 text-sm font-medium">Coming soon</p>
        <p className="text-foreground/40 text-xs mt-1">
          Custom skills and automations are on the way.
        </p>
      </div>
    </ContentShell>
  );
}
