"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import ContentShell from "@/components/ContentShell";
import { Button } from "@/components/ui/button";
import PushOptIn from "@/components/PushOptIn";
import ModelPreference from "@/components/ModelPreference";
import { useSession, signOut } from "@/lib/auth-client";
import { ChevronRight, LogOut } from "lucide-react";

export default function AccountPage() {
  const router = useRouter();
  const { data: session, isPending } = useSession();
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [editingName, setEditingName] = useState(false);

  useEffect(() => {
    if (!isPending && !session) {
      router.push("/");
      return;
    }
    if (session) {
      setName(session.user.name || "");
      setEmail(session.user.email);
    }
  }, [session, isPending, router]);

  async function handleLogout() {
    await signOut();
    router.push("/");
  }

  if (isPending || !session) return null;

  return (
    <ContentShell title="Settings">
      {/* Profile */}
      <div className="flex items-center gap-4 pb-8 mb-8 border-b border-border">
        <div className="w-10 h-10 rounded-full bg-gradient-to-br from-foreground/10 to-foreground/5 flex items-center justify-center text-sm font-medium text-foreground/60 shrink-0">
          {(name || email).charAt(0).toUpperCase()}
        </div>
        <div className="min-w-0 flex-1">
          {editingName ? (
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              onBlur={() => setEditingName(false)}
              onKeyDown={(e) => e.key === "Enter" && setEditingName(false)}
              autoFocus
              className="bg-transparent text-[14px] font-medium text-foreground outline-none border-b border-foreground/20 pb-0.5 w-full"
            />
          ) : (
            <button
              className="text-[14px] font-medium text-foreground hover:text-foreground/70 transition-colors text-left"
              onClick={() => setEditingName(true)}
            >
              {name || "Add name"}
            </button>
          )}
          <p className="text-[12px] text-muted-foreground mt-0.5">{email}</p>
        </div>
        <Button
          variant="ghost"
          size="sm"
          onClick={handleLogout}
          className="text-muted-foreground hover:text-red-400 hover:bg-red-500/5 h-8 px-2.5 text-[12px] font-normal shrink-0"
        >
          <LogOut className="size-3.5 mr-1.5" />
          Log out
        </Button>
      </div>

      {/* Settings */}
      <div className="space-y-8">
        <SettingsRow
          label="Model"
          description="Override the default model for AI tasks"
        >
          <ModelPreference />
        </SettingsRow>

        <SettingsRow
          label="Notifications"
          description="Push notifications for updates"
        >
          <PushOptIn />
        </SettingsRow>

        <button
          onClick={() => router.push("/plugins")}
          className="w-full flex items-center justify-between group"
        >
          <div>
            <p className="text-[13px] text-foreground text-left">Connections</p>
            <p className="text-[11px] text-muted-foreground mt-0.5 text-left">
              Manage Gmail, Calendar, and other integrations
            </p>
          </div>
          <ChevronRight className="size-4 text-muted-foreground/40 group-hover:text-muted-foreground transition-colors shrink-0" />
        </button>

        <button
          onClick={() => router.push("/memory")}
          className="w-full flex items-center justify-between group"
        >
          <div>
            <p className="text-[13px] text-foreground text-left">Memory</p>
            <p className="text-[11px] text-muted-foreground mt-0.5 text-left">
              Facts, conversations, and decisions
            </p>
          </div>
          <ChevronRight className="size-4 text-muted-foreground/40 group-hover:text-muted-foreground transition-colors shrink-0" />
        </button>
      </div>
    </ContentShell>
  );
}

function SettingsRow({
  label,
  description,
  children,
}: {
  label: string;
  description: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <p className="text-[13px] text-foreground">{label}</p>
      <p className="text-[11px] text-muted-foreground mt-0.5">{description}</p>
      <div className="mt-3">{children}</div>
    </div>
  );
}
