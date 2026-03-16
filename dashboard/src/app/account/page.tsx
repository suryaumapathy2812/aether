"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import ContentShell from "@/components/ContentShell";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import PushOptIn from "@/components/PushOptIn";
import ModelPreference from "@/components/ModelPreference";
import { useSession, signOut } from "@/lib/auth-client";
import { LogOut, User, Bell, Cpu } from "lucide-react";

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
      <div className="space-y-1">
        {/* Profile */}
        <section className="rounded-lg border border-border bg-card p-5">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-9 h-9 rounded-full bg-secondary flex items-center justify-center shrink-0">
              <User className="size-4 text-muted-foreground" />
            </div>
            <div className="min-w-0">
              {editingName ? (
                <input
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  onBlur={() => setEditingName(false)}
                  onKeyDown={(e) => e.key === "Enter" && setEditingName(false)}
                  autoFocus
                  className="bg-transparent border-b border-border text-sm text-foreground outline-none pb-0.5 w-full"
                />
              ) : (
                <p
                  className="text-sm font-medium text-foreground cursor-pointer hover:text-foreground/80 transition-colors"
                  onClick={() => setEditingName(true)}
                >
                  {name || "Set name"}
                </p>
              )}
              <p className="text-xs text-muted-foreground">{email}</p>
            </div>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={handleLogout}
            className="text-red-400 hover:text-red-300 hover:bg-red-500/10 h-8 px-3 text-xs"
          >
            <LogOut className="size-3 mr-1.5" />
            Log out
          </Button>
        </section>

        <Separator className="opacity-0" />

        {/* Model */}
        <section className="rounded-lg border border-border bg-card p-5">
          <div className="flex items-center gap-2 mb-4">
            <Cpu className="size-3.5 text-muted-foreground" />
            <h3 className="text-xs font-medium text-foreground">Model</h3>
          </div>
          <ModelPreference />
        </section>

        <Separator className="opacity-0" />

        {/* Notifications */}
        <section className="rounded-lg border border-border bg-card p-5">
          <div className="flex items-center gap-2 mb-4">
            <Bell className="size-3.5 text-muted-foreground" />
            <h3 className="text-xs font-medium text-foreground">Notifications</h3>
          </div>
          <PushOptIn />
        </section>
      </div>
    </ContentShell>
  );
}
