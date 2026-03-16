"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import ContentShell from "@/components/ContentShell";
import MinimalInput from "@/components/MinimalInput";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import PushOptIn from "@/components/PushOptIn";
import ModelPreference from "@/components/ModelPreference";
import { useSession, signOut } from "@/lib/auth-client";
import { useUIPreferences } from "@/lib/ui-preferences";

/**
 * Account — profile info, edit, log out, settings.
 */
export default function AccountPage() {
  const router = useRouter();
  const { data: session, isPending } = useSession();
  const { dockBehavior, setDockBehavior } = useUIPreferences();
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [view, setView] = useState<"menu" | "edit">("menu");

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

  if (view === "edit") {
    return (
      <ContentShell title="Edit Account" back="/account">
        <div className="w-full max-w-[300px] mx-auto">
          <MinimalInput label="Name" value={name} onChange={setName} />
          <MinimalInput
            label="Email"
            type="email"
            value={email}
            onChange={setEmail}
          />
          <Button
            variant="aether"
            size="aether"
            className="w-full mt-2"
          >
            save
          </Button>
          <Button
            variant="aether-link"
            size="aether-link"
            onClick={() => setView("menu")}
            className="w-full text-center mt-6"
          >
            cancel
          </Button>
        </div>
      </ContentShell>
    );
  }

  return (
    <ContentShell title={name || "Account"} back="/home">
      <div className="w-full max-w-[300px] mx-auto flex flex-col items-center gap-6">
        
        {/* Profile Actions */}
        <div className="w-full space-y-2">
          <Button
            variant="aether-menu"
            size="aether-menu"
            onClick={() => setView("edit")}
            className="w-full justify-center text-center"
          >
            Edit Profile
          </Button>
          <Button
            variant="aether-menu"
            size="aether-menu"
            onClick={handleLogout}
            className="w-full justify-center text-center text-red-300 hover:text-red-200"
          >
            Log Out
          </Button>
        </div>

        <Separator className="w-full opacity-30" />

        {/* Dock Settings */}
        <div className="w-full px-1">
          <p className="text-[10px] uppercase tracking-[0.15em] text-muted-foreground mb-3 text-center">
            Dock Behavior
          </p>
          <div className="flex gap-2">
            <button
              onClick={() => setDockBehavior("shrink")}
              className={`flex-1 py-2 rounded-lg text-xs transition-colors border ${
                dockBehavior === "shrink"
                  ? "bg-white/10 border-white/20 text-foreground"
                  : "bg-transparent border-transparent text-muted-foreground hover:bg-white/5"
              }`}
            >
              Shrink
            </button>
            <button
              onClick={() => setDockBehavior("auto-hide")}
              className={`flex-1 py-2 rounded-lg text-xs transition-colors border ${
                dockBehavior === "auto-hide"
                  ? "bg-white/10 border-white/20 text-foreground"
                  : "bg-transparent border-transparent text-muted-foreground hover:bg-white/5"
              }`}
            >
              Auto-Hide
            </button>
          </div>
        </div>

        <Separator className="w-full opacity-30" />

        {/* Push Notifications */}
        <div className="w-full px-1">
          <PushOptIn />
        </div>

        <Separator className="w-full opacity-30" />

        {/* Model Preference */}
        <div className="w-full px-1">
          <ModelPreference />
        </div>

      </div>
    </ContentShell>
  );
}
