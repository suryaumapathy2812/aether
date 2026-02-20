"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import PageShell from "@/components/PageShell";
import MinimalInput from "@/components/MinimalInput";
import { Button } from "@/components/ui/button";
import { useSession, signOut } from "@/lib/auth-client";

/**
 * Account — profile info, edit, log out.
 */
export default function AccountPage() {
  const router = useRouter();
  const { data: session, isPending } = useSession();
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
      <PageShell title="Edit Account" back="/account">
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
      </PageShell>
    );
  }

  return (
    <PageShell title={name || "Account"} back="/home" centered>
      <div className="w-full max-w-[300px] mx-auto flex flex-col items-center">
        <Button
          variant="aether-menu"
          size="aether-menu"
          onClick={() => setView("edit")}
          className="text-[15px] py-4 justify-center text-center"
        >
          Edit Account
        </Button>
        <Button
          variant="aether-menu"
          size="aether-menu"
          onClick={handleLogout}
          className="text-[15px] py-4 border-b justify-center text-center"
        >
          Log Out
        </Button>
      </div>
    </PageShell>
  );
}
