"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import PageShell from "@/components/PageShell";
import MinimalInput from "@/components/MinimalInput";
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
          <button className="btn w-full mt-2">save</button>
          <button
            onClick={() => setView("menu")}
            className="w-full text-center text-xs text-[var(--color-text-muted)] mt-6 hover:text-[var(--color-text-secondary)] transition-colors duration-300"
          >
            cancel
          </button>
        </div>
      </PageShell>
    );
  }

  return (
    <PageShell title={name || "Account"} back="/home" centered>
      <div className="w-full max-w-[300px] mx-auto">
        <button
          onClick={() => setView("edit")}
          className="w-full py-4 text-[15px] text-[var(--color-text-secondary)] hover:text-[var(--color-text)] transition-colors duration-300 font-light text-center border border-[var(--color-border)] border-b-0"
        >
          Edit Account
        </button>
        <button
          onClick={handleLogout}
          className="w-full py-4 text-[15px] text-[var(--color-text-secondary)] hover:text-[var(--color-text)] transition-colors duration-300 font-light text-center border border-[var(--color-border)]"
        >
          Log Out
        </button>
      </div>
    </PageShell>
  );
}
