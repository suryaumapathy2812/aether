"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useSession } from "@/lib/auth-client";

/**
 * Home — redirects to chat (the primary surface).
 */
export default function HomePage() {
  const router = useRouter();
  const { data: session, isPending } = useSession();

  useEffect(() => {
    if (isPending) return;
    if (!session) {
      router.push("/");
    } else {
      router.replace("/chat");
    }
  }, [session, isPending, router]);

  return null;
}
