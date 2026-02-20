"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import MinimalInput from "@/components/MinimalInput";
import { Button } from "@/components/ui/button";
import { signIn, signUp, useSession } from "@/lib/auth-client";

/**
 * Login / Sign Up — centered form, dark bg, minimal.
 * Uses better-auth for cookie-based session management.
 */
export default function LoginPage() {
  const router = useRouter();
  const { data: session, isPending } = useSession();
  const [mode, setMode] = useState<"login" | "signup">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  // If already logged in, redirect to home
  useEffect(() => {
    if (!isPending && session) {
      router.push("/home");
    }
  }, [session, isPending, router]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      if (mode === "signup") {
        const { error } = await signUp.email({
          email,
          password,
          name: name || email.split("@")[0],
        });
        if (error) throw new Error(error.message || "Signup failed");
      } else {
        const { error } = await signIn.email({
          email,
          password,
        });
        if (error) throw new Error(error.message || "Login failed");
      }
      router.push("/home");
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  // Don't show form while checking session
  if (isPending) return null;

  return (
    <div className="h-full flex flex-col items-center justify-center px-6 sm:px-8">
      {/* Brand */}
      <h1 className="logo-wordmark text-xs text-muted-foreground font-medium mb-12">
        aether
      </h1>

      <form
        onSubmit={handleSubmit}
        className="w-full max-w-[340px] px-2 py-2"
      >
        {mode === "signup" && (
          <MinimalInput label="Name" value={name} onChange={setName} />
        )}
        <MinimalInput
          label="Email"
          type="email"
          value={email}
          onChange={setEmail}
        />
        <MinimalInput
          label="Password"
          type="password"
          value={password}
          onChange={setPassword}
        />

        {error && (
          <p className="text-muted-foreground text-xs mb-4 animate-[fade-in_0.2s_ease]">
            {error}
          </p>
        )}

        <Button
          type="submit"
          disabled={loading}
          variant="aether"
          size="aether"
          className="w-full mt-2"
        >
          {loading
            ? "..."
            : mode === "login"
              ? "sign in"
              : "create account"}
        </Button>

        <Button
          type="button"
          variant="aether-link"
          size="aether-link"
          onClick={() => {
            setMode(mode === "login" ? "signup" : "login");
            setError("");
          }}
          className="w-full text-center mt-8"
        >
          {mode === "login"
            ? "don't have an account?"
            : "already have an account?"}
        </Button>
      </form>
    </div>
  );
}
