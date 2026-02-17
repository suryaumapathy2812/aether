"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import MinimalInput from "@/components/MinimalInput";
import { login, signup } from "@/lib/api";

/**
 * Login / Sign Up — centered form, dark bg, minimal.
 */
export default function LoginPage() {
  const router = useRouter();
  const [mode, setMode] = useState<"login" | "signup">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      if (mode === "signup") {
        await signup(email, password, name);
      } else {
        await login(email, password);
      }
      router.push("/home");
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-6">
      {/* Brand */}
      <h1 className="text-xs tracking-[0.35em] text-[var(--color-text-muted)] italic font-light mb-16">
        aether
      </h1>

      <form onSubmit={handleSubmit} className="w-full max-w-[280px]">
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
          <p className="text-[var(--color-text-muted)] text-xs mb-4 animate-[fade-in_0.2s_ease]">
            {error}
          </p>
        )}

        <button
          type="submit"
          disabled={loading}
          className="btn w-full mt-2 disabled:opacity-30 disabled:cursor-not-allowed"
        >
          {loading
            ? "..."
            : mode === "login"
              ? "sign in"
              : "create account"}
        </button>

        <button
          type="button"
          onClick={() => {
            setMode(mode === "login" ? "signup" : "login");
            setError("");
          }}
          className="w-full text-center text-xs text-[var(--color-text-muted)] mt-8 hover:text-[var(--color-text-secondary)] transition-colors duration-300"
        >
          {mode === "login"
            ? "don't have an account?"
            : "already have an account?"}
        </button>
      </form>
    </div>
  );
}
