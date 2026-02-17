"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import PageShell from "@/components/PageShell";
import MinimalInput from "@/components/MinimalInput";
import { listDevices, confirmPairing, isLoggedIn } from "@/lib/api";

/**
 * Devices — list paired devices + pair new ones.
 */
export default function DevicesPage() {
  const router = useRouter();
  const [devices, setDevices] = useState<
    { id: string; name: string; device_type: string }[]
  >([]);
  const [pairing, setPairing] = useState(false);
  const [code, setCode] = useState("");
  const [status, setStatus] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!isLoggedIn()) {
      router.push("/");
      return;
    }
    listDevices().then(setDevices).catch(() => {});
  }, [router]);

  async function handlePair() {
    setStatus("");
    setLoading(true);
    try {
      await confirmPairing(code);
      setStatus("paired");
      setPairing(false);
      setCode("");
      listDevices().then(setDevices);
    } catch (err: unknown) {
      setStatus(err instanceof Error ? err.message : "Pairing failed");
    } finally {
      setLoading(false);
    }
  }

  // Empty state + pairing flow use centered layout
  const showCentered = devices.length === 0 && !pairing;

  return (
    <PageShell title="Devices" back="/home" centered={showCentered}>
      {devices.length === 0 && !pairing ? (
        <div className="text-center">
          <p className="text-[var(--color-text-muted)] text-xs mb-8">
            no devices yet
          </p>
          <button onClick={() => setPairing(true)} className="btn">
            pair a device
          </button>
        </div>
      ) : pairing ? (
        <div className="w-full max-w-[280px]">
          <p className="text-[var(--color-text-muted)] text-xs mb-8 text-center tracking-wider">
            Enter the code shown on your device
          </p>
          <MinimalInput
            label="Pairing Code"
            value={code}
            onChange={setCode}
            placeholder="AETHER-XXXX-XXXX"
          />
          {status && (
            <p className="text-[var(--color-text-muted)] text-xs mb-4 text-center animate-[fade-in_0.2s_ease]">
              {status}
            </p>
          )}
          <button
            onClick={handlePair}
            disabled={loading || !code.trim()}
            className="btn w-full disabled:opacity-30 disabled:cursor-not-allowed"
          >
            {loading ? "..." : "pair"}
          </button>
          <button
            onClick={() => {
              setPairing(false);
              setCode("");
              setStatus("");
            }}
            className="w-full text-center text-xs text-[var(--color-text-muted)] mt-6 hover:text-[var(--color-text-secondary)] transition-colors duration-300"
          >
            cancel
          </button>
        </div>
      ) : (
        <div className="w-full max-w-sm">
          {devices.map((d) => (
            <div
              key={d.id}
              className="flex items-center justify-between py-4 border-b border-[var(--color-border)] last:border-b-0"
            >
              <span className="text-sm text-[var(--color-text-secondary)] font-light">
                {d.name}
              </span>
              <span className="text-[10px] text-[var(--color-text-muted)] tracking-wider">
                {d.device_type}
              </span>
            </div>
          ))}
          <button
            onClick={() => setPairing(true)}
            className="w-full text-center text-xs text-[var(--color-text-muted)] mt-8 hover:text-[var(--color-text-secondary)] transition-colors duration-300"
          >
            pair another device
          </button>
        </div>
      )}
    </PageShell>
  );
}
