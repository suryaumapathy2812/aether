"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import PageShell from "@/components/PageShell";
import MinimalInput from "@/components/MinimalInput";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { useSession } from "@/lib/auth-client";
import { listDevices, confirmPairing } from "@/lib/api";

/**
 * Devices — list paired devices + pair new ones.
 */
export default function DevicesPage() {
  const router = useRouter();
  const { data: session, isPending } = useSession();
  const [devices, setDevices] = useState<
    { id: string; name: string; device_type: string }[]
  >([]);
  const [pairing, setPairing] = useState(false);
  const [code, setCode] = useState("");
  const [status, setStatus] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (isPending) return;
    if (!session) {
      router.push("/");
      return;
    }
    listDevices().then(setDevices).catch(() => {});
  }, [session, isPending, router]);

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

  if (isPending || !session) return null;

  // Empty state + pairing flow use centered layout
  const showCentered = devices.length === 0 && !pairing;

  return (
    <PageShell title="Devices" back="/home" centered={showCentered}>
      {devices.length === 0 && !pairing ? (
        <div className="text-center">
          <p className="text-muted-foreground text-xs mb-8">
            no devices yet
          </p>
          <Button
            variant="aether"
            size="aether"
            onClick={() => setPairing(true)}
          >
            pair a device
          </Button>
        </div>
      ) : pairing ? (
        <div className="w-full max-w-[280px]">
          <p className="text-muted-foreground text-xs mb-8 text-center tracking-wider">
            Enter the code shown on your device
          </p>
          <MinimalInput
            label="Pairing Code"
            value={code}
            onChange={setCode}
            placeholder="AETHER-XXXX-XXXX"
          />
          {status && (
            <p className="text-muted-foreground text-xs mb-4 text-center animate-[fade-in_0.2s_ease]">
              {status}
            </p>
          )}
          <Button
            variant="aether"
            size="aether"
            onClick={handlePair}
            disabled={loading || !code.trim()}
            className="w-full"
          >
            {loading ? "..." : "pair"}
          </Button>
          <Button
            variant="aether-link"
            size="aether-link"
            onClick={() => {
              setPairing(false);
              setCode("");
              setStatus("");
            }}
            className="w-full text-center mt-6"
          >
            cancel
          </Button>
        </div>
      ) : (
        <div className="w-full max-sm">
          {devices.map((d, index) => (
            <div key={d.id}>
              <div className="flex items-center justify-between py-4">
                <span className="text-sm text-secondary-foreground font-light">
                  {d.name}
                </span>
                <span className="text-[10px] text-muted-foreground tracking-wider">
                  {d.device_type}
                </span>
              </div>
              {index < devices.length - 1 && <Separator />}
            </div>
          ))}
          <Button
            variant="aether-link"
            size="aether-link"
            onClick={() => setPairing(true)}
            className="w-full text-center mt-8"
          >
            pair another device
          </Button>
        </div>
      )}
    </PageShell>
  );
}
