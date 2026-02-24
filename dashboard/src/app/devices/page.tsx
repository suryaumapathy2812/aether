"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import PageShell from "@/components/PageShell";
import MinimalInput from "@/components/MinimalInput";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { useSession } from "@/lib/auth-client";
import {
  listDevices,
  confirmPairing,
  addTelegramDevice,
  removeDevice,
} from "@/lib/api";

type PairingMode = null | "pick" | "ios" | "telegram";

export default function DevicesPage() {
  const router = useRouter();
  const { data: session, isPending } = useSession();
  const [devices, setDevices] = useState<
    { id: string; name: string; device_type: string }[]
  >([]);
  const [mode, setMode] = useState<PairingMode>(null);
  const [status, setStatus] = useState("");
  const [loading, setLoading] = useState(false);

  // iOS pairing
  const [code, setCode] = useState("");

  // Telegram pairing
  const [botToken, setBotToken] = useState("");
  const [secretToken, setSecretToken] = useState("");
  const [allowedChatIds, setAllowedChatIds] = useState("");

  useEffect(() => {
    if (isPending) return;
    if (!session) {
      router.push("/");
      return;
    }
    listDevices().then(setDevices).catch(() => {});
  }, [session, isPending, router]);

  function resetForm() {
    setMode(null);
    setCode("");
    setBotToken("");
    setSecretToken("");
    setAllowedChatIds("");
    setStatus("");
  }

  async function handleIosPair() {
    setStatus("");
    setLoading(true);
    try {
      await confirmPairing(code);
      setStatus("paired");
      resetForm();
      listDevices().then(setDevices);
    } catch (err: unknown) {
      setStatus(err instanceof Error ? err.message : "Pairing failed");
    } finally {
      setLoading(false);
    }
  }

  async function handleTelegramConnect() {
    setStatus("");
    setLoading(true);
    try {
      const result = await addTelegramDevice({
        bot_token: botToken,
        secret_token: secretToken || undefined,
        allowed_chat_ids: allowedChatIds || undefined,
      });
      setStatus(`connected @${result.bot_username}`);
      resetForm();
      listDevices().then(setDevices);
    } catch (err: unknown) {
      setStatus(err instanceof Error ? err.message : "Connection failed");
    } finally {
      setLoading(false);
    }
  }

  async function handleRemove(deviceId: string) {
    try {
      await removeDevice(deviceId);
      setDevices((prev) => prev.filter((d) => d.id !== deviceId));
    } catch {
      // silent
    }
  }

  if (isPending || !session) return null;

  const showCentered = devices.length === 0 && mode === null;
  const hasTelegram = devices.some((d) => d.device_type === "telegram");

  return (
    <PageShell title="Devices" back="/home" centered={showCentered}>
      {/* ── Empty state ── */}
      {devices.length === 0 && mode === null && (
        <div className="text-center">
          <p className="text-muted-foreground text-xs mb-8">
            no devices connected
          </p>
          <Button
            variant="aether"
            size="aether"
            onClick={() => setMode("pick")}
          >
            connect a device
          </Button>
        </div>
      )}

      {/* ── Device type picker ── */}
      {mode === "pick" && (
        <div className="w-full max-w-[280px]">
          <p className="text-muted-foreground text-xs mb-8 text-center tracking-wider">
            choose device type
          </p>
          <div className="space-y-3">
            <Button
              variant="aether"
              size="aether"
              onClick={() => setMode("ios")}
              className="w-full"
            >
              iOS / Android
            </Button>
            {!hasTelegram && (
              <Button
                variant="aether"
                size="aether"
                onClick={() => setMode("telegram")}
                className="w-full"
              >
                Telegram Bot
              </Button>
            )}
          </div>
          <Button
            variant="aether-link"
            size="aether-link"
            onClick={resetForm}
            className="w-full text-center mt-6"
          >
            cancel
          </Button>
        </div>
      )}

      {/* ── iOS / Android pairing ── */}
      {mode === "ios" && (
        <div className="w-full max-w-[280px]">
          <p className="text-muted-foreground text-xs mb-8 text-center tracking-wider">
            enter the code shown on your device
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
            onClick={handleIosPair}
            disabled={loading || !code.trim()}
            className="w-full"
          >
            {loading ? "..." : "pair"}
          </Button>
          <Button
            variant="aether-link"
            size="aether-link"
            onClick={resetForm}
            className="w-full text-center mt-6"
          >
            cancel
          </Button>
        </div>
      )}

      {/* ── Telegram bot connection ── */}
      {mode === "telegram" && (
        <div className="w-full max-w-[280px]">
          <p className="text-muted-foreground text-xs mb-6 text-center tracking-wider">
            connect a Telegram bot
          </p>
          <p className="text-muted-foreground text-[10px] mb-6 text-center leading-relaxed">
            Create a bot with{" "}
            <a
              href="https://t.me/BotFather"
              target="_blank"
              rel="noopener noreferrer"
              className="underline"
            >
              @BotFather
            </a>{" "}
            on Telegram and paste the token below
          </p>
          <div className="space-y-4">
            <MinimalInput
              label="Bot Token"
              value={botToken}
              onChange={setBotToken}
              placeholder="123456:ABC-DEF..."
              type="password"
            />
            <MinimalInput
              label="Webhook Secret (optional)"
              value={secretToken}
              onChange={setSecretToken}
              placeholder="auto-generated if blank"
            />
            <MinimalInput
              label="Allowed Chat IDs (optional)"
              value={allowedChatIds}
              onChange={setAllowedChatIds}
              placeholder="comma-separated"
            />
          </div>
          {status && (
            <p className="text-muted-foreground text-xs mt-4 mb-2 text-center animate-[fade-in_0.2s_ease]">
              {status}
            </p>
          )}
          <Button
            variant="aether"
            size="aether"
            onClick={handleTelegramConnect}
            disabled={loading || !botToken.trim()}
            className="w-full mt-6"
          >
            {loading ? "..." : "connect"}
          </Button>
          <Button
            variant="aether-link"
            size="aether-link"
            onClick={resetForm}
            className="w-full text-center mt-6"
          >
            cancel
          </Button>
        </div>
      )}

      {/* ── Device list ── */}
      {mode === null && devices.length > 0 && (
        <div className="w-full max-sm">
          {devices.map((d, index) => (
            <div key={d.id}>
              <div className="flex items-center justify-between py-4">
                <div className="flex items-center gap-3">
                  <span className="text-sm text-secondary-foreground font-light">
                    {d.name}
                  </span>
                  <span className="text-[10px] text-muted-foreground tracking-wider">
                    {d.device_type}
                  </span>
                </div>
                <Button
                  variant="aether-link"
                  size="aether-link"
                  onClick={() => handleRemove(d.id)}
                  className="text-[10px] text-muted-foreground hover:text-destructive"
                >
                  remove
                </Button>
              </div>
              {index < devices.length - 1 && <Separator />}
            </div>
          ))}
          <Button
            variant="aether-link"
            size="aether-link"
            onClick={() => setMode("pick")}
            className="w-full text-center mt-8"
          >
            connect another device
          </Button>
        </div>
      )}
    </PageShell>
  );
}
