"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { toast } from "sonner";
import ContentShell from "@/components/ContentShell";
import MinimalInput from "@/components/MinimalInput";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { useSession } from "@/lib/auth-client";
import { IconBrandTelegram } from "@tabler/icons-react";
import {
  listDevices,
  registerTelegramDevice,
  deleteDevice,
  type Device,
} from "@/lib/api";

export default function TelegramDevicePage() {
  const router = useRouter();
  const { data: session, isPending } = useSession();
  const [devices, setDevices] = useState<Device[]>([]);
  const [loading, setLoading] = useState(true);
  const [registering, setRegistering] = useState(false);

  // Form
  const [botToken, setBotToken] = useState("");
  const [chatIds, setChatIds] = useState("");
  const [name, setName] = useState("");

  useEffect(() => {
    if (isPending) return;
    if (!session) {
      router.push("/");
      return;
    }
    loadDevices();
  }, [session, isPending, router]);

  async function loadDevices() {
    try {
      setLoading(true);
      const all = await listDevices();
      setDevices(all.filter((d) => d.device_type === "telegram"));
    } catch {
      // Fail silently
    } finally {
      setLoading(false);
    }
  }

  async function handleRegister(e: React.FormEvent) {
    e.preventDefault();
    if (!botToken.trim()) {
      toast.error("Bot token is required");
      return;
    }

    try {
      setRegistering(true);
      await registerTelegramDevice(
        botToken.trim(),
        chatIds.trim() || undefined,
        name.trim() || undefined
      );
      toast.success("Telegram bot connected");
      setBotToken("");
      setChatIds("");
      setName("");
      await loadDevices();
    } catch (e: unknown) {
      toast.error(
        e instanceof Error ? e.message : "Failed to connect"
      );
    } finally {
      setRegistering(false);
    }
  }

  async function handleDelete(device: Device) {
    try {
      await deleteDevice(device.id);
      toast.success("Device removed");
      await loadDevices();
    } catch (e: unknown) {
      toast.error(e instanceof Error ? e.message : "Failed to remove");
    }
  }

  if (isPending || !session) return null;

  return (
    <ContentShell title="Telegram Bot" back="/devices">
      <div className="space-y-6">
        <div className="flex items-start gap-3">
          <IconBrandTelegram
            className="size-5 shrink-0 text-muted-foreground mt-0.5"
            strokeWidth={1.5}
          />
          <p className="text-sm text-muted-foreground leading-relaxed max-w-[72ch]">
            Connect a Telegram bot to chat with Aether from Telegram. Create one
            with{" "}
            <a
              href="https://t.me/BotFather"
              target="_blank"
              rel="noopener noreferrer"
              className="underline underline-offset-2"
            >
              @BotFather
            </a>{" "}
            first.
          </p>
        </div>

        <form onSubmit={handleRegister} className="space-y-1">
          <MinimalInput
            label="Bot Token"
            type="password"
            value={botToken}
            onChange={setBotToken}
            placeholder="123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"
          />
          <MinimalInput
            label="Allowed Chat IDs"
            value={chatIds}
            onChange={setChatIds}
            placeholder="123456789, 987654321"
          />
          <p className="text-xs text-muted-foreground -mt-3 mb-2">
            Comma-separated. Leave empty to allow any chat.
          </p>
          <MinimalInput
            label="Device Name"
            value={name}
            onChange={setName}
            placeholder="My Telegram Bot"
          />
          <Button
            type="submit"
            variant="aether"
            size="aether"
            disabled={registering || !botToken.trim()}
            className="w-full"
          >
            {registering ? "connecting..." : "connect bot"}
          </Button>
        </form>

        {!loading && devices.length > 0 && (
          <>
            <Separator />
            <div className="space-y-2">
              <h2 className="text-xs tracking-[0.15em] text-muted-foreground uppercase font-normal">
                Connected bots
              </h2>
              {devices.map((device) => (
                <TelegramRow
                  key={device.id}
                  device={device}
                  onDelete={() => handleDelete(device)}
                />
              ))}
            </div>
          </>
        )}
      </div>
    </ContentShell>
  );
}

function TelegramRow({
  device,
  onDelete,
}: {
  device: Device;
  onDelete: () => void;
}) {
  const [busy, setBusy] = useState(false);

  async function handleDelete() {
    setBusy(true);
    await onDelete();
    setBusy(false);
  }

  return (
    <div className="flex items-center justify-between gap-3 rounded-2xl bg-accent/30 border border-border px-4 py-3">
      <div className="flex items-center gap-3 min-w-0">
        <IconBrandTelegram
          className="size-4 shrink-0 text-muted-foreground"
          strokeWidth={1.5}
        />
        <div className="min-w-0">
          <p className="text-sm text-foreground font-medium truncate">
            {device.name}
          </p>
          <p className="text-sm text-muted-foreground/60">
            {device.paired_at
              ? `Added ${new Date(device.paired_at).toLocaleDateString()}`
              : "Telegram bot"}
          </p>
        </div>
      </div>

      <button
        onClick={handleDelete}
        disabled={busy}
        className="text-sm tracking-wider px-2.5 py-1 rounded-full bg-red-500/10 text-red-400 hover:bg-red-500/20 transition-colors shrink-0"
      >
        {busy ? "..." : "remove"}
      </button>
    </div>
  );
}
