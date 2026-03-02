"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { toast } from "sonner";
import PageShell from "@/components/PageShell";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  listDevices,
  registerTelegramDevice,
  deleteDevice,
  type Device,
} from "@/lib/api";
import { useSession } from "@/lib/auth-client";

export default function DevicesPage() {
  return <DevicesContent />;
}

function DevicesContent() {
  const router = useRouter();
  const { data: session, isPending } = useSession();
  const [devices, setDevices] = useState<Device[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [registering, setRegistering] = useState(false);

  // Form state
  const [botToken, setBotToken] = useState("");
  const [allowedChatIds, setAllowedChatIds] = useState("");
  const [deviceName, setDeviceName] = useState("");

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
      setError("");
      const data = await listDevices();
      setDevices(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to load devices");
    } finally {
      setLoading(false);
    }
  }

  async function handleRegisterTelegram(e: React.FormEvent) {
    e.preventDefault();
    if (!botToken.trim()) {
      toast.error("Bot token is required");
      return;
    }

    try {
      setRegistering(true);
      await registerTelegramDevice(
        botToken.trim(),
        allowedChatIds.trim() || undefined,
        deviceName.trim() || undefined
      );
      toast.success("Telegram device registered successfully");
      setBotToken("");
      setAllowedChatIds("");
      setDeviceName("");
      await loadDevices();
    } catch (e: unknown) {
      toast.error(e instanceof Error ? e.message : "Failed to register device");
    } finally {
      setRegistering(false);
    }
  }

  async function handleDeleteDevice(deviceId: string) {
    try {
      await deleteDevice(deviceId);
      toast.success("Device removed");
      await loadDevices();
    } catch (e: unknown) {
      toast.error(e instanceof Error ? e.message : "Failed to delete device");
    }
  }

  if (isPending || !session) return null;

  return (
    <PageShell title="Devices" back="/home" centered={loading}>
      {loading ? (
        <p className="text-muted-foreground text-xs tracking-wider">
          loading...
        </p>
      ) : error ? (
        <div>
          <p className="text-muted-foreground text-xs mb-4">{error}</p>
          <Button variant="aether" size="aether" onClick={loadDevices}>
            try again
          </Button>
        </div>
      ) : (
        <div className="space-y-6 max-w-md">
          {/* Telegram Registration */}
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Telegram</CardTitle>
              <CardDescription className="text-xs">
                Connect your Telegram bot to receive messages and notifications.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleRegisterTelegram} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="botToken" className="text-xs">
                    Bot Token <span className="text-destructive">*</span>
                  </Label>
                  <Input
                    id="botToken"
                    type="password"
                    placeholder="123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"
                    value={botToken}
                    onChange={(e) => setBotToken(e.target.value)}
                    disabled={registering}
                  />
                  <p className="text-[10px] text-muted-foreground">
                    Create a bot with @BotFather on Telegram
                  </p>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="allowedChatIds" className="text-xs">
                    Allowed Chat IDs
                  </Label>
                  <Input
                    id="allowedChatIds"
                    placeholder="123456789, 987654321"
                    value={allowedChatIds}
                    onChange={(e) => setAllowedChatIds(e.target.value)}
                    disabled={registering}
                  />
                  <p className="text-[10px] text-muted-foreground">
                    Comma-separated. Leave empty to allow any chat.
                  </p>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="deviceName" className="text-xs">
                    Device Name
                  </Label>
                  <Input
                    id="deviceName"
                    placeholder="My Telegram Bot"
                    value={deviceName}
                    onChange={(e) => setDeviceName(e.target.value)}
                    disabled={registering}
                  />
                </div>
                <Button
                  type="submit"
                  variant="aether"
                  size="aether"
                  disabled={registering}
                  className="w-full"
                >
                  {registering ? "Connecting..." : "Connect Telegram"}
                </Button>
              </form>
            </CardContent>
          </Card>

          {/* Connected Devices */}
          <div>
            <h3 className="text-sm font-medium mb-3">Connected Devices</h3>
            {devices.length === 0 ? (
              <p className="text-muted-foreground text-xs">
                No devices connected yet.
              </p>
            ) : (
              <div className="space-y-2">
                {devices.map((device) => (
                  <div
                    key={device.id}
                    className="flex items-center justify-between p-3 border rounded-lg"
                  >
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium truncate">
                        {device.name}
                      </p>
                      <p className="text-xs text-muted-foreground">
                        {device.device_type}
                        {device.paired_at && (
                          <> • {new Date(device.paired_at).toLocaleDateString()}</>
                        )}
                      </p>
                    </div>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => handleDeleteDevice(device.id)}
                      className="text-destructive hover:text-destructive ml-2"
                    >
                      Remove
                    </Button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </PageShell>
  );
}
