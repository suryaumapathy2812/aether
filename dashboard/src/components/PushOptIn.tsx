"use client";

import { useEffect, useState } from "react";
import { useSession } from "@/lib/auth-client";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  isPushSupported,
  getPermissionState,
  subscribeToPush,
  unsubscribeFromPush,
  isSubscribedToPush,
} from "@/lib/push";
import { toast } from "sonner";

/**
 * Push notification opt-in for the account/settings page.
 *
 * Shows the current push state and allows the user to enable/disable.
 * Hidden entirely if the browser doesn't support push.
 */
export default function PushOptIn() {
  const { data: session } = useSession();
  const userId = session?.user?.id || "";

  const [supported, setSupported] = useState(false);
  const [subscribed, setSubscribed] = useState(false);
  const [loading, setLoading] = useState(false);
  const [permissionState, setPermissionState] = useState<string>("default");

  useEffect(() => {
    const sup = isPushSupported();
    setSupported(sup);
    if (sup) {
      setPermissionState(getPermissionState());
      isSubscribedToPush().then(setSubscribed);
    }
  }, []);

  if (!supported) return null;

  async function handleToggle() {
    if (!userId) return;
    setLoading(true);
    try {
      if (subscribed) {
        await unsubscribeFromPush(userId);
        setSubscribed(false);
        toast.success("Notifications disabled");
      } else {
        const ok = await subscribeToPush(userId);
        setSubscribed(ok);
        setPermissionState(getPermissionState());
        if (ok) {
          toast.success("Notifications enabled");
        } else {
          toast.error("Could not enable notifications");
        }
      }
    } finally {
      setLoading(false);
    }
  }

  const denied = permissionState === "denied";

  return (
    <div className="flex items-center justify-between gap-4">
      <div className="flex items-center gap-3">
        <Badge variant={subscribed ? "default" : "secondary"}>
          {denied ? "Blocked" : subscribed ? "Enabled" : "Disabled"}
        </Badge>
        <p className="text-sm text-muted-foreground">
          {denied
            ? "Enable in browser site settings"
            : subscribed
            ? "You'll receive notifications when the app is closed"
            : "Push notifications are off"}
        </p>
      </div>
      <Button
        variant="outline"
        size="sm"
        onClick={handleToggle}
        disabled={loading || denied}
      >
        {loading ? "..." : subscribed ? "Disable" : "Enable"}
      </Button>
    </div>
  );
}
