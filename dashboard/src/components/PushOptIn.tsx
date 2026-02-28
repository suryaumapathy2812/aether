"use client";

import { useEffect, useState } from "react";
import { useSession } from "@/lib/auth-client";
import { Switch } from "@/components/ui/switch";
import {
  isPushSupported,
  getPermissionState,
  subscribeToPush,
  unsubscribeFromPush,
  isSubscribedToPush,
} from "@/lib/push";

/**
 * Push notification opt-in toggle for the account/settings page.
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

  async function handleToggle(checked: boolean) {
    if (!userId) return;
    setLoading(true);
    try {
      if (checked) {
        const ok = await subscribeToPush(userId);
        setSubscribed(ok);
        setPermissionState(getPermissionState());
      } else {
        await unsubscribeFromPush(userId);
        setSubscribed(false);
      }
    } finally {
      setLoading(false);
    }
  }

  const denied = permissionState === "denied";

  return (
    <div className="flex items-center justify-between gap-4 py-3">
      <div>
        <p className="text-[13px] text-foreground">Push notifications</p>
        <p className="text-[11px] text-muted-foreground mt-0.5">
          {denied
            ? "Blocked by browser \u2014 enable in site settings"
            : subscribed
            ? "You\u2019ll receive notifications even when the app is closed"
            : "Get notified about task updates and messages"}
        </p>
      </div>
      <Switch
        checked={subscribed}
        onCheckedChange={handleToggle}
        disabled={loading || denied}
        size="sm"
      />
    </div>
  );
}
