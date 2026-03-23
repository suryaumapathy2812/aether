/**
 * Web Push subscription utilities.
 *
 * Handles the browser-side flow:
 *   1. Request notification permission
 *   2. Get VAPID public key from agent
 *   3. Subscribe via PushManager
 *   4. Send subscription to agent for server-side push delivery
 */

import { directAgentFetch } from "@/lib/api";

/**
 * Check if push notifications are supported in the current browser.
 */
export function isPushSupported(): boolean {
  return (
    typeof window !== "undefined" &&
    "serviceWorker" in navigator &&
    "PushManager" in window &&
    "Notification" in window
  );
}

/**
 * Get the current notification permission state.
 */
export function getPermissionState(): NotificationPermission | "unsupported" {
  if (!isPushSupported()) return "unsupported";
  return Notification.permission;
}

/**
 * Request notification permission from the user.
 */
export async function requestPermission(): Promise<NotificationPermission> {
  if (!isPushSupported()) return "denied";
  return Notification.requestPermission();
}

/**
 * Get the VAPID public key from the agent.
 */
async function getVapidKey(): Promise<string> {
  const res = await directAgentFetch("/api/push/vapid-key");
  if (!res.ok) throw new Error("Failed to get VAPID key");
  const data = await res.json();
  return data.public_key || "";
}

/**
 * Convert a base64 URL-safe string to a Uint8Array (for applicationServerKey).
 */
function urlBase64ToUint8Array(base64String: string): Uint8Array {
  const padding = "=".repeat((4 - (base64String.length % 4)) % 4);
  const base64 = (base64String + padding).replace(/-/g, "+").replace(/_/g, "/");
  const raw = window.atob(base64);
  const arr = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) {
    arr[i] = raw.charCodeAt(i);
  }
  return arr;
}

/**
 * Subscribe to push notifications.
 * Returns true if successful, false otherwise.
 */
export async function subscribeToPush(userId: string): Promise<boolean> {
  if (!isPushSupported()) return false;

  const permission = await requestPermission();
  if (permission !== "granted") return false;

  try {
    const vapidKey = await getVapidKey();
    if (!vapidKey) {
      console.warn("push: VAPID key not configured on server");
      return false;
    }

    const registration = await navigator.serviceWorker.ready;
    const keyBytes = urlBase64ToUint8Array(vapidKey);
    const subscription = await registration.pushManager.subscribe({
      userVisibleOnly: true,
      applicationServerKey: keyBytes.buffer as ArrayBuffer,
    });

    const subJSON = subscription.toJSON();

    const res = await directAgentFetch("/api/push/subscribe", {
      method: "POST",
      body: JSON.stringify({
        user_id: userId,
        subscription: {
          endpoint: subJSON.endpoint,
          keys: {
            p256dh: subJSON.keys?.p256dh || "",
            auth: subJSON.keys?.auth || "",
          },
        },
      }),
    });

    return res.ok;
  } catch (err) {
    console.error("push: subscribe failed", err);
    return false;
  }
}

/**
 * Unsubscribe from push notifications.
 */
export async function unsubscribeFromPush(userId: string): Promise<boolean> {
  if (!isPushSupported()) return false;

  try {
    const registration = await navigator.serviceWorker.ready;
    const subscription = await registration.pushManager.getSubscription();
    if (subscription) {
      // Tell the agent to remove this subscription
      await directAgentFetch("/api/push/subscribe", {
        method: "DELETE",
        body: JSON.stringify({
          user_id: userId,
          endpoint: subscription.endpoint,
        }),
      });
      await subscription.unsubscribe();
    }
    return true;
  } catch (err) {
    console.error("push: unsubscribe failed", err);
    return false;
  }
}

/**
 * Check if the user is currently subscribed to push.
 */
export async function isSubscribedToPush(): Promise<boolean> {
  if (!isPushSupported()) return false;
  try {
    const registration = await navigator.serviceWorker.ready;
    const subscription = await registration.pushManager.getSubscription();
    return subscription !== null;
  } catch {
    return false;
  }
}
