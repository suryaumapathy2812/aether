/**
 * Custom service worker code injected by next-pwa.
 *
 * Handles Web Push notifications when the app tab is closed.
 * next-pwa automatically merges this with the generated Workbox SW.
 */

// Handle incoming push messages
self.addEventListener("push", (event) => {
  if (!event.data) return;

  let payload;
  try {
    payload = event.data.json();
  } catch {
    payload = { title: "Aether", body: event.data.text() };
  }

  const title = payload.title || "Aether";
  const options = {
    body: payload.body || "",
    icon: "/icons/icon-192.png",
    badge: "/icons/icon-192.png",
    tag: payload.tag || "aether-notification",
    data: {
      url: payload.url || "/home",
    },
    // Vibrate pattern for mobile
    vibrate: [100, 50, 100],
  };

  event.waitUntil(self.registration.showNotification(title, options));
});

// Handle notification click — focus or open the app
self.addEventListener("notificationclick", (event) => {
  event.notification.close();

  const url = event.notification.data?.url || "/home";

  event.waitUntil(
    self.clients
      .matchAll({ type: "window", includeUncontrolled: true })
      .then((clientList) => {
        // If a tab is already open, focus it
        for (const client of clientList) {
          if (client.url.includes(self.location.origin) && "focus" in client) {
            client.navigate(url);
            return client.focus();
          }
        }
        // Otherwise open a new tab
        return self.clients.openWindow(url);
      })
  );
});
