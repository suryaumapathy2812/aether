import { useState, useEffect } from "react";
import { IconBell } from "@tabler/icons-react";
import { Link } from "@tanstack/react-router";
import { useNotifications } from "#/components/NotificationProvider";

export default function NotificationBell() {
  const {
    notifications,
    unreadCount,
    markRead,
    markAllRead,
    clearAll,
    connected,
    pendingApprovals,
    loadingApprovals,
  } = useNotifications();

  const [open, setOpen] = useState(false);
  const waitingCount = pendingApprovals.length;
  const badgeCount = unreadCount + waitingCount;

  useEffect(() => {
    if (!open) return;
    function handleKey(e: KeyboardEvent) {
      if (e.key === "Escape") setOpen(false);
    }
    document.addEventListener("keydown", handleKey);
    return () => document.removeEventListener("keydown", handleKey);
  }, [open]);

  return (
    <>
      <button
        onClick={() => setOpen((prev) => !prev)}
        className="relative w-8 h-8 flex items-center justify-center text-muted-foreground hover:text-foreground transition-colors"
        aria-label="Notifications"
      >
        <IconBell className="size-[18px]" strokeWidth={1.5} />
        {badgeCount > 0 && (
          <span className="absolute -top-0.5 -right-0.5 min-w-[16px] h-[16px] rounded-full bg-red-500 text-white text-xs font-medium flex items-center justify-center px-1">
            {badgeCount > 99 ? "99+" : badgeCount}
          </span>
        )}
        {!connected && (
          <span className="absolute bottom-0 right-0 w-1.5 h-1.5 rounded-full bg-red-400/80" />
        )}
      </button>

      {open && (
        <div className="fixed inset-0 z-40" onClick={() => setOpen(false)} />
      )}

      <div
        className={`fixed top-0 right-0 z-50 h-full w-[360px] max-w-[90vw] transition-transform duration-300 ease-out ${
          open ? "translate-x-0" : "translate-x-full"
        }`}
      >
        <div className="h-[calc(100%-24px)] m-3 ml-0 flex flex-col rounded-2xl border border-border/70 bg-background/95 backdrop-blur-xl shadow-2xl overflow-hidden">
          <div className="px-4 py-3 border-b border-border/40">
            <div className="flex items-center justify-between">
              <p className="text-sm tracking-[0.14em] uppercase text-muted-foreground">
                Notifications
              </p>
              <div className="flex items-center gap-3">
                {unreadCount > 0 && (
                  <button
                    onClick={markAllRead}
                    className="text-xs tracking-wider text-secondary-foreground hover:text-foreground transition-colors"
                  >
                    mark all read
                  </button>
                )}
                <button
                  onClick={clearAll}
                  className="text-xs tracking-wider text-muted-foreground hover:text-foreground transition-colors"
                >
                  clear
                </button>
              </div>
            </div>
          </div>

          {(waitingCount > 0 || loadingApprovals) && (
            <div className="px-4 py-3 border-b border-amber-500/30 bg-amber-500/10">
              <p className="text-sm text-amber-100">
                {loadingApprovals
                  ? "Checking if any agents need your input..."
                  : `${waitingCount} agent${waitingCount === 1 ? "" : "s"} waiting for your input`}
              </p>
              <Link
                to="/chat"
                onClick={() => setOpen(false)}
                className="inline-block mt-2 text-xs tracking-[0.1em] uppercase text-amber-100 hover:text-white"
              >
                review agents
              </Link>
            </div>
          )}

          <div className="flex-1 overflow-y-auto">
            {notifications.length === 0 ? (
              <div className="flex items-center justify-center h-full min-h-[200px] px-6 text-center">
                <p className="text-muted-foreground text-xs">
                  no updates yet
                </p>
              </div>
            ) : (
              notifications.map((notif) => (
                <button
                  key={notif.id}
                  onClick={() => markRead(notif.id)}
                  className={`w-full text-left px-4 py-3 border-b border-border/20 hover:bg-accent/30 transition-colors ${
                    !notif.read ? "bg-accent/20" : ""
                  }`}
                >
                  <div className="flex items-start gap-2">
                    {!notif.read && (
                      <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-blue-400 shrink-0" />
                    )}
                    <div className="min-w-0 flex-1">
                      <p className="text-sm text-foreground truncate">
                        {notif.title}
                      </p>
                      {notif.body && (
                        <p className="text-sm text-muted-foreground mt-0.5 line-clamp-2">
                          {notif.body}
                        </p>
                      )}
                      <p className="text-xs text-muted-foreground/70 mt-1">
                        {formatTimeAgo(notif.timestamp)}
                      </p>
                    </div>
                  </div>
                </button>
              ))
            )}
          </div>

          <div className="px-4 py-3 border-t border-border/30">
            <Link
              to="/notifications"
              onClick={() => setOpen(false)}
              className="text-xs tracking-[0.1em] uppercase text-secondary-foreground hover:text-foreground transition-colors"
            >
              open notifications page
            </Link>
          </div>
        </div>
      </div>
    </>
  );
}

function formatTimeAgo(ts: number): string {
  const diff = Date.now() - ts;
  const seconds = Math.floor(diff / 1000);
  if (seconds < 60) return "just now";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}
