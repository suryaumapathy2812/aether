"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import PageShell from "@/components/PageShell";
import {
  Empty,
  EmptyDescription,
  EmptyHeader,
  EmptyTitle,
} from "@/components/ui/empty";
import { useSession } from "@/lib/auth-client";
import { useNotifications } from "@/components/NotificationProvider";

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

export default function NotificationsPage() {
  const router = useRouter();
  const { data: session, isPending } = useSession();
  const { notifications, unreadCount, markRead, markAllRead, clearAll } =
    useNotifications();

  useEffect(() => {
    if (!isPending && !session) {
      router.push("/");
    }
  }, [isPending, router, session]);

  if (isPending || !session) return null;

  return (
    <PageShell title="Notifications" back="/home">
      <div className="space-y-6 max-w-[980px] mx-auto">
        {notifications.length === 0 ? (
          <Empty className="border border-border/60 bg-white/[0.02]">
            <EmptyHeader>
              <EmptyTitle className="text-[16px] text-foreground font-medium">
                No notifications yet
              </EmptyTitle>
              <EmptyDescription className="text-[12px] max-w-[520px]">
                Updates from Aether will appear here when there is something for
                you to review.
              </EmptyDescription>
            </EmptyHeader>
          </Empty>
        ) : (
          <div className="rounded-2xl border border-border/70 bg-white/[0.03] overflow-hidden">
            <div className="px-4 py-3 border-b border-border/30 flex items-center justify-between">
              <h2 className="text-[14px] text-foreground">Recent updates</h2>
              <div className="flex items-center gap-3">
                {unreadCount > 0 && (
                  <button
                    onClick={markAllRead}
                    className="text-[10px] tracking-wider text-secondary-foreground hover:text-foreground transition-colors"
                  >
                    mark all read
                  </button>
                )}
                <button
                  onClick={clearAll}
                  className="text-[10px] tracking-wider text-muted-foreground hover:text-foreground transition-colors"
                >
                  clear
                </button>
              </div>
            </div>

            {notifications.map((n) => (
              <button
                key={n.id}
                onClick={() => markRead(n.id)}
                className={`w-full text-left px-4 py-3 border-b border-border/20 hover:bg-white/[0.03] transition-colors ${
                  n.read ? "" : "bg-white/[0.03]"
                }`}
              >
                <div className="flex items-start gap-2">
                  {!n.read && (
                    <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-blue-400 shrink-0" />
                  )}
                  <div className="min-w-0 flex-1">
                    <p className="text-[12px] text-foreground">{n.title}</p>
                    {n.body ? (
                      <p className="text-[11px] text-muted-foreground mt-0.5 line-clamp-2">
                        {n.body}
                      </p>
                    ) : null}
                    <p className="text-[10px] text-muted-foreground/70 mt-1">
                      {formatTimeAgo(n.timestamp)}
                    </p>
                  </div>
                </div>
              </button>
            ))}
          </div>
        )}
      </div>
    </PageShell>
  );
}
