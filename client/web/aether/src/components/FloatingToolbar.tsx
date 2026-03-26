import { Suspense } from "react";
import { useNavigate, useLocation } from "@tanstack/react-router";
import { useTheme } from "#/components/ThemeProvider";
import { cn } from "#/lib/utils";
import { useSession } from "#/lib/auth-client";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "#/components/ui/tooltip";
import { Avatar, AvatarFallback, AvatarImage } from "#/components/ui/avatar";
import {
  IconMessage,
  IconMessage2Filled,
  IconBrain,
  IconDeviceMobile,
  IconDeviceMobileFilled,
  IconPlugConnected,
  IconSparkles,
  IconSun,
  IconMoon,
} from "@tabler/icons-react";

const NAV_ITEMS = [
  { href: "/devices", label: "Devices" },
  { href: "/memory", label: "Memory" },
  { href: "/integrations", label: "Integrations" },
  { href: "/skills", label: "Skills" },
] as const;

const HIDE_TOOLBAR = ["/"];

function NavIcon({ href, active }: { href: string; active: boolean }) {
  switch (href) {
    case "/devices":
      return active ? <IconDeviceMobileFilled className="size-5" /> : <IconDeviceMobile className="size-5" strokeWidth={1.5} />;
    case "/memory":
      return <IconBrain className="size-5" strokeWidth={active ? 2 : 1.5} />;
    case "/integrations":
      return <IconPlugConnected className="size-5" strokeWidth={active ? 2 : 1.5} />;
    case "/skills":
      return <IconSparkles className="size-5" strokeWidth={active ? 2 : 1.5} />;
    default:
      return null;
  }
}

export default function FloatingToolbar() {
  return (
    <Suspense>
      <FloatingToolbarInner />
    </Suspense>
  );
}

function FloatingToolbarInner() {
  const navigate = useNavigate();
  const { pathname } = useLocation();
  const { data: session } = useSession();
  const { theme, setTheme } = useTheme();

  if (HIDE_TOOLBAR.includes(pathname)) return null;

  const toggleTheme = () => setTheme(theme === "dark" ? "light" : "dark");

  return (
    <>
      {/* Gradient fade from background to transparent */}
      <div className="fixed top-0 left-0 right-0 z-30 h-20 bg-gradient-to-b from-background to-transparent pointer-events-none" />

      <div className="fixed top-0 left-0 right-0 z-40 flex items-center justify-between px-4 py-3 pointer-events-none">
      {/* Left — logo (new chat) */}
      <div className="flex items-center pointer-events-auto">
        <button
          onClick={() => navigate({ to: "/chat" })}
          className="flex items-center justify-center rounded-lg transition-colors cursor-pointer"
          aria-label="Home"
        >
          <img
            src={theme === "dark" ? "/icon-animated-white.svg" : "/icon-animated-black.svg"}
            alt="Aether"
            className="w-8 h-8"
          />
        </button>
      </div>

      {/* Right — nav icons + account */}
      <div className="flex items-center gap-1 pointer-events-auto">
        <Tooltip>
          <TooltipTrigger asChild>
            <button
              onClick={() => navigate({ to: "/sessions" })}
              className={cn(
                "w-9 h-9 flex items-center justify-center rounded-lg text-foreground transition-colors",
                pathname.startsWith("/sessions")
                  ? "bg-foreground text-background"
                  : "hover:bg-accent/60"
              )}
              aria-label="Sessions"
            >
              {pathname.startsWith("/sessions") ? (
                <IconMessage2Filled className="size-5" />
              ) : (
                <IconMessage className="size-5" strokeWidth={1.5} />
              )}
            </button>
          </TooltipTrigger>
          <TooltipContent side="bottom">Sessions</TooltipContent>
        </Tooltip>

        {NAV_ITEMS.map((item) => {
          const isActive = pathname.startsWith(item.href);
          return (
            <Tooltip key={item.href}>
              <TooltipTrigger asChild>
                <button
                  onClick={() => navigate({ to: item.href })}
                  className={cn(
                    "w-9 h-9 flex items-center justify-center rounded-lg text-foreground transition-colors",
                    isActive
                      ? "bg-foreground text-background"
                      : "hover:bg-accent/60"
                  )}
                  aria-label={item.label}
                >
                  <NavIcon href={item.href} active={isActive} />
                </button>
              </TooltipTrigger>
              <TooltipContent side="bottom">{item.label}</TooltipContent>
            </Tooltip>
          );
        })}

        {/* Theme toggle */}
        <Tooltip>
          <TooltipTrigger asChild>
            <button
              onClick={toggleTheme}
              className="w-9 h-9 flex items-center justify-center rounded-lg text-foreground transition-colors hover:bg-accent/60"
              aria-label="Toggle theme"
            >
              {theme === "dark" ? (
                <IconSun className="size-5" strokeWidth={1.5} />
              ) : (
                <IconMoon className="size-5" strokeWidth={1.5} />
              )}
            </button>
          </TooltipTrigger>
          <TooltipContent side="bottom">Toggle theme</TooltipContent>
        </Tooltip>

        <Tooltip>
          <TooltipTrigger asChild>
            <button
              onClick={() => navigate({ to: "/account" })}
              className={cn(
                "w-9 h-9 flex items-center justify-center rounded-lg transition-colors",
                pathname.startsWith("/account")
                  ? "bg-foreground text-background"
                  : "text-foreground hover:bg-accent/60"
              )}
              aria-label="Account"
            >
              <Avatar size="sm">
                {session?.user?.image && (
                  <AvatarImage
                    src={session.user.image}
                    alt={session?.user?.name || ""}
                  />
                )}
                <AvatarFallback className="text-sm">
                  {(session?.user?.name || session?.user?.email || "U")
                    .charAt(0)
                    .toUpperCase()}
                </AvatarFallback>
              </Avatar>
            </button>
          </TooltipTrigger>
          <TooltipContent side="bottom">Account</TooltipContent>
        </Tooltip>
      </div>
    </div>
    </>
  );
}
