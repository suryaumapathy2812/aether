import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { useNavigate } from "@tanstack/react-router";
import ContentShell from "#/components/ContentShell";
import { Button } from "#/components/ui/button";
import { Input } from "#/components/ui/input";
import { Avatar, AvatarFallback } from "#/components/ui/avatar";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "#/components/ui/card";
import PushOptIn from "#/components/PushOptIn";
import ModelPreference from "#/components/ModelPreference";
import { useSession, signOut } from "#/lib/auth-client";
import { IconLogout } from "@tabler/icons-react";

export const Route = createFileRoute("/account")({
  component: AccountPage,
});

function AccountPage() {
  const navigate = useNavigate();
  const { data: session, isPending } = useSession();
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [editingName, setEditingName] = useState(false);

  useEffect(() => {
    if (!isPending && !session) {
      navigate({ to: "/" });
      return;
    }
    if (session) {
      setName(session.user.name || "");
      setEmail(session.user.email);
    }
  }, [session, isPending, navigate]);

  async function handleLogout() {
    await signOut();
    navigate({ to: "/" });
  }

  if (isPending || !session) return null;

  return (
    <ContentShell title="Settings">
      <div className="flex flex-col gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Profile</CardTitle>
            <CardDescription>Manage your account details</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-4">
              <Avatar className="size-11">
                <AvatarFallback className="text-base font-medium">
                  {(name || email).charAt(0).toUpperCase()}
                </AvatarFallback>
              </Avatar>
              <div className="min-w-0 flex-1">
                {editingName ? (
                  <Input
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    onBlur={() => setEditingName(false)}
                    onKeyDown={(e) => e.key === "Enter" && setEditingName(false)}
                    autoFocus
                    className="h-9"
                  />
                ) : (
                  <button
                    className="text-base font-medium text-foreground hover:text-foreground/70 transition-colors text-left"
                    onClick={() => setEditingName(true)}
                  >
                    {name || "Add name"}
                  </button>
                )}
                <p className="text-sm text-muted-foreground mt-0.5">{email}</p>
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={handleLogout}
                className="text-destructive hover:text-destructive hover:bg-destructive/10 shrink-0"
              >
                <IconLogout data-icon="inline-start" />
                Log out
              </Button>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Notifications</CardTitle>
            <CardDescription>Push notifications for updates</CardDescription>
          </CardHeader>
          <CardContent>
            <PushOptIn />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Model</CardTitle>
            <CardDescription>Override the default model for AI tasks</CardDescription>
          </CardHeader>
          <CardContent>
            <ModelPreference />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Voice &amp; Media Model</CardTitle>
            <CardDescription>Model for voice and media turns (must support audio input)</CardDescription>
          </CardHeader>
          <CardContent>
            <ModelPreference prefKey="voice_model" placeholder="google/gemini-2.5-flash" />
          </CardContent>
        </Card>
      </div>
    </ContentShell>
  );
}
