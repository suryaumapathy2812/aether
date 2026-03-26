import { getStatusBadge } from "./tool";
import type { ToolRendererProps } from "./tool-renderer";

// ── Helpers ──────────────────────────────────────────────────────────────

function tryParseJSON(value: unknown): Record<string, unknown> | null {
  if (typeof value !== "string") return null;
  try {
    const parsed = JSON.parse(value);
    return typeof parsed === "object" && parsed !== null ? parsed : null;
  } catch {
    return null;
  }
}

function tryParseArray(value: unknown): unknown[] | null {
  if (typeof value !== "string") return null;
  try {
    const parsed = JSON.parse(value);
    return Array.isArray(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

function getToolName(type: string): string {
  return type.replace("tool-", "");
}

function extractFirstString(arr: unknown): string | null {
  if (!Array.isArray(arr) || arr.length === 0) return null;
  const first = arr[0] as Record<string, unknown>;
  return typeof first.value === "string" ? first.value : typeof first.displayName === "string" ? first.displayName : null;
}

function extractAll(arr: unknown, key: string): string[] {
  if (!Array.isArray(arr)) return [];
  return arr
    .map((item) => {
      if (typeof item === "object" && item !== null) {
        const val = (item as Record<string, unknown>)[key];
        return typeof val === "string" ? val : "";
      }
      return "";
    })
    .filter(Boolean);
}

// ── Renderers ────────────────────────────────────────────────────────────

function ContactCard({ person }: { person: Record<string, unknown> }) {
  const names = Array.isArray(person.names) ? (person.names as Record<string, unknown>[]) : [];
  const emails = Array.isArray(person.emailAddresses) ? (person.emailAddresses as Record<string, unknown>[]) : [];
  const phones = Array.isArray(person.phoneNumbers) ? (person.phoneNumbers as Record<string, unknown>[]) : [];
  const orgs = Array.isArray(person.organizations) ? (person.organizations as Record<string, unknown>[]) : [];
  const addresses = Array.isArray(person.addresses) ? (person.addresses as Record<string, unknown>[]) : [];
  const birthdays = Array.isArray(person.birthdays) ? (person.birthdays as Record<string, unknown>[]) : [];
  const bio = typeof person.biography === "string" ? person.biography : "";

  const displayName = names.length > 0 ? (typeof names[0].displayName === "string" ? names[0].displayName : "Unknown") : "Unknown";
  const resourceName = typeof person.resourceName === "string" ? person.resourceName : "";

  return (
    <div className="rounded-lg border border-border bg-accent/20 p-3 space-y-2">
      <div className="flex items-center gap-3">
        <div className="size-9 rounded-full bg-gradient-to-br from-foreground/10 to-foreground/5 flex items-center justify-center text-sm font-medium text-foreground/60 shrink-0">
          {displayName.charAt(0).toUpperCase()}
        </div>
        <div className="min-w-0">
          <div className="text-sm font-semibold text-foreground truncate">{displayName}</div>
          {orgs.length > 0 && (
            <div className="text-xs text-muted-foreground truncate">
              {typeof orgs[0].title === "string" ? orgs[0].title + " · " : ""}
              {typeof orgs[0].name === "string" ? orgs[0].name : ""}
            </div>
          )}
        </div>
      </div>

      {emails.length > 0 && (
        <div className="space-y-0.5">
          {emails.map((email, i) => (
            <div key={i} className="text-xs text-muted-foreground">
              <span className="text-muted-foreground/60">{typeof email.type === "string" ? email.type : "Email"}:</span>{" "}
              <span className="text-foreground/80">{typeof email.value === "string" ? email.value : ""}</span>
            </div>
          ))}
        </div>
      )}

      {phones.length > 0 && (
        <div className="space-y-0.5">
          {phones.map((phone, i) => (
            <div key={i} className="text-xs text-muted-foreground">
              <span className="text-muted-foreground/60">{typeof phone.type === "string" ? phone.type : "Phone"}:</span>{" "}
              <span className="text-foreground/80">{typeof phone.value === "string" ? phone.value : ""}</span>
            </div>
          ))}
        </div>
      )}

      {addresses.length > 0 && (
        <div className="space-y-0.5">
          {addresses.map((addr, i) => (
            <div key={i} className="text-xs text-muted-foreground">
              <span className="text-muted-foreground/60">{typeof addr.type === "string" ? addr.type : "Address"}:</span>{" "}
              <span className="text-foreground/80">{typeof addr.formattedValue === "string" ? addr.formattedValue : ""}</span>
            </div>
          ))}
        </div>
      )}

      {birthdays.length > 0 && (
        <div className="text-xs text-muted-foreground">
          {birthdays.map((b, i) => {
            const date = typeof b.date === "object" && b.date !== null ? b.date as Record<string, unknown> : null;
            if (!date) return null;
            const month = typeof date.month === "number" ? date.month : 0;
            const day = typeof date.day === "number" ? date.day : 0;
            if (!month || !day) return null;
            return <span key={i}>Birthday: {month}/{day}</span>;
          })}
        </div>
      )}

      {bio && <div className="text-xs text-muted-foreground/80">{bio}</div>}
      {resourceName && (
        <div className="text-[10px] text-muted-foreground/40 font-mono">{resourceName}</div>
      )}
    </div>
  );
}

function SearchResultsCard({ results, title }: { results: unknown[]; title: string }) {
  const people = results.filter((r): r is Record<string, unknown> => typeof r === "object" && r !== null);

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-xs text-muted-foreground">
        <span>{title}</span>
        <span>{people.length} contact{people.length !== 1 ? "s" : ""}</span>
      </div>
      {people.length === 0 ? (
        <div className="text-sm text-muted-foreground/60 py-2">No contacts found.</div>
      ) : (
        <div className="space-y-2 max-h-96 overflow-y-auto">
          {people.map((person, i) => {
            const personObj = typeof person.person === "object" && person.person !== null
              ? (person.person as Record<string, unknown>)
              : person;
            return <ContactCard key={i} person={personObj} />;
          })}
        </div>
      )}
    </div>
  );
}

// ── Main Component ───────────────────────────────────────────────────────

export function ContactsTool({ part }: ToolRendererProps) {
  const toolName = getToolName(part.type);
  const isRunning =
    part.state === "input-streaming" || part.state === "input-available";

  if (isRunning) {
    return (
      <div className="w-full rounded-lg border border-border bg-accent/10 px-3 py-2 flex items-center gap-2">
        {getStatusBadge(part.state as "input-available")}
        <span className="text-xs text-muted-foreground">
          {toolName.replace(/_/g, " ")}
        </span>
      </div>
    );
  }

  if (part.state === "output-error") {
    return (
      <div className="w-full rounded-lg border border-red-500/20 bg-red-500/[0.05] px-3 py-2 text-xs text-red-300/80">
        {part.errorText || "Contacts operation failed"}
      </div>
    );
  }

  if (part.state !== "output-available") return null;

  switch (toolName) {
    case "search_contacts": {
      const results = tryParseArray(part.output) ?? [];
      return (
        <div className="w-full">
          <SearchResultsCard results={results} title="Search results" />
        </div>
      );
    }

    case "get_contact": {
      const data = tryParseJSON(part.output);
      if (!data) break;
      return (
        <div className="w-full">
          <ContactCard person={data} />
        </div>
      );
    }
  }

  return (
    <div className="w-full rounded-lg border border-border bg-accent/20 px-3 py-2 text-sm text-foreground">
      {typeof part.output === "string" ? part.output : "Done"}
    </div>
  );
}
