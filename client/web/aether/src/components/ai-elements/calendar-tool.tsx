import { CodeBlock } from "./code-block";
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

function extractSuccessMessage(output: unknown): string {
  if (typeof output !== "string") return "";
  const jsonStart = output.indexOf("{");
  const arrStart = output.indexOf("[");
  const cut = jsonStart === -1 ? arrStart : arrStart === -1 ? jsonStart : Math.min(jsonStart, arrStart);
  if (cut <= 0) return "";
  return output.slice(0, cut).trim();
}

function parseDateTime(dt: Record<string, unknown>): { date: string; time: string; isAllDay: boolean } {
  if (typeof dt.dateTime === "string") {
    const d = new Date(dt.dateTime);
    return {
      date: d.toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric" }),
      time: d.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" }),
      isAllDay: false,
    };
  }
  if (typeof dt.date === "string") {
    const d = new Date(dt.date + "T00:00:00");
    return {
      date: d.toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric" }),
      time: "All day",
      isAllDay: true,
    };
  }
  return { date: "Unknown", time: "", isAllDay: false };
}

function getToolName(type: string): string {
  return type.replace("tool-", "");
}

// ── Renderers ────────────────────────────────────────────────────────────

function EventCard({ event }: { event: Record<string, unknown> }) {
  const summary = typeof event.summary === "string" ? event.summary : "(No title)";
  const start = typeof event.start === "object" && event.start !== null
    ? parseDateTime(event.start as Record<string, unknown>)
    : { date: "Unknown", time: "", isAllDay: false };
  const end = typeof event.end === "object" && event.end !== null
    ? parseDateTime(event.end as Record<string, unknown>)
    : null;
  const location = typeof event.location === "string" ? event.location : "";
  const description = typeof event.description === "string" ? event.description : "";
  const status = typeof event.status === "string" ? event.status : "";
  const organizer = typeof event.organizer === "object" && event.organizer !== null
    ? (event.organizer as Record<string, unknown>)
    : null;
  const attendees = Array.isArray(event.attendees)
    ? (event.attendees as Record<string, unknown>[])
    : [];
  const htmlLink = typeof event.htmlLink === "string" ? event.htmlLink : "";

  const organizerName = organizer
    ? (typeof organizer.displayName === "string" ? organizer.displayName : typeof organizer.email === "string" ? organizer.email : "")
    : "";

  return (
    <div className="rounded-lg border border-border bg-accent/20 p-3 space-y-2">
      <div className="flex items-start justify-between gap-2">
        <div className="text-sm font-semibold text-foreground">{summary}</div>
        {status && (
          <span className="shrink-0 rounded-full border border-border bg-background px-2 py-0.5 text-[10px] text-muted-foreground capitalize">
            {status}
          </span>
        )}
      </div>

      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        <span className="font-medium">{start.date}</span>
        <span>{start.time}</span>
        {end && !start.isAllDay && end.time && (
          <>
            <span className="text-muted-foreground/40">→</span>
            <span>{end.time}</span>
          </>
        )}
      </div>

      {location && (
        <div className="text-xs text-muted-foreground">📍 {location}</div>
      )}

      {organizerName && (
        <div className="text-xs text-muted-foreground">Organized by {organizerName}</div>
      )}

      {attendees.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {attendees.map((a, i) => {
            const name = typeof a.displayName === "string" ? a.displayName : typeof a.email === "string" ? a.email : `Attendee ${i + 1}`;
            const responseStatus = typeof a.responseStatus === "string" ? a.responseStatus : "";
            const dot = responseStatus === "accepted" ? "bg-emerald-500"
              : responseStatus === "declined" ? "bg-red-400"
              : responseStatus === "tentative" ? "bg-amber-400"
              : "bg-muted-foreground/30";
            return (
              <span key={`${typeof a.email === "string" ? a.email : i}`} className="inline-flex items-center gap-1 rounded-full border border-border bg-background px-2 py-0.5 text-[10px] text-muted-foreground">
                <span className={`size-1.5 rounded-full ${dot}`} />
                {name}
              </span>
            );
          })}
        </div>
      )}

      {description && (
        <div className="text-xs text-muted-foreground/80 line-clamp-3">{description}</div>
      )}

      {htmlLink && (
        <a href={htmlLink} target="_blank" rel="noreferrer" className="text-[10px] text-sky-300/80 hover:underline">
          Open in Google Calendar
        </a>
      )}
    </div>
  );
}

function EventsList({ items, title }: { items: unknown[]; title: string }) {
  const events = items.filter((e): e is Record<string, unknown> => typeof e === "object" && e !== null);

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-xs text-muted-foreground">
        <span>{title}</span>
        <span>{events.length} event{events.length !== 1 ? "s" : ""}</span>
      </div>
      {events.length === 0 ? (
        <div className="text-sm text-muted-foreground/60 py-2">No events found.</div>
      ) : (
        <div className="space-y-2 max-h-96 overflow-y-auto">
          {events.map((event, i) => (
            <EventCard key={`${String(event.id)}-${i}`} event={event} />
          ))}
        </div>
      )}
    </div>
  );
}

function CalendarListCard({ items }: { items: unknown[] }) {
  const calendars = items.filter((c): c is Record<string, unknown> => typeof c === "object" && c !== null);

  return (
    <div className="space-y-1.5 max-h-60 overflow-y-auto">
      {calendars.map((cal, i) => {
        const summary = typeof cal.summary === "string" ? cal.summary : "Unknown";
        const primary = cal.primary === true;
        const accessRole = typeof cal.accessRole === "string" ? cal.accessRole : "";
        const bgColor = typeof cal.backgroundColor === "string" ? cal.backgroundColor : undefined;

        return (
          <div key={`${String(cal.id)}-${i}`} className="flex items-center gap-2 rounded border border-border bg-accent/10 px-3 py-1.5">
            {bgColor && <span className="size-2.5 rounded-full shrink-0" style={{ backgroundColor: bgColor }} />}
            <span className="text-sm text-foreground flex-1 truncate">{summary}</span>
            {primary && <span className="text-[10px] text-muted-foreground bg-accent/50 px-1.5 py-0.5 rounded">Primary</span>}
            {accessRole && <span className="text-[10px] text-muted-foreground">{accessRole}</span>}
          </div>
        );
      })}
    </div>
  );
}

function CreatedEventCard({ data, message }: { data: Record<string, unknown>; message: string }) {
  return (
    <div className="rounded-lg border border-border bg-accent/20 p-3 space-y-1.5">
      <div className="text-sm text-foreground">{message}</div>
      {typeof data.htmlLink === "string" && (
        <a href={data.htmlLink} target="_blank" rel="noreferrer" className="text-[10px] text-sky-300/80 hover:underline">
          Open in Google Calendar
        </a>
      )}
    </div>
  );
}

function SimpleConfirmationCard({ message }: { message: string }) {
  return (
    <div className="rounded-lg border border-border bg-accent/20 px-3 py-2 text-sm text-foreground">
      {message}
    </div>
  );
}

// ── Main Component ───────────────────────────────────────────────────────

export function CalendarTool({ part }: ToolRendererProps) {
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
        {part.errorText || "Calendar operation failed"}
      </div>
    );
  }

  if (part.state !== "output-available") return null;

  switch (toolName) {
    case "upcoming_events": {
      const items = tryParseArray(part.output) ?? [];
      return (
        <div className="w-full">
          <EventsList items={items} title="Upcoming events" />
        </div>
      );
    }

    case "search_events": {
      const items = tryParseArray(part.output) ?? [];
      return (
        <div className="w-full">
          <EventsList items={items} title="Search results" />
        </div>
      );
    }

    case "get_event": {
      const data = tryParseJSON(part.output);
      if (!data) break;
      return (
        <div className="w-full">
          <EventCard event={data} />
        </div>
      );
    }

    case "create_event": {
      const successMsg = extractSuccessMessage(part.output) || "Event created.";
      const data = tryParseJSON(part.output);
      return (
        <div className="w-full">
          <CreatedEventCard data={data ?? {}} message={successMsg} />
        </div>
      );
    }

    case "update_event":
    case "delete_event": {
      const successMsg = extractSuccessMessage(part.output) || (toolName === "delete_event" ? "Event deleted." : "Event updated.");
      return (
        <div className="w-full">
          <SimpleConfirmationCard message={successMsg} />
        </div>
      );
    }

    case "list_calendars": {
      const items = tryParseArray(part.output) ?? [];
      return (
        <div className="w-full">
          <CalendarListCard items={items} />
        </div>
      );
    }
  }

  // Fallback
  const data = tryParseJSON(part.output);
  if (data) {
    return (
      <div className="w-full overflow-hidden rounded border border-border bg-accent/20">
        <CodeBlock code={JSON.stringify(data, null, 2)} language="json" />
      </div>
    );
  }

  return (
    <div className="w-full rounded-lg border border-border bg-accent/20 px-3 py-2 text-sm text-foreground">
      {typeof part.output === "string" ? part.output : "Done"}
    </div>
  );
}
