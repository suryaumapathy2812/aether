import { html, reactive } from '@arrow-js/core'

const raw = `{{DATA}}`
let data = []
try { data = JSON.parse(raw) } catch {}
if (!Array.isArray(data)) data = []

function parseDT(dt) {
  if (!dt) return { date: '', time: '', allDay: false }
  if (dt.dateTime) {
    const d = new Date(dt.dateTime)
    return {
      date: d.toLocaleDateString(undefined, { weekday:'short', month:'short', day:'numeric' }),
      time: d.toLocaleTimeString(undefined, { hour:'numeric', minute:'2-digit' }),
      allDay: false
    }
  }
  if (dt.date) {
    const d = new Date(dt.date + 'T00:00:00')
    return { date: d.toLocaleDateString(undefined, { weekday:'short', month:'short', day:'numeric' }), time: 'All day', allDay: true }
  }
  return { date: 'Unknown', time: '', allDay: false }
}

const state = reactive({ items: data })

export default html`
  <div style="font-family:system-ui,sans-serif;">
    <div style="padding:0 4px 8px;font-size:12px;font-weight:500;color:rgba(255,255,255,.6);">
      ${state.items.length} event${state.items.length !== 1 ? 's' : ''}
    </div>
    ${() => state.items.length === 0
      ? html`<div style="text-align:center;padding:32px;color:rgba(255,255,255,.3);font-size:13px;">No events found.</div>`
      : html`<div style="display:flex;flex-direction:column;gap:8px;">
        ${state.items.map((ev, i) => {
          const start = parseDT(ev.start)
          const end = parseDT(ev.end)
          return html`
            <div style="border:1px solid rgba(255,255,255,.08);border-radius:12px;padding:14px;" .key="${ev.id || i}">
              <div style="font-size:13px;font-weight:600;color:rgba(255,255,255,.9);">${ev.summary || '(No title)'}</div>
              <div style="display:flex;align-items:center;gap:6px;margin-top:6px;font-size:12px;color:rgba(255,255,255,.5);">
                <span style="font-weight:500;">${start.date}</span>
                <span>${start.time}</span>
                ${!start.allDay && end.time ? html`<span style="color:rgba(255,255,255,.25);">→</span><span>${end.time}</span>` : ''}
              </div>
              ${ev.location ? html`<div style="font-size:11px;color:rgba(255,255,255,.4);margin-top:4px;">${ev.location}</div>` : ''}
              ${ev.htmlLink ? html`<a href="${ev.htmlLink}" target="_blank" rel="noreferrer" style="font-size:10px;color:#38bdf8;margin-top:6px;display:inline-block;text-decoration:none;">Open in Calendar →</a>` : ''}
            </div>`
        })}
      </div>`
    }
  </div>
`
