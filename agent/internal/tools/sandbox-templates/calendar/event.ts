import { html, reactive } from '@arrow-js/core'

const raw = `{{DATA}}`
let data = {}
try { data = JSON.parse(raw) } catch {}

function parseDT(dt) {
  if (!dt) return { date: '', time: '', allDay: false }
  if (dt.dateTime) {
    const d = new Date(dt.dateTime)
    return {
      date: d.toLocaleDateString(undefined, { weekday:'long', month:'long', day:'numeric', year:'numeric' }),
      time: d.toLocaleTimeString(undefined, { hour:'numeric', minute:'2-digit' }),
      allDay: false
    }
  }
  if (dt.date) {
    const d = new Date(dt.date + 'T00:00:00')
    return { date: d.toLocaleDateString(undefined, { weekday:'long', month:'long', day:'numeric', year:'numeric' }), time: 'All day', allDay: true }
  }
  return { date: 'Unknown', time: '', allDay: false }
}

const start = parseDT(data.start)
const end = parseDT(data.end)
const attendees = Array.isArray(data.attendees) ? data.attendees : []

const state = reactive({ event: data, start, end, attendees })

export default html`
  <div style="font-family:system-ui,sans-serif;border:1px solid rgba(255,255,255,.08);border-radius:16px;padding:16px;">
    <div style="font-size:15px;font-weight:600;color:rgba(255,255,255,.9);">${data.summary || '(No title)'}</div>
    <div style="display:flex;align-items:center;gap:6px;margin-top:8px;font-size:12px;color:rgba(255,255,255,.5);">
      <span style="font-weight:500;">${state.start.date}</span>
      <span>${state.start.time}</span>
      ${!state.start.allDay && state.end.time ? html`<span style="color:rgba(255,255,255,.25);">→</span><span>${state.end.time}</span>` : ''}
    </div>
    ${data.location ? html`<div style="font-size:12px;color:rgba(255,255,255,.45);margin-top:6px;">${data.location}</div>` : ''}
    ${() => state.attendees.length > 0 ? html`<div style="display:flex;flex-wrap:wrap;gap:4px;margin-top:8px;">
      ${state.attendees.map((a, i) => {
        const name = a.displayName || a.email || 'Attendee ' + (i+1)
        const dot = a.responseStatus === 'accepted' ? '#10b981' : a.responseStatus === 'declined' ? '#f87171' : a.responseStatus === 'tentative' ? '#fbbf24' : 'rgba(255,255,255,.2)'
        return html`<span style="display:inline-flex;align-items:center;gap:4px;padding:2px 8px;border-radius:9999px;border:1px solid rgba(255,255,255,.08);font-size:10px;color:rgba(255,255,255,.5);" .key="${a.email || i}"><span style="width:6px;height:6px;border-radius:50%;background:${dot};"></span>${name}</span>`
      })}
    </div>` : ''}
    ${data.description ? html`<div style="font-size:12px;color:rgba(255,255,255,.4);margin-top:8px;overflow:hidden;display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;">${data.description}</div>` : ''}
    ${data.htmlLink ? html`<a href="${data.htmlLink}" target="_blank" rel="noreferrer" style="font-size:10px;color:#38bdf8;margin-top:8px;display:inline-block;text-decoration:none;">Open in Google Calendar →</a>` : ''}
  </div>
`
