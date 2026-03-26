import { html, reactive } from '@arrow-js/core'

const raw = `{{DATA}}`
let data = {}
try { data = JSON.parse(raw) } catch {}

function getInitial(name) {
  return name ? name[0].toUpperCase() : '?'
}

function hashColor(s) {
  let h = 0; for (const c of (s || '')) h += c.charCodeAt(0)
  const colors = ['#f43f5e','#0ea5e9','#10b981','#8b5cf6','#f59e0b','#14b8a6','#ec4899','#6366f1']
  return colors[h % colors.length]
}

const name = data.name || data.display_name || ''
const emails = Array.isArray(data.emails) ? data.emails : (data.email ? [data.email] : [])
const phones = Array.isArray(data.phones) ? data.phones : (data.phone ? [data.phone] : [])
const color = hashColor(emails[0] || name)

const state = reactive({ name, emails, phones, color })

export default html`
  <div style="font-family:system-ui,sans-serif;border:1px solid rgba(255,255,255,.08);border-radius:16px;padding:16px;">
    <div style="display:flex;align-items:center;gap:12px;">
      <div style="width:44px;height:44px;border-radius:50%;background:${state.color}22;color:${state.color};display:flex;align-items:center;justify-content:center;font-size:18px;font-weight:600;flex-shrink:0;">${getInitial(state.name)}</div>
      <div>
        <div style="font-size:14px;font-weight:600;color:rgba(255,255,255,.9);">${state.name || 'Unknown'}</div>
        ${() => state.emails.map(e => html`<div style="font-size:12px;color:rgba(255,255,255,.45);">${e}</div>`)}
      </div>
    </div>
    ${() => state.phones.length > 0 ? html`<div style="margin-top:12px;padding-top:12px;border-top:1px solid rgba(255,255,255,.06);">
      ${state.phones.map(p => html`<div style="font-size:12px;color:rgba(255,255,255,.5);">${p}</div>`)}
    </div>` : ''}
  </div>
`
