import { html, reactive } from '@arrow-js/core'

const raw = `{{DATA}}`
let data = {}
try { data = JSON.parse(raw) } catch {}

const contacts = Array.isArray(data.results) ? data.results : (Array.isArray(data) ? data : [])

function getInitial(c) {
  const name = c.name || c.display_name || c.email || ''
  return name ? name[0].toUpperCase() : '?'
}

function hashColor(s) {
  let h = 0; for (const c of (s || '')) h += c.charCodeAt(0)
  const colors = ['#f43f5e','#0ea5e9','#10b981','#8b5cf6','#f59e0b','#14b8a6','#ec4899','#6366f1']
  return colors[h % colors.length]
}

const state = reactive({ contacts })

export default html`
  <div style="font-family:system-ui,sans-serif;">
    <div style="padding:0 4px 8px;font-size:12px;font-weight:500;color:rgba(255,255,255,.6);">
      ${state.contacts.length} contact${state.contacts.length !== 1 ? 's' : ''}
    </div>
    ${() => state.contacts.length === 0
      ? html`<div style="text-align:center;padding:32px;color:rgba(255,255,255,.3);font-size:13px;">No contacts found.</div>`
      : html`<div style="display:flex;flex-direction:column;gap:6px;">
        ${state.contacts.map((c, i) => {
          const name = c.name || c.display_name || ''
          const email = c.email || (Array.isArray(c.emails) ? c.emails[0] : '') || ''
          const color = hashColor(email || name)
          return html`
            <div style="display:flex;align-items:center;gap:10px;padding:8px 12px;border:1px solid rgba(255,255,255,.06);border-radius:10px;" .key="${c.resourceName || i}">
              <div style="width:28px;height:28px;border-radius:50%;background:${color}22;color:${color};display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:600;flex-shrink:0;">${getInitial(c)}</div>
              <div style="min-width:0;flex:1;">
                <div style="font-size:12px;font-weight:500;color:rgba(255,255,255,.8);">${name || 'Unknown'}</div>
                ${email ? html`<div style="font-size:11px;color:rgba(255,255,255,.4);">${email}</div>` : ''}
              </div>
            </div>`
        })}
      </div>`
    }
  </div>
`
