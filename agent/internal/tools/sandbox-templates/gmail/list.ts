import { html, reactive } from '@arrow-js/core'

const raw = `{{DATA}}`
let data = {}
try { data = JSON.parse(raw) } catch {}

const messages = Array.isArray(data.messages) ? data.messages : []
const estimate = Number(data.resultSizeEstimate) || messages.length

function parseFrom(raw) {
  if (!raw) return { name: '?', email: '' }
  const m = raw.match(/^(.+?)\s*<(.+?)>$/)
  if (m) return { name: m[1].replace(/^["']|["']$/g,'').trim(), email: m[2].trim() }
  return { name: '', email: raw.trim() }
}

function getHeaders(payload) {
  const h = {}
  if (!payload || !Array.isArray(payload.headers)) return h
  for (const x of payload.headers) if (x && x.name && x.value) h[x.name.toLowerCase()] = x.value
  return h
}

function getInitial(s) {
  if (!s) return '?'
  return s[0].toUpperCase()
}

function hashColor(s) {
  let h = 0; for (const c of s) h += c.charCodeAt(0)
  const colors = ['#f43f5e','#0ea5e9','#10b981','#8b5cf6','#f59e0b','#14b8a6','#ec4899','#6366f1']
  return colors[h % colors.length]
}

function isUnread(msg) {
  return Array.isArray(msg.labelIds) && msg.labelIds.includes('UNREAD')
}

const state = reactive({ messages, estimate })

export default html`
  <div style="font-family:system-ui,sans-serif;">
    <div style="display:flex;justify-content:space-between;align-items:center;padding:0 4px 8px;">
      <span style="font-size:12px;font-weight:500;color:rgba(255,255,255,.6);">Messages</span>
      <span style="font-size:11px;color:rgba(255,255,255,.35);">${() => state.messages.length} results${() => state.estimate > state.messages.length ? ' (~' + state.estimate + ' total)' : ''}</span>
    </div>
    ${() => state.messages.length === 0
      ? html`<div style="text-align:center;padding:32px;color:rgba(255,255,255,.3);font-size:13px;">No messages found.</div>`
      : html`<div style="border:1px solid rgba(255,255,255,.06);border-radius:12px;overflow:hidden;">
        ${() => state.messages.map((msg, i) => {
          const h = getHeaders(msg.payload)
          const from = parseFrom(h.from || '')
          const subject = h.subject || '(no subject)'
          const snippet = msg.snippet || ''
          const unread = isUnread(msg)
          const color = hashColor(from.email || from.name)
          return html`
            <div style="display:flex;align-items:flex-start;gap:12px;padding:10px 12px;${unread ? 'background:rgba(56,189,248,.04);' : ''}border-bottom:1px solid rgba(255,255,255,.04);" .key="${msg.id || i}">
              <div style="width:28px;height:28px;border-radius:50%;background:${color}22;color:${color};display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:600;flex-shrink:0;">${getInitial(from.name || from.email)}</div>
              <div style="min-width:0;flex:1;">
                <div style="display:flex;align-items:baseline;gap:6px;">
                  <span style="font-size:12px;${unread ? 'font-weight:600;color:rgba(255,255,255,.9);' : 'font-weight:500;color:rgba(255,255,255,.7);'}">${from.name || from.email}</span>
                  ${unread ? html`<span style="width:6px;height:6px;border-radius:50%;background:#38bdf8;flex-shrink:0;"></span>` : ''}
                </div>
                <div style="font-size:12px;margin-top:2px;${unread ? 'font-weight:500;color:rgba(255,255,255,.8);' : 'color:rgba(255,255,255,.6);'}overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${subject}</div>
                ${snippet ? html`<div style="font-size:11px;margin-top:2px;color:rgba(255,255,255,.35);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${snippet}</div>` : ''}
              </div>
            </div>`
        })}
      </div>`
    }
  </div>
`
