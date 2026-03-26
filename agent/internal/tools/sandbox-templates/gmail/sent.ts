import { html, reactive } from '@arrow-js/core'

const raw = `{{DATA}}`
let data = {}
try { data = JSON.parse(raw) } catch {}

function parseFrom(raw) {
  if (!raw) return []
  return raw.split(',').map(s => {
    const m = s.trim().match(/^(.+?)\s*<(.+?)>$/)
    if (m) return m[1].replace(/^["']|["']$/g,'').trim()
    return s.trim()
  })
}

let toList = []
let subject = ''
try {
  const h = {}
  if (data.payload && Array.isArray(data.payload.headers)) {
    for (const x of data.payload.headers) if (x && x.name && x.value) h[x.name.toLowerCase()] = x.value
  }
  toList = parseFrom(h.to || '')
  subject = h.subject || ''
} catch {}

const state = reactive({ toList, subject, id: data.id || '', threadId: data.threadId || '' })

export default html`
  <div style="font-family:system-ui,sans-serif;border:1px solid rgba(255,255,255,.08);border-radius:16px;padding:16px;display:flex;align-items:flex-start;gap:12px;">
    <div style="width:32px;height:32px;border-radius:50%;background:rgba(16,185,129,.1);color:#10b981;display:flex;align-items:center;justify-content:center;flex-shrink:0;">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"></polyline></svg>
    </div>
    <div style="min-width:0;flex:1;">
      <div style="font-size:13px;font-weight:500;color:rgba(255,255,255,.85);">Message sent</div>
      ${() => state.toList.length > 0 ? html`<div style="font-size:12px;color:rgba(255,255,255,.4);margin-top:4px;">To: ${state.toList.join(', ')}</div>` : ''}
      ${() => state.subject ? html`<div style="font-size:12px;color:rgba(255,255,255,.35);margin-top:2px;">${state.subject}</div>` : ''}
    </div>
  </div>
`
