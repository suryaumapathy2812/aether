import { html, reactive } from '@arrow-js/core'

const raw = `{{DATA}}`
let data = {}
try { data = JSON.parse(raw) } catch {}

function getHeaders(payload) {
  const h = {}
  if (!payload || !Array.isArray(payload.headers)) return h
  for (const x of payload.headers) if (x && x.name && x.value) h[x.name.toLowerCase()] = x.value
  return h
}

function parseFrom(raw) {
  if (!raw) return { name: '?', email: '' }
  const m = raw.match(/^(.+?)\s*<(.+?)>$/)
  if (m) return { name: m[1].replace(/^["']|["']$/g,'').trim(), email: m[2].trim() }
  return { name: '', email: raw.trim() }
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

function decodeBody(payload) {
  if (!payload) return { text: '', html: '' }
  const b = payload.body
  if (b && b.data) return { text: atob(b.data.replace(/-/g,'+').replace(/_/g,'/')), html: '' }
  if (Array.isArray(payload.parts)) {
    let text = '', htmlContent = ''
    for (const p of payload.parts) {
      if (p.mimeType === 'text/plain' && p.body?.data && !text) text = atob(p.body.data.replace(/-/g,'+').replace(/_/g,'/'))
      if (p.mimeType === 'text/html' && p.body?.data && !htmlContent) htmlContent = atob(p.body.data.replace(/-/g,'+').replace(/_/g,'/'))
    }
    return { text, html: htmlContent }
  }
  return { text: '', html: '' }
}

function formatDate(d) {
  if (!d) return ''
  const dt = new Date(d)
  if (isNaN(dt)) return d
  return dt.toLocaleString(undefined, { month:'short', day:'numeric', year:'numeric', hour:'numeric', minute:'2-digit' })
}

const headers = getHeaders(data.payload)
const from = parseFrom(headers.from || 'Unknown')
const to = headers.to || ''
const cc = headers.cc || ''
const subject = headers.subject || '(no subject)'
const date = headers.date || ''
const body = decodeBody(data.payload)
const labels = Array.isArray(data.labelIds) ? data.labelIds : []
const userLabels = labels.filter(l => l.startsWith('Label_'))
const color = hashColor(from.email || from.name)

const state = reactive({ from, to, cc, subject, date, body, labels, userLabels, color, snippet: data.snippet || '' })

export default html`
  <div style="font-family:system-ui,sans-serif;border:1px solid rgba(255,255,255,.08);border-radius:16px;overflow:hidden;">
    <!-- Subject -->
    <div style="padding:16px;border-bottom:1px solid rgba(255,255,255,.06);">
      <h3 style="margin:0;font-size:14px;font-weight:600;color:rgba(255,255,255,.9);line-height:1.4;">${state.subject}</h3>
      ${() => state.userLabels.length > 0 ? html`<div style="display:flex;gap:4px;margin-top:8px;flex-wrap:wrap;">
        ${state.userLabels.map(l => html`<span style="font-size:10px;padding:2px 8px;border-radius:9999px;background:rgba(255,255,255,.06);color:rgba(255,255,255,.5);">${l.replace('Label_','')}</span>`)}
      </div>` : ''}
    </div>
    <!-- Sender -->
    <div style="display:flex;align-items:flex-start;gap:12px;padding:16px;border-bottom:1px solid rgba(255,255,255,.04);">
      <div style="width:36px;height:36px;border-radius:50%;background:${state.color}22;color:${state.color};display:flex;align-items:center;justify-content:center;font-size:14px;font-weight:600;flex-shrink:0;">${getInitial(state.from.name || state.from.email)}</div>
      <div style="min-width:0;flex:1;">
        <div style="display:flex;align-items:baseline;gap:8px;flex-wrap:wrap;">
          <span style="font-size:13px;font-weight:500;color:rgba(255,255,255,.85);">${state.from.name || state.from.email}</span>
          ${state.from.name ? html`<span style="font-size:11px;color:rgba(255,255,255,.4);">${state.from.email}</span>` : ''}
        </div>
        <div style="font-size:11px;color:rgba(255,255,255,.4);margin-top:2px;">
          to ${state.to ? state.to.split(',')[0].replace(/<[^>]+>/g,'').trim() : 'me'}
          ${state.to && state.to.split(',').length > 1 ? html`<span style="color:rgba(255,255,255,.25);"> +${state.to.split(',').length - 1}</span>` : ''}
          ${state.cc ? html`<span style="color:rgba(255,255,255,.25);"> cc ${state.cc.split(',').length}</span>` : ''}
        </div>
      </div>
      <span style="font-size:11px;color:rgba(255,255,255,.4);white-space:nowrap;">${formatDate(state.date)}</span>
    </div>
    <!-- Body -->
    <div style="padding:16px;">
      ${() => state.body.text
        ? html`<div style="font-size:13px;color:rgba(255,255,255,.8);white-space:pre-wrap;word-break:break-word;line-height:1.6;">${state.body.text}</div>`
        : state.body.html
          ? html`<div style="font-size:13px;color:rgba(255,255,255,.8);line-height:1.6;" .innerHTML="${state.body.html}"></div>`
          : state.snippet
            ? html`<div style="font-size:13px;color:rgba(255,255,255,.4);font-style:italic;">${state.snippet}</div>`
            : html`<div style="font-size:13px;color:rgba(255,255,255,.25);">No message content.</div>`
      }
    </div>
  </div>
`
