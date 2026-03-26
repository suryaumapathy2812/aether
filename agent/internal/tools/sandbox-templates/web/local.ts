import { html, reactive } from '@arrow-js/core'

const raw = `{{DATA}}`
let data = {}
try { data = JSON.parse(raw) } catch {}

const places = Array.isArray(data.places) ? data.places : []

const state = reactive({ places })

export default html`
  <div style="font-family:system-ui,sans-serif;display:flex;flex-direction:column;gap:8px;">
    ${() => state.places.map((p, i) => html`
      <div style="border:1px solid rgba(255,255,255,.08);border-radius:12px;padding:12px;" .key="${i}">
        <div style="font-size:13px;font-weight:600;color:rgba(255,255,255,.9);">${p.name || 'Place'}</div>
        ${p.address ? html`<div style="font-size:11px;color:rgba(255,255,255,.45);margin-top:2px;">${p.address}</div>` : ''}
        ${p.rating != null ? html`<div style="font-size:11px;color:rgba(255,255,255,.4);margin-top:2px;">Rating ${p.rating.toFixed(1)}${p.user_rating_count ? ' (' + p.user_rating_count + ' reviews)' : ''}</div>` : ''}
        ${p.phone ? html`<div style="font-size:11px;color:rgba(255,255,255,.4);margin-top:2px;">${p.phone}</div>` : ''}
        ${p.maps_url ? html`<a href="${p.maps_url}" target="_blank" rel="noreferrer" style="font-size:10px;color:#38bdf8;margin-top:4px;display:inline-block;text-decoration:none;">Open in Maps →</a>` : ''}
      </div>`
    )}
  </div>
`
