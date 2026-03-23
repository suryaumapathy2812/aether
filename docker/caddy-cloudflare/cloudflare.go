package cloudflare

import (
	"fmt"
	"regexp"

	"github.com/caddyserver/caddy/v2"
	"github.com/caddyserver/caddy/v2/caddyconfig/caddyfile"
	libdnscloudflare "github.com/libdns/cloudflare"
)

// Cloudflare API tokens now commonly use the cfut_ prefix and exceed 50 chars.
var cloudflareTokenRegexp = regexp.MustCompile(`^[A-Za-z0-9_-]{35,120}$`)

type Provider struct{ *libdnscloudflare.Provider }

func init() {
	caddy.RegisterModule(Provider{})
}

func (Provider) CaddyModule() caddy.ModuleInfo {
	return caddy.ModuleInfo{
		ID:  "dns.providers.cloudflare",
		New: func() caddy.Module { return &Provider{new(libdnscloudflare.Provider)} },
	}
}

func (p *Provider) Provision(ctx caddy.Context) error {
	p.Provider.APIToken = caddy.NewReplacer().ReplaceAll(p.Provider.APIToken, "")
	p.Provider.ZoneToken = caddy.NewReplacer().ReplaceAll(p.Provider.ZoneToken, "")
	if !validCloudflareToken(p.Provider.APIToken) {
		return fmt.Errorf("API token '%s' appears invalid; ensure it's correctly entered and not wrapped in braces nor quotes", p.Provider.APIToken)
	}
	return nil
}

func validCloudflareToken(token string) bool {
	return cloudflareTokenRegexp.MatchString(token)
}

func (p *Provider) UnmarshalCaddyfile(d *caddyfile.Dispenser) error {
	d.Next()

	if d.NextArg() {
		p.Provider.APIToken = d.Val()
	} else {
		for nesting := d.Nesting(); d.NextBlock(nesting); {
			switch d.Val() {
			case "api_token":
				if d.NextArg() {
					p.Provider.APIToken = d.Val()
				} else {
					return d.ArgErr()
				}
			case "zone_token":
				if d.NextArg() {
					p.Provider.ZoneToken = d.Val()
				} else {
					return d.ArgErr()
				}
			default:
				return d.Errf("unrecognized subdirective '%s'", d.Val())
			}
		}
	}
	if d.NextArg() {
		return d.Errf("unexpected argument '%s'", d.Val())
	}
	if p.Provider.APIToken == "" {
		return d.Err("missing API token")
	}
	return nil
}

var (
	_ caddyfile.Unmarshaler = (*Provider)(nil)
	_ caddy.Provisioner     = (*Provider)(nil)
)
