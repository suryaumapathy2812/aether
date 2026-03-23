package caddy

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"

	"github.com/jackc/pgx/v5/pgxpool"
)

type RouteManager struct {
	adminURL string
	domain   string
	client   *http.Client
}

func NewRouteManager(adminURL, domain string) *RouteManager {
	return &RouteManager{
		adminURL: strings.TrimRight(strings.TrimSpace(adminURL), "/"),
		domain:   strings.Trim(strings.TrimSpace(domain), "."),
		client:   &http.Client{},
	}
}

func (rm *RouteManager) Enabled() bool {
	return rm != nil && rm.adminURL != "" && rm.domain != ""
}

func (rm *RouteManager) AddRoute(prefix, agentHost string, agentPort int) error {
	if !rm.Enabled() {
		return nil
	}
	host := strings.TrimSpace(agentHost)
	pfx := strings.TrimSpace(prefix)
	if pfx == "" || host == "" || agentPort <= 0 {
		return fmt.Errorf("prefix, agent host and agent port are required")
	}
	route := map[string]any{
		"@id": routeID(pfx),
		"match": []map[string]any{{
			"host": []string{rm.hostnameForPrefix(pfx)},
		}},
		"handle": []map[string]any{{
			"handler": "reverse_proxy",
			"upstreams": []map[string]any{{
				"dial": host + ":" + strconv.Itoa(agentPort),
			}},
			"flush_interval": -1,
		}},
		"terminal": true,
	}
	return rm.sendConfigRequest(http.MethodPut, "/id/"+routeID(pfx), route)
}

func (rm *RouteManager) RemoveRoute(prefix string) error {
	if !rm.Enabled() {
		return nil
	}
	pfx := strings.TrimSpace(prefix)
	if pfx == "" {
		return nil
	}
	req, err := http.NewRequest(http.MethodDelete, rm.adminURL+"/id/"+routeID(pfx), nil)
	if err != nil {
		return err
	}
	resp, err := rm.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusNotFound {
		return nil
	}
	if resp.StatusCode >= 300 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		return fmt.Errorf("caddy delete route failed: status=%d body=%s", resp.StatusCode, strings.TrimSpace(string(body)))
	}
	return nil
}

func (rm *RouteManager) SyncRoutes(ctx context.Context, db *pgxpool.Pool) error {
	if !rm.Enabled() || db == nil {
		return nil
	}
	rows, err := db.Query(ctx, `
		SELECT subdomain_prefix, host, port
		FROM agents
		WHERE status = 'running' AND subdomain_prefix IS NOT NULL AND subdomain_prefix <> ''
	`)
	if err != nil {
		return err
	}
	defer rows.Close()
	for rows.Next() {
		var prefix string
		var host string
		var port int
		if err := rows.Scan(&prefix, &host, &port); err != nil {
			return err
		}
		if err := rm.AddRoute(prefix, host, port); err != nil {
			return err
		}
	}
	return rows.Err()
}

func (rm *RouteManager) hostnameForPrefix(prefix string) string {
	return strings.TrimSpace(prefix) + "." + rm.domain
}

func routeID(prefix string) string {
	return "aether-direct-" + strings.TrimSpace(prefix)
}

func (rm *RouteManager) sendConfigRequest(method, path string, payload any) error {
	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	req, err := http.NewRequest(method, rm.adminURL+path, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := rm.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		respBody, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		return fmt.Errorf("caddy config request failed: status=%d body=%s", resp.StatusCode, strings.TrimSpace(string(respBody)))
	}
	return nil
}
