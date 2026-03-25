package integrations

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/suryaumapathy2812/core-ai/agent/internal/db"
)

// rcloneToken is the token format rclone expects in rclone.conf.
type rcloneToken struct {
	AccessToken  string `json:"access_token"`
	TokenType    string `json:"token_type"`
	RefreshToken string `json:"refresh_token"`
	Expiry       string `json:"expiry"`
}

// GenerateRcloneConfig writes an rclone.conf from the google-workspace integration
// credentials stored in the database. This allows rclone to authenticate with
// Google Drive using the same OAuth credentials the agent already manages.
func GenerateRcloneConfig(ctx context.Context, store *db.Store) error {
	state := NewPluginState(store, "google-workspace")

	enabled, err := state.Enabled(ctx)
	if err != nil {
		return fmt.Errorf("failed to check integration status: %w", err)
	}
	if !enabled {
		return fmt.Errorf("google-workspace integration is not enabled")
	}

	cfg, err := state.Config(ctx)
	if err != nil {
		return fmt.Errorf("failed to read integration config: %w", err)
	}
	if cfg == nil {
		cfg = map[string]string{}
	}

	clientID := strings.TrimSpace(cfg["client_id"])
	clientSecret := maybeDecryptOrPlain(state, cfg["client_secret"])
	accessToken := maybeDecryptOrPlain(state, cfg["access_token"])
	refreshToken := maybeDecryptOrPlain(state, cfg["refresh_token"])

	if clientID == "" || clientSecret == "" {
		return fmt.Errorf("missing client_id or client_secret for google-workspace")
	}
	if refreshToken == "" {
		return fmt.Errorf("missing refresh_token for google-workspace")
	}

	// Determine token expiry — use stored value or default to 1 hour from now.
	expiry := time.Now().Add(1 * time.Hour).UTC().Format(time.RFC3339)
	if storedExpiry := strings.TrimSpace(cfg["expires_at"]); storedExpiry != "" {
		if t, err := time.Parse(time.RFC3339, storedExpiry); err == nil {
			expiry = t.UTC().Format(time.RFC3339)
		}
	}

	tokenJSON, err := json.Marshal(rcloneToken{
		AccessToken:  accessToken,
		TokenType:    "Bearer",
		RefreshToken: refreshToken,
		Expiry:       expiry,
	})
	if err != nil {
		return fmt.Errorf("failed to marshal rclone token: %w", err)
	}

	// Escape values for INI format.
	iniContent := fmt.Sprintf(`[gdrive]
type = drive
client_id = %s
client_secret = %s
token = %s
scope = drive
root_folder_id =
team_drive =
`,
		escapeINI(clientID),
		escapeINI(clientSecret),
		escapeINI(string(tokenJSON)),
	)

	configDir := filepath.Join(os.Getenv("HOME"), ".config", "rclone")
	if err := os.MkdirAll(configDir, 0o700); err != nil {
		return fmt.Errorf("failed to create rclone config dir: %w", err)
	}

	configPath := filepath.Join(configDir, "rclone.conf")
	if err := os.WriteFile(configPath, []byte(iniContent), 0o600); err != nil {
		return fmt.Errorf("failed to write rclone.conf: %w", err)
	}

	log.Printf("rclone: wrote config to %s", configPath)
	return nil
}

// maybeDecryptOrPlain returns a decrypted value if it's encrypted, otherwise returns as-is.
func maybeDecryptOrPlain(state PluginState, value string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return ""
	}
	if strings.HasPrefix(value, "enc:v1:") {
		decrypted, err := state.DecryptString(value)
		if err != nil {
			log.Printf("rclone: warning: failed to decrypt value: %v", err)
			return ""
		}
		return decrypted
	}
	return value
}

// escapeINI escapes a value for INI file format.
func escapeINI(s string) string {
	if strings.ContainsAny(s, "\n\r") {
		// Wrap multi-line values.
		return `"` + strings.ReplaceAll(s, `"`, `\"`) + `"`
	}
	return s
}
