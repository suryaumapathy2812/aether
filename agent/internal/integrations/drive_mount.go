package integrations

import (
	"bytes"
	"context"
	"fmt"
	"log"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/suryaumapathy2812/core-ai/agent/internal/db"
)

// DriveMountManager manages the rclone FUSE mount for Google Drive.
type DriveMountManager struct {
	store    *db.Store
	mountDir string
	remote   string
}

// NewDriveMountManager creates a new mount manager.
// mountDir is the local path where Google Drive will be mounted.
func NewDriveMountManager(store *db.Store, mountDir string) *DriveMountManager {
	if mountDir == "" {
		mountDir = "/workspace/gdrive"
	}
	return &DriveMountManager{
		store:    store,
		mountDir: mountDir,
		remote:   "gdrive:/",
	}
}

// Mount starts the rclone FUSE mount in the background.
// It generates the rclone config first, then mounts.
func (m *DriveMountManager) Mount(ctx context.Context) error {
	if m.store == nil {
		return fmt.Errorf("store is nil")
	}

	// Check if google-workspace is enabled.
	state := NewPluginState(m.store, "google-workspace")
	enabled, err := state.Enabled(ctx)
	if err != nil || !enabled {
		return fmt.Errorf("google-workspace integration is not enabled")
	}

	// Generate rclone config from integration credentials.
	if err := GenerateRcloneConfig(ctx, m.store); err != nil {
		return fmt.Errorf("failed to generate rclone config: %w", err)
	}

	// Verify rclone is available.
	if _, err := exec.LookPath("rclone"); err != nil {
		log.Printf("drive mount: rclone not found in PATH, skipping mount")
		return nil
	}

	// Verify config works.
	if err := m.verifyConfig(ctx); err != nil {
		return fmt.Errorf("rclone config verification failed: %w", err)
	}

	// Check if already mounted.
	if m.isMounted() {
		log.Printf("drive mount: already mounted at %s", m.mountDir)
		return nil
	}

	// Create mount point.
	if err := os.MkdirAll(m.mountDir, 0o755); err != nil {
		return fmt.Errorf("failed to create mount dir: %w", err)
	}

	// Unmount any stale mount first.
	_ = m.unmount()

	log.Printf("drive mount: mounting %s at %s", m.remote, m.mountDir)

	mountCmd := exec.CommandContext(ctx, "rclone", "mount",
		m.remote, m.mountDir,
		"--vfs-cache-mode", "writes",
		"--vfs-cache-max-age", "1h",
		"--vfs-cache-max-size", "500M",
		"--dir-cache-time", "1m",
		"--vfs-read-chunk-size", "8M",
		"--vfs-read-ahead", "16M",
		"--buffer-size", "16M",
		"--tpslimit", "10",
		"--poll-interval", "30s",
		"--allow-other",
		"--daemon",
		"--log-level", "NOTICE",
	)

	var stderr bytes.Buffer
	mountCmd.Stderr = &stderr

	if err := mountCmd.Run(); err != nil {
		return fmt.Errorf("failed to mount: %s: %w", strings.TrimSpace(stderr.String()), err)
	}

	// Wait a moment for mount to be ready.
	time.Sleep(1 * time.Second)

	if !m.isMounted() {
		return fmt.Errorf("mount started but %s is not mounted", m.mountDir)
	}

	log.Printf("drive mount: mounted successfully at %s", m.mountDir)
	return nil
}

// Unmount stops the rclone FUSE mount.
func (m *DriveMountManager) Unmount() error {
	if !m.isMounted() {
		return nil
	}
	log.Printf("drive mount: unmounting %s", m.mountDir)
	return m.unmount()
}

// Status returns the current mount status.
func (m *DriveMountManager) Status() MountStatus {
	mounted := m.isMounted()
	status := MountStatus{
		Mounted:  mounted,
		MountDir: m.mountDir,
		Remote:   m.remote,
	}
	if mounted {
		status.Healthy = m.checkHealth()
	}
	return status
}

// MountStatus holds the status of the drive mount.
type MountStatus struct {
	Mounted  bool   `json:"mounted"`
	Healthy  bool   `json:"healthy"`
	MountDir string `json:"mount_dir"`
	Remote   string `json:"remote"`
}

func (m *DriveMountManager) isMounted() bool {
	cmd := exec.Command("mountpoint", "-q", m.mountDir)
	return cmd.Run() == nil
}

func (m *DriveMountManager) unmount() error {
	cmd := exec.Command("fusermount", "-u", m.mountDir)
	return cmd.Run()
}

func (m *DriveMountManager) verifyConfig(ctx context.Context) error {
	cmd := exec.CommandContext(ctx, "rclone", "about", "gdrive:")
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("rclone about failed: %s: %w", strings.TrimSpace(stderr.String()), err)
	}
	return nil
}

func (m *DriveMountManager) checkHealth() bool {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "rclone", "about", "gdrive:")
	return cmd.Run() == nil
}
