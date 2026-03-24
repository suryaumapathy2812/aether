// Package container manages per-user Docker containers for agent sandboxing.
// Each user gets a persistent container with a shared workspace filesystem.
package container

import (
	"bytes"
	"context"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// Config holds the container manager configuration.
type Config struct {
	// Image is the Docker image for sandbox containers.
	Image string

	// BaseDataDir is the host directory where per-user workspace volumes are stored.
	// Each user gets a subdirectory: BaseDataDir/{userID}/workspace
	BaseDataDir string

	// Memory limit per container (e.g., "512m").
	Memory string

	// CPU limit per container (e.g., "0.5").
	CPUs string

	// NetworkMode is the Docker network mode ("host", "bridge", etc.).
	NetworkMode string
}

// DefaultConfig returns sensible defaults for container management.
func DefaultConfig() Config {
	return Config{
		Image:       "aether-sandbox:latest",
		BaseDataDir: "/data/aether/users",
		Memory:      "512m",
		CPUs:        "0.5",
		NetworkMode: "host",
	}
}

// Manager manages per-user Docker containers.
type Manager struct {
	cfg Config
	mu  sync.Mutex
}

// NewManager creates a new container manager.
func NewManager(cfg Config) *Manager {
	if cfg.Image == "" {
		cfg.Image = "aether-sandbox:latest"
	}
	if cfg.BaseDataDir == "" {
		cfg.BaseDataDir = "/data/aether/users"
	}
	if cfg.Memory == "" {
		cfg.Memory = "512m"
	}
	if cfg.CPUs == "" {
		cfg.CPUs = "0.5"
	}
	if cfg.NetworkMode == "" {
		cfg.NetworkMode = "host"
	}
	return &Manager{cfg: cfg}
}

// containerName returns the Docker container name for a user.
func (m *Manager) containerName(userID string) string {
	return "aether-user-" + userID
}

// workspaceDir returns the host workspace path for a user.
func (m *Manager) workspaceDir(userID string) string {
	return filepath.Join(m.cfg.BaseDataDir, userID, "workspace")
}

// Ensure creates the user's container if it doesn't exist.
// If the container exists but is stopped, it starts it.
func (m *Manager) Ensure(ctx context.Context, userID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	name := m.containerName(userID)

	// Check if container already exists.
	exists, err := m.containerExists(ctx, name)
	if err != nil {
		return fmt.Errorf("failed to check container existence: %w", err)
	}
	if exists {
		// Ensure it's running.
		running, err := m.containerRunning(ctx, name)
		if err != nil {
			return fmt.Errorf("failed to check container status: %w", err)
		}
		if !running {
			log.Printf("container: starting stopped container for user=%s", userID)
			return m.startContainer(ctx, name)
		}
		return nil
	}

	// Create workspace directory on host.
	wsDir := m.workspaceDir(userID)
	if err := os.MkdirAll(wsDir, 0o755); err != nil {
		return fmt.Errorf("failed to create workspace dir: %w", err)
	}

	// Initialize workspace subdirectories.
	for _, subdir := range []string{"documents", "downloads", "scripts", "data", "temp"} {
		if err := os.MkdirAll(filepath.Join(wsDir, subdir), 0o755); err != nil {
			return fmt.Errorf("failed to create workspace subdir %s: %w", subdir, err)
		}
	}

	log.Printf("container: creating new container for user=%s image=%s", userID, m.cfg.Image)

	args := []string{
		"run", "-d",
		"--name", name,
		"--restart", "unless-stopped",
		"-v", wsDir + ":/workspace",
		"--memory", m.cfg.Memory,
		"--cpus", m.cfg.CPUs,
		"--network", m.cfg.NetworkMode,
		m.cfg.Image,
	}

	cmd := exec.CommandContext(ctx, "docker", args...)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		return fmt.Errorf("failed to create container: %s: %w", strings.TrimSpace(stderr.String()), err)
	}

	log.Printf("container: created container for user=%s name=%s", userID, name)
	return nil
}

// Exec runs a command inside the user's container with optional environment variables.
// Returns stdout, stderr, and any error.
func (m *Manager) Exec(ctx context.Context, userID string, command string, envVars []string, timeoutSec int) (string, string, error) {
	if err := m.Ensure(ctx, userID); err != nil {
		return "", "", fmt.Errorf("failed to ensure container: %w", err)
	}

	name := m.containerName(userID)

	if timeoutSec <= 0 {
		timeoutSec = 30
	}
	if timeoutSec > 120 {
		timeoutSec = 120
	}

	execCtx, cancel := context.WithTimeout(ctx, time.Duration(timeoutSec)*time.Second)
	defer cancel()

	args := []string{"exec", "-i"}
	for _, env := range envVars {
		args = append(args, "-e", env)
	}
	args = append(args, "-w", "/workspace", name, "bash", "-c", command)

	cmd := exec.CommandContext(execCtx, "docker", args...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	err := cmd.Run()
	outStr := strings.TrimSpace(stdout.String())
	errStr := strings.TrimSpace(stderr.String())

	if execCtx.Err() == context.DeadlineExceeded {
		return outStr, errStr, fmt.Errorf("command timed out after %ds", timeoutSec)
	}
	if err != nil {
		return outStr, errStr, fmt.Errorf("command failed: %s: %w", errStr, err)
	}
	return outStr, errStr, nil
}

// ExecRaw runs a raw docker exec command and returns the result.
// This matches the interface expected by the execute tool.
func (m *Manager) ExecRaw(ctx context.Context, userID string, command string, envVars []string, timeoutSec int) (ExecResult, error) {
	stdout, stderr, err := m.Exec(ctx, userID, command, envVars, timeoutSec)
	result := ExecResult{
		Stdout: stdout,
		Stderr: stderr,
	}
	if err != nil {
		result.Error = err.Error()
		result.ExitCode = 1
		return result, err
	}
	result.ExitCode = 0
	return result, nil
}

// ExecResult holds the result of a container command execution.
type ExecResult struct {
	Stdout   string `json:"stdout"`
	Stderr   string `json:"stderr"`
	Error    string `json:"error,omitempty"`
	ExitCode int    `json:"exit_code"`
}

// Stop stops the user's container.
func (m *Manager) Stop(ctx context.Context, userID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	name := m.containerName(userID)
	cmd := exec.CommandContext(ctx, "docker", "stop", name)
	cmd.Stderr = &bytes.Buffer{}
	return cmd.Run()
}

// Remove stops and removes the user's container.
func (m *Manager) Remove(ctx context.Context, userID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	name := m.containerName(userID)

	// Stop first (ignore errors if already stopped).
	_ = exec.CommandContext(ctx, "docker", "stop", name).Run()

	cmd := exec.CommandContext(ctx, "docker", "rm", name)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		if !strings.Contains(stderr.String(), "No such container") {
			return fmt.Errorf("failed to remove container: %s: %w", strings.TrimSpace(stderr.String()), err)
		}
	}
	return nil
}

// Status returns the current status of the user's container.
func (m *Manager) Status(ctx context.Context, userID string) (ContainerStatus, error) {
	name := m.containerName(userID)

	exists, err := m.containerExists(ctx, name)
	if err != nil {
		return ContainerStatus{}, err
	}
	if !exists {
		return ContainerStatus{Exists: false}, nil
	}

	running, err := m.containerRunning(ctx, name)
	if err != nil {
		return ContainerStatus{Exists: true}, err
	}

	return ContainerStatus{
		Exists:    true,
		Running:   running,
		Name:      name,
		Workspace: m.workspaceDir(userID),
	}, nil
}

// ContainerStatus holds the status of a user's container.
type ContainerStatus struct {
	Exists    bool   `json:"exists"`
	Running   bool   `json:"running"`
	Name      string `json:"name"`
	Workspace string `json:"workspace"`
}

// DockerAvailable checks if Docker is available on the system.
func DockerAvailable() bool {
	cmd := exec.Command("docker", "version", "--format", "{{.Server.Version}}")
	err := cmd.Run()
	return err == nil
}

func (m *Manager) containerExists(ctx context.Context, name string) (bool, error) {
	cmd := exec.CommandContext(ctx, "docker", "inspect", "--format", "{{.State.Status}}", name)
	err := cmd.Run()
	if err != nil {
		// Docker returns non-zero if container doesn't exist.
		return false, nil
	}
	return true, nil
}

func (m *Manager) containerRunning(ctx context.Context, name string) (bool, error) {
	cmd := exec.CommandContext(ctx, "docker", "inspect", "--format", "{{.State.Running}}", name)
	var stdout bytes.Buffer
	cmd.Stdout = &stdout
	if err := cmd.Run(); err != nil {
		return false, nil
	}
	return strings.TrimSpace(stdout.String()) == "true", nil
}

func (m *Manager) startContainer(ctx context.Context, name string) error {
	cmd := exec.CommandContext(ctx, "docker", "start", name)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("failed to start container: %s: %w", strings.TrimSpace(stderr.String()), err)
	}
	return nil
}
