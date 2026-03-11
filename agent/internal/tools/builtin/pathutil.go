package builtin

import (
	"fmt"
	"path/filepath"
	"strings"
)

func resolveSafePath(root, requested string) (string, error) {
	if strings.TrimSpace(root) == "" {
		return "", fmt.Errorf("working directory is not configured")
	}
	if strings.TrimSpace(requested) == "" {
		requested = "."
	}
	rootAbs, err := filepath.Abs(root)
	if err != nil {
		return "", err
	}

	// Resolve the root itself through symlinks for consistent comparison.
	// TempDir() on macOS returns /var/... which is a symlink to /private/var/...,
	// so both sides of every comparison must use the same resolved form.
	rootReal, err := filepath.EvalSymlinks(rootAbs)
	if err != nil {
		// Root doesn't exist yet — fall back to the absolute path
		rootReal = rootAbs
	}

	joined := filepath.Join(rootReal, requested)
	cleanAbs, err := filepath.Abs(joined)
	if err != nil {
		return "", err
	}

	// First check: logical path must be under the resolved root
	if cleanAbs != rootReal && !strings.HasPrefix(cleanAbs, rootReal+string(filepath.Separator)) {
		return "", fmt.Errorf("path escapes working directory")
	}

	// Second check: resolve symlinks in the final path and verify it's still
	// under the real root. This prevents symlink-based sandbox escapes.
	realPath, err := filepath.EvalSymlinks(cleanAbs)
	if err == nil {
		// Path exists — verify the resolved path is still under the real root
		if realPath != rootReal && !strings.HasPrefix(realPath, rootReal+string(filepath.Separator)) {
			return "", fmt.Errorf("path escapes working directory via symlink")
		}
		return realPath, nil
	}

	// Path doesn't exist yet (new file being created) — check the parent
	// directory to ensure it doesn't resolve outside the sandbox.
	parentDir := filepath.Dir(cleanAbs)
	realParent, err := filepath.EvalSymlinks(parentDir)
	if err == nil {
		if realParent != rootReal && !strings.HasPrefix(realParent, rootReal+string(filepath.Separator)) {
			return "", fmt.Errorf("path escapes working directory via symlink")
		}
	}
	// If parent doesn't exist either, the original prefix check is sufficient

	return cleanAbs, nil
}
