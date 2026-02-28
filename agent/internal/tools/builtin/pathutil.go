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
	joined := filepath.Join(rootAbs, requested)
	cleanAbs, err := filepath.Abs(joined)
	if err != nil {
		return "", err
	}
	if cleanAbs != rootAbs && !strings.HasPrefix(cleanAbs, rootAbs+string(filepath.Separator)) {
		return "", fmt.Errorf("path escapes working directory")
	}
	return cleanAbs, nil
}
