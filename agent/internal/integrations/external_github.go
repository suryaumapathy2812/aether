package integrations

import (
	"fmt"
	"regexp"
	"strings"
)

var sourceWithPath = regexp.MustCompile(`^([^/\s]+)/([^/@\s]+)@([^\s]+)$`)
var sourceRepoOnly = regexp.MustCompile(`^([^/\s]+)/([^/\s]+)$`)

func parseExternalSource(source string) (owner, repo, pluginPath string, err error) {
	source = strings.TrimSpace(source)
	if source == "" {
		return "", "", "", fmt.Errorf("%w: empty source", ErrInvalidSource)
	}

	if matches := sourceWithPath.FindStringSubmatch(source); len(matches) == 4 {
		return matches[1], matches[2], strings.Trim(matches[3], "/"), nil
	}

	if strings.Contains(source, ":") || strings.HasPrefix(source, ".") {
		return "", "", "", fmt.Errorf("%w: %s", ErrInvalidSource, source)
	}

	if matches := sourceRepoOnly.FindStringSubmatch(source); len(matches) == 3 {
		return matches[1], matches[2], "", nil
	}

	return "", "", "", fmt.Errorf("%w: %s", ErrInvalidSource, source)
}

func buildRawManifestURL(baseURL, owner, repo, pluginPath string) string {
	base := strings.TrimRight(baseURL, "/") + "/" + owner + "/" + repo + "/main"
	if strings.TrimSpace(pluginPath) == "" {
		return base + "/integration.yaml"
	}
	return base + "/" + strings.Trim(pluginPath, "/") + "/integration.yaml"
}

func buildRawSkillURL(baseURL, owner, repo, pluginPath string) string {
	base := strings.TrimRight(baseURL, "/") + "/" + owner + "/" + repo + "/main"
	if strings.TrimSpace(pluginPath) == "" {
		return base + "/SKILL.md"
	}
	return base + "/" + strings.Trim(pluginPath, "/") + "/SKILL.md"
}
