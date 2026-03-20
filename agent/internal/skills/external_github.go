package skills

import (
	"fmt"
	"regexp"
	"strings"
)

var sourceWithSkill = regexp.MustCompile(`^([^/\s]+)/([^/@\s]+)@([^\s]+)$`)
var sourceRepoOnly = regexp.MustCompile(`^([^/\s]+)/([^/\s]+)$`)

func parseExternalSource(source string) (owner, repo, skillName string, err error) {
	source = strings.TrimSpace(source)
	if source == "" {
		return "", "", "", fmt.Errorf("%w: empty source", ErrInvalidSource)
	}

	if matches := sourceWithSkill.FindStringSubmatch(source); len(matches) == 4 {
		return matches[1], matches[2], matches[3], nil
	}

	if strings.Contains(source, ":") || strings.HasPrefix(source, ".") {
		return "", "", "", fmt.Errorf("%w: %s", ErrInvalidSource, source)
	}

	if matches := sourceRepoOnly.FindStringSubmatch(source); len(matches) == 3 {
		return matches[1], matches[2], "", nil
	}

	return "", "", "", fmt.Errorf("%w: %s", ErrInvalidSource, source)
}

func buildRawURL(baseURL, owner, repo, skillName string) string {
	base := strings.TrimRight(baseURL, "/") + "/" + owner + "/" + repo + "/main"
	if strings.TrimSpace(skillName) == "" {
		return base + "/SKILL.md"
	}
	return base + "/" + skillName + "/SKILL.md"
}

func buildRawCandidates(baseURL, owner, repo, skillName string) []string {
	base := strings.TrimRight(baseURL, "/") + "/" + owner + "/" + repo + "/main"
	if strings.TrimSpace(skillName) == "" {
		return []string{base + "/SKILL.md"}
	}
	name := strings.TrimSpace(skillName)
	return []string{
		base + "/" + name + "/SKILL.md",
		base + "/skills/" + name + "/SKILL.md",
		base + "/.agents/skills/" + name + "/SKILL.md",
		base + "/.claude/skills/" + name + "/SKILL.md",
		base + "/.opencode/skills/" + name + "/SKILL.md",
	}
}
