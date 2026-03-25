package skills

import (
	"fmt"
	"strings"
)

type parsedSkill struct {
	Name        string
	Description string
	AlwaysLoad  bool
	Integration string
	Body        string
}

func parseSkillMarkdown(raw string) (parsedSkill, error) {
	lines := strings.Split(raw, "\n")
	if len(lines) == 0 || strings.TrimSpace(lines[0]) != "---" {
		return parsedSkill{}, fmt.Errorf("%w: missing frontmatter", ErrInvalidSkill)
	}

	end := -1
	for i := 1; i < len(lines); i++ {
		if strings.TrimSpace(lines[i]) == "---" {
			end = i
			break
		}
	}
	if end == -1 {
		return parsedSkill{}, fmt.Errorf("%w: unterminated frontmatter", ErrInvalidSkill)
	}

	meta := map[string]string{}
	for _, line := range lines[1:end] {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.SplitN(line, ":", 2)
		if len(parts) != 2 {
			continue
		}
		key := strings.TrimSpace(parts[0])
		val := strings.TrimSpace(parts[1])
		meta[key] = val
	}

	name := strings.TrimSpace(meta["name"])
	desc := strings.TrimSpace(meta["description"])
	if name == "" || desc == "" {
		return parsedSkill{}, fmt.Errorf("%w: name and description required", ErrInvalidSkill)
	}
	alwaysLoad := strings.EqualFold(strings.TrimSpace(meta["always_load"]), "true")
	integration := strings.TrimSpace(meta["integration"])

	body := ""
	if end+1 < len(lines) {
		body = strings.TrimSpace(strings.Join(lines[end+1:], "\n"))
	}

	return parsedSkill{Name: name, Description: desc, AlwaysLoad: alwaysLoad, Integration: integration, Body: body}, nil
}

func buildSkillMarkdown(name, description, body string) string {
	return fmt.Sprintf("---\nname: %s\ndescription: %s\n---\n\n%s\n", strings.TrimSpace(name), strings.TrimSpace(description), strings.TrimSpace(body))
}
