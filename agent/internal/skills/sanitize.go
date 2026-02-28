package skills

import "regexp"

var nonNameChars = regexp.MustCompile(`[^a-z0-9._]+`)
var edgeTrimChars = regexp.MustCompile(`^[.\-]+|[.\-]+$`)

func sanitizeName(name string) string {
	trimmed := nonNameChars.ReplaceAllString(lowerASCII(trimSpace(name)), "-")
	trimmed = edgeTrimChars.ReplaceAllString(trimmed, "")
	if len(trimmed) > 255 {
		trimmed = trimmed[:255]
	}
	if trimmed == "" {
		return "unnamed-skill"
	}
	return trimmed
}

func normalizeName(name string) string {
	return lowerASCII(trimSpace(name))
}

func trimSpace(s string) string {
	start, end := 0, len(s)
	for start < end && isSpace(s[start]) {
		start++
	}
	for end > start && isSpace(s[end-1]) {
		end--
	}
	return s[start:end]
}

func isSpace(b byte) bool {
	return b == ' ' || b == '\n' || b == '\r' || b == '\t'
}

func lowerASCII(s string) string {
	b := []byte(s)
	for i := range b {
		if b[i] >= 'A' && b[i] <= 'Z' {
			b[i] = b[i] + 32
		}
	}
	return string(b)
}
