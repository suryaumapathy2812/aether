package plugins

import "errors"

var (
	ErrNotFound      = errors.New("plugin not found")
	ErrProtected     = errors.New("plugin is protected")
	ErrInvalidPlugin = errors.New("invalid plugin")
	ErrDuplicateName = errors.New("duplicate plugin name")
	ErrInvalidSource = errors.New("invalid source")
)
