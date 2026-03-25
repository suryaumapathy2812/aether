package integrations

import "errors"

var (
	ErrNotFound      = errors.New("integration not found")
	ErrProtected     = errors.New("integration is protected")
	ErrInvalidPlugin = errors.New("invalid integration")
	ErrDuplicateName = errors.New("duplicate integration name")
	ErrInvalidSource = errors.New("invalid source")
)
