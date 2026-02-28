package skills

import "errors"

var (
	ErrNotFound      = errors.New("skill not found")
	ErrProtected     = errors.New("skill is protected")
	ErrInvalidSkill  = errors.New("invalid skill")
	ErrDuplicateName = errors.New("duplicate skill name")
	ErrInvalidSource = errors.New("invalid source")
)
