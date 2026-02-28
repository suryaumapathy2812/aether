package tools

import "context"

type TaskRuntimeContext struct {
	TaskID    string
	LockToken string
	UserID    string
}

type taskRuntimeContextKey struct{}

func WithTaskRuntimeContext(ctx context.Context, taskCtx TaskRuntimeContext) context.Context {
	return context.WithValue(ctx, taskRuntimeContextKey{}, taskCtx)
}

func TaskRuntimeContextFromContext(ctx context.Context) (TaskRuntimeContext, bool) {
	v := ctx.Value(taskRuntimeContextKey{})
	taskCtx, ok := v.(TaskRuntimeContext)
	return taskCtx, ok
}
