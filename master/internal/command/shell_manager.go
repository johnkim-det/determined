// Package command provides utilities for commands. This package comment is to satisfy linters
// without disabling golint for the file.
//nolint:dupl // So easy with generics, so hard without; just wait.
package command

import (
	"net/http"

	"github.com/labstack/echo/v4"

	"github.com/determined-ai/determined/master/pkg/model"

	"github.com/determined-ai/determined/master/internal/db"
	"github.com/determined-ai/determined/master/internal/task"
	"github.com/determined-ai/determined/master/pkg/actor"
	"github.com/determined-ai/determined/master/pkg/tasks"
	"github.com/determined-ai/determined/proto/pkg/apiv1"
	"github.com/determined-ai/determined/proto/pkg/shellv1"
)

type shellManager struct {
	db         *db.PgDB
	taskLogger *task.Logger
}

func (s *shellManager) Receive(ctx *actor.Context) error {
	switch msg := ctx.Message().(type) {
	case actor.PreStart, actor.PostStop, actor.ChildFailed, actor.ChildStopped:

	case *apiv1.GetShellsRequest:
		resp := &apiv1.GetShellsResponse{}
		users := make(map[int32]bool)
		for _, user := range msg.UserIds {
			users[user] = true
		}
		for _, shell := range ctx.AskAll(&shellv1.Shell{}, ctx.Children()...).GetAll() {
			if typed := shell.(*shellv1.Shell); len(users) == 0 || users[typed.UserId] {
				resp.Shells = append(resp.Shells, typed)
			}
		}
		ctx.Respond(resp)

	case tasks.GenericCommandSpec:
		taskID := model.NewTaskID()
		jobID := model.NewJobID()
		if err := createGenericCommandActor(
			ctx, s.db, s.taskLogger, taskID, model.TaskTypeShell, jobID, model.JobTypeShell, msg,
		); err != nil {
			ctx.Log().WithError(err).Error("failed to launch shell")
			ctx.Respond(err)
		} else {
			ctx.Respond(taskID)
		}

	case echo.Context:
		ctx.Respond(echo.NewHTTPError(http.StatusNotFound, ErrAPIRemoved))

	default:
		return actor.ErrUnexpectedMessage(ctx)
	}
	return nil
}
