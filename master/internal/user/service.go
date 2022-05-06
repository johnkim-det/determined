package user

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/labstack/echo/v4"
	"github.com/pkg/errors"

	"github.com/determined-ai/determined/master/internal/api"
	"github.com/determined-ai/determined/master/internal/context"
	"github.com/determined-ai/determined/master/internal/db"
	"github.com/determined-ai/determined/master/internal/telemetry"
	"github.com/determined-ai/determined/master/pkg/actor"
	"github.com/determined-ai/determined/master/pkg/model"
)

var once sync.Once
var userService *Service

type agentUserGroup struct {
	UID   *int   `json:"uid,omitempty"`
	GID   *int   `json:"gid,omitempty"`
	User  string `json:"user"`
	Group string `json:"group"`
}

func (h *agentUserGroup) Validate() (*model.AgentUserGroup, error) {
	switch {
	case h.UID == nil:
		return nil, errors.New("uid must be set")
	case h.GID == nil:
		return nil, errors.New("gid must be set")
	case len(h.User) == 0:
		return nil, errors.New("user must be set")
	case len(h.Group) == 0:
		return nil, errors.New("group must be set")
	}

	return &model.AgentUserGroup{
		UID:   *h.UID,
		GID:   *h.GID,
		User:  h.User,
		Group: h.Group,
	}, nil
}

// Service describes a user manager.
type Service struct {
	db        *db.PgDB
	system    *actor.System
	extConfig *model.ExternalSessions
}

// InitService creates the user service singleton.
func InitService(db *db.PgDB, system *actor.System, extConfig *model.ExternalSessions) {
	once.Do(func() {
		userService = &Service{db, system, extConfig}
	})
}

// GetService returns a reference to the user service singleton.
func GetService() *Service {
	if userService == nil {
		panic("Singleton UserService is not yet initialized.")
	}
	return userService
}

// The middleware looks for a token in two places (in this order):
// 1. The HTTP Authorization header.
// 2. A cookie named "auth".
func (s *Service) extractToken(r *http.Request) (string, error) {
	authRaw := r.Header.Get("Authorization")
	if authRaw != "" {
		// We attempt to parse out the token, which should be
		// transmitted as a Bearer authentication token.
		if !strings.HasPrefix(authRaw, "Bearer ") {
			return "", echo.ErrUnauthorized
		}
		return strings.TrimPrefix(authRaw, "Bearer "), nil
	} else if cookie, err := r.Cookie("auth"); err == nil {
		return cookie.Value, nil
	}
	// If we found no token, then abort the request with an HTTP 401.
	return "", echo.NewHTTPError(http.StatusUnauthorized)
}

// UserAndSessionFromRequest gets the user and session corresponding to the given request.
func (s *Service) UserAndSessionFromRequest(
	r *http.Request,
) (*model.User, *model.UserSession, error) {
	token, err := s.extractToken(r)
	if err != nil {
		return nil, nil, err
	}
	return s.db.UserByToken(token, s.extConfig)
}

// ProcessAuthentication is a middleware processing function that attempts
// to authenticate incoming HTTP requests.
func (s *Service) ProcessAuthentication(next echo.HandlerFunc) echo.HandlerFunc {
	return s.processAuthentication(next, false)
}

// ProcessAdminAuthentication is a middleware processing function that authenticates requests much
// like ProcessAuthentication but requires the user to be an admin.
func (s *Service) ProcessAdminAuthentication(next echo.HandlerFunc) echo.HandlerFunc {
	return s.processAuthentication(next, true)
}

func (s *Service) processAuthentication(next echo.HandlerFunc, adminOnly bool) echo.HandlerFunc {
	return func(c echo.Context) error {
		user, session, err := s.UserAndSessionFromRequest(c.Request())
		switch err {
		case nil:
			if !user.Active {
				return echo.NewHTTPError(http.StatusForbidden, "user not active")
			}
			if adminOnly && !user.Admin {
				return echo.NewHTTPError(http.StatusForbidden, "user not admin")
			}

			// Set data on the request context that might be useful to
			// event handlers.
			c.(*context.DetContext).SetUser(*user)
			c.(*context.DetContext).SetUserSession(*session)
			return next(c)
		case db.ErrNotFound:
			return echo.NewHTTPError(http.StatusUnauthorized)
		default:
			return err
		}
	}
}

// ProcessProxyAuthentication is a middleware processing function that attempts
// to authenticate incoming HTTP requests coming through proxies.
func (s *Service) ProcessProxyAuthentication(c echo.Context) (done bool, err error) {
	token, err := s.extractToken(c.Request())
	if err != nil {
		return true, redirectToLogin(c)
	}

	switch user, _, err := s.db.UserByToken(token, s.extConfig); err {
	case nil:
		if !user.Active {
			return true, redirectToLogin(c)
		}
		return false, nil
	case db.ErrNotFound:
		return true, redirectToLogin(c)
	default:
		return true, err
	}
}

func redirectToLogin(c echo.Context) error {
	return c.Redirect(
		http.StatusSeeOther,
		fmt.Sprintf("/det/login?redirect=%s", c.Request().URL),
	)
}

func (s *Service) postLogout(c echo.Context) (interface{}, error) {
	// Delete the cookie if one is set.
	if cookie, err := c.Cookie("auth"); err == nil {
		cookie.Value = ""
		cookie.Expires = time.Unix(0, 0)
		c.SetCookie(cookie)
	}

	// Delete the user session information from the database.
	sess := c.(*context.DetContext).MustGetUserSession()

	if err := s.db.DeleteUserSessionByID(sess.ID); err != nil {
		return nil, err
	}

	return "", nil
}

func (s *Service) postLogin(c echo.Context) (interface{}, error) {
	if s.extConfig.JwtKey != "" {
		return nil, echo.NewHTTPError(http.StatusMisdirectedRequest,
			"authentication is configured to be external")
	}

	type (
		request struct {
			Username string `json:"username"`
			Password string `json:"password"`
		}
		response struct {
			Token  string       `json:"token"`
			UserId model.UserID `json:"userId"`
		}
	)

	body, err := ioutil.ReadAll(c.Request().Body)
	if err != nil {
		return nil, err
	}

	var params request
	if err = json.Unmarshal(body, &params); err != nil {
		return nil, echo.NewHTTPError(http.StatusBadRequest)
	}

	// Get the user from the database.
	user, err := s.db.UserByUsername(params.Username)
	switch err {
	case nil:
	case db.ErrNotFound:
		return nil, echo.NewHTTPError(http.StatusForbidden, "user not found")
	default:
		return nil, err
	}

	// The user must be active.
	if !user.Active {
		return nil, echo.NewHTTPError(http.StatusForbidden, "user not active")
	}

	var token string
	if !user.ValidatePassword(params.Password) {
		return nil, echo.NewHTTPError(http.StatusForbidden, "invalid credentials")
	}

	token, err = s.db.StartUserSession(user)
	if err != nil {
		return nil, err
	}

	// The caller of this REST endpoint can request that the master set a cookie.
	// This is used by the WebUI for persistence of sessions.
	if c.QueryParam("cookie") == "true" {
		c.SetCookie(NewCookieFromToken(token))
	}

	return response{
		Token:  token,
		UserId: user.ID,
	}, nil
}

// NewCookieFromToken creates a new cookie from the given token.
func NewCookieFromToken(token string) *http.Cookie {
	cookie := new(http.Cookie)
	cookie.Name = "auth"
	cookie.Value = token
	cookie.Path = "/"
	cookie.Expires = time.Now().Add(db.SessionDuration)
	return cookie
}

// getMe returns information about the current authenticated user.
func (s *Service) getMe(c echo.Context) (interface{}, error) {
	me := c.(*context.DetContext).MustGetUser()

	return s.db.UserByID(me.ID)
}

func (s *Service) getUsers(c echo.Context) (interface{}, error) {
	return s.db.UserList()
}

func (s *Service) patchUser(c echo.Context) (interface{}, error) {
	type (
		request struct {
			Password *string `json:"password,omitempty"`
			Active   *bool   `json:"active,omitempty"`
			Admin    *bool   `json:"admin,omitempty"`

			AgentUserGroup *agentUserGroup `json:"agent_user_group,omitempty"`
		}
		response struct {
			message string
		}
	)

	body, err := ioutil.ReadAll(c.Request().Body)
	if err != nil {
		return nil, err
	}

	args := struct {
		Username string `path:"username"`
	}{}
	if err = api.BindArgs(&args, c); err != nil {
		return nil, err
	}

	var params request
	if err = json.Unmarshal(body, &params); err != nil {
		malformedRequestError := echo.NewHTTPError(http.StatusBadRequest, "bad request")
		return nil, malformedRequestError
	}

	forbiddenError := echo.NewHTTPError(
		http.StatusForbidden,
		"user not authorized")

	authenticatedUser := c.(*context.DetContext).MustGetUser()
	user, err := s.db.UserByUsername(args.Username)
	switch err {
	case nil:
	case db.ErrNotFound:
		if authenticatedUser.Admin {
			return nil, echo.NewHTTPError(
				http.StatusBadRequest,
				fmt.Sprintf("failed to get user '%s'", args.Username))
		}
		return nil, echo.NewHTTPError(http.StatusForbidden, "user not found")
	default:
		return nil, err
	}

	var toUpdate []string

	if params.Password != nil {
		if !user.PasswordCanBeModifiedBy(authenticatedUser) {
			return nil, forbiddenError
		}
		if err = user.UpdatePasswordHash(*params.Password); err != nil {
			return nil, err
		}
		toUpdate = append(toUpdate, "password_hash")
	}

	if params.Active != nil {
		if !user.ActiveCanBeModifiedBy(authenticatedUser) {
			return nil, forbiddenError
		}
		user.Active = *params.Active
		toUpdate = append(toUpdate, "active")
	}

	if params.Admin != nil {
		if !user.AdminCanBeModifiedBy(authenticatedUser) {
			return nil, forbiddenError
		}
		user.Admin = *params.Admin
		toUpdate = append(toUpdate, "admin")
	}

	var ug *model.AgentUserGroup
	if pug := params.AgentUserGroup; pug != nil {
		if !user.AdminCanBeModifiedBy(authenticatedUser) {
			return nil, forbiddenError
		}

		u, pErr := pug.Validate()
		if pErr != nil {
			return nil, echo.NewHTTPError(http.StatusBadRequest, pErr.Error())
		}
		ug = u
	}

	if err = s.db.UpdateUser(user, toUpdate, ug); err != nil {
		return nil, err
	}

	return response{
		message: fmt.Sprintf("successfully updated %v", args.Username),
	}, nil
}

func (s *Service) patchUsername(c echo.Context) (interface{}, error) {
	type (
		request struct {
			NewUsername *string `json:"username,omitempty"`
		}
		response struct {
			message string
		}
	)

	body, err := ioutil.ReadAll(c.Request().Body)
	if err != nil {
		return nil, err
	}

	args := struct {
		Username string `path:"username"`
	}{}
	if err = api.BindArgs(&args, c); err != nil {
		return nil, err
	}

	var params request
	if err = json.Unmarshal(body, &params); err != nil {
		malformedRequestError := echo.NewHTTPError(http.StatusBadRequest, "bad request")
		return nil, malformedRequestError
	}

	forbiddenError := echo.NewHTTPError(http.StatusForbidden,
		"user not authorized")
	authenticatedUser := c.(*context.DetContext).MustGetUser()
	if !authenticatedUser.Admin {
		return nil, forbiddenError
	}

	user, err := s.db.UserByUsername(args.Username)
	if err != nil {
		return nil, err
	}

	if params.NewUsername == nil {
		malformedRequestError := echo.NewHTTPError(http.StatusBadRequest, "username is required")
		return nil, malformedRequestError
	}

	switch u, uErr := s.db.UserByUsername(*params.NewUsername); {
	case uErr == db.ErrNotFound:
	case uErr != nil:
		return nil, uErr
	case u != nil:
		return nil, echo.NewHTTPError(http.StatusBadRequest, "username is taken")
	}

	if err = s.db.UpdateUsername(&user.ID, *params.NewUsername); err != nil {
		return nil, err
	}

	return response{
		message: fmt.Sprintf("successfully updated %v", args.Username),
	}, nil
}

func (s *Service) postUser(c echo.Context) (interface{}, error) {
	type (
		request struct {
			Username string `json:"username"`
			Admin    bool   `json:"admin"`
			Active   bool   `json:"active"`

			AgentUserGroup *agentUserGroup `json:"agent_user_group,omitempty"`
		}
		response struct {
			message string
		}
	)

	body, err := ioutil.ReadAll(c.Request().Body)
	if err != nil {
		return nil, err
	}

	var params request
	if err = json.Unmarshal(body, &params); err != nil {
		malformedRequestError := echo.NewHTTPError(http.StatusBadRequest, "bad request")
		return nil, malformedRequestError
	}

	currUser := c.(*context.DetContext).MustGetUser()
	if !currUser.CanCreateUser() {
		insufficientPermissionsError := echo.NewHTTPError(
			http.StatusForbidden,
			"insufficient permissions")
		return nil, insufficientPermissionsError
	}

	var ug *model.AgentUserGroup
	if pug := params.AgentUserGroup; pug != nil {
		u, pErr := pug.Validate()
		if pErr != nil {
			return nil, echo.NewHTTPError(http.StatusBadRequest, pErr.Error())
		}
		ug = u
	}

	params.Username = strings.ToLower(params.Username)
	_, err = s.db.AddUser(&model.User{
		Username: params.Username,
		Admin:    params.Admin,
		Active:   params.Active,
	}, ug)

	switch {
	case err == db.ErrDuplicateRecord:
		return nil, echo.NewHTTPError(http.StatusBadRequest, "user already exists")
	case err != nil:
		return nil, err
	}

	telemetry.ReportUserCreated(s.system, params.Admin, params.Active)

	return response{
		message: fmt.Sprintf("successfully created user: %s", params.Username),
	}, nil
}

func (s *Service) getUserImage(c echo.Context) (interface{}, error) {
	args := struct {
		Username string `path:"username"`
	}{}
	if err := api.BindArgs(&args, c); err != nil {
		return nil, err
	}
	c.Response().Header().Set("cache-control", "public, max-age=3600")

	return s.db.UserImage(args.Username)
}
