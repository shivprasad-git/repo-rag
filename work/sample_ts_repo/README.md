# Sample TypeScript Service

A small demo service used by the evaluation golden set.

## Authentication

Authentication uses JWT tokens. The AuthService issues login tokens and validates them.

## Roles

Users carry one of the Admin, Editor, or Viewer roles.

## Session

Sessions are created by issuing a signed JWT token with an expiry time.