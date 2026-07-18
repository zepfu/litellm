# UI Settings API

Proxy admins can read and update Admin UI configuration flags through the UI Settings endpoints. These flags control client-side UI behavior and a small set of server-side policy toggles stored in the `litellm_uisettings` table.

## Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/get/ui_settings` | Authenticated users | Return current values plus a JSON schema of allowlisted fields |
| `PATCH` | `/update/ui_settings` | Proxy admin only | Partially update allowlisted UI settings (`exclude_unset` merge) |

Requires a connected database. Updates also require `STORE_MODEL_IN_DB=True`.

### Get current settings

```bash
curl -X GET 'http://localhost:4000/get/ui_settings' \
  -H 'Authorization: Bearer <your-key>'
```

Response shape (abridged):

```json
{
  "values": {
    "disable_model_add_for_internal_users": false,
    "disable_custom_api_keys": false
  },
  "field_schema": {
    "properties": {
      "disable_custom_api_keys": {
        "type": "boolean",
        "description": "If true, users cannot specify custom API key values. All keys must be auto-generated."
      }
    }
  }
}
```

Unknown keys stored in the database are stripped before return. Only allowlisted fields appear in `values` and `field_schema`.

### Update settings

```bash
curl -X PATCH 'http://localhost:4000/update/ui_settings' \
  -H 'Authorization: Bearer <your-admin-key>' \
  -H 'Content-Type: application/json' \
  -d '{
    "disable_custom_api_keys": true
  }'
```

`PATCH` merges the request body with the existing record. Fields the caller does not send are preserved when they are allowlisted. Non-allowlisted fields are ignored on input and stripped from the merged record before persistence, so legacy unknown keys are not re-written.

## `disable_custom_api_keys`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `disable_custom_api_keys` | `boolean` | `false` | Exposes the persisted policy value consumed by key-management code that rejects caller-supplied key material. |

### Behavior

- **Settings surface (this page / these endpoints):** The flag is part of the `UISettings` model and the server allowlist. Admins can persist it; authenticated clients can read it via `GET /get/ui_settings` and the cached helper used by other proxy code paths.
- **Key-generation enforcement:** Enforcement against `data.key` / `data.new_key` on `/key/generate` and `/key/regenerate` is owned by key-management code (`_check_custom_key_allowed`), not by the UI settings endpoints. Enabling the toggle alone only exposes and stores the configuration value on the settings surface.

## Related allowlisted flags

Other allowlisted UI settings on the same endpoints include (non-exhaustive):

- `disable_model_add_for_internal_users`
- `enabled_ui_pages_internal_users` (see [page visibility](./page_visibility.md))
- `enable_projects_ui`
- `require_auth_for_public_ai_hub`
- `forward_client_headers_to_llm_api`
- `scope_user_search_to_org`
- agent / vector-store internal-user access toggles

Only fields present on the server allowlist are readable or writable through these endpoints.
