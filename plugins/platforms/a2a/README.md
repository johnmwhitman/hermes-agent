# A2A — Agent-to-Agent protocol for Hermes

Talk to other agents, and let other agents talk to you, over the open
[A2A protocol](https://a2a-protocol.org) **v1.0**. Works with any A2A-compliant
peer (another Hermes, LangChain, CrewAI, Google ADK, OpenClaw, …). Stdlib only —
no `a2a-sdk` dependency.

## Enable

```bash
hermes gateway setup      # pick A2A, or:
```

```yaml
# ~/.hermes/config.yaml
gateway:
  platforms:
    a2a:
      enabled: true
      extra:
        port: 9900

# peers you want to call (outbound):
a2a_agents:
  researcher:
    url: "http://localhost:9999"
    auth: { type: bearer, token: "sk-..." }
    timeout: 120
    capabilities: [web_search, research]
```

### Listener modes

Enabling the platform preserves the historical inbound-listener default. Use
`role: remote` when this profile should call peers but must not expose a local
HTTP endpoint:

```yaml
gateway:
  platforms:
    a2a:
      enabled: true
      role: remote
      port: 0                    # ignored because no listener is constructed
      extra:
        inbound_disabled: true  # exact YAML boolean; defense in depth
```

The five outbound tools remain registered in this mode. `role: inbound`,
`role: local`, or an omitted role starts the inbound listener. An exact
`inbound_disabled: true` disables inbound regardless of a valid role; `false`
leaves the role in control. Top-level `role`, `port`, and `inbound_disabled`
are accepted for operator-friendly YAML, while an explicitly nested `extra`
value takes precedence. Unknown roles and non-boolean `inbound_disabled`
values are rejected before listener construction. Bind host and port settings,
including `A2A_HOST` and `A2A_PORT`, are not parsed in listener-free mode.

Hermes accepts four historical YAML locations. Their precedence from lowest
to highest is `gateway.platforms.a2a`, `gateway.a2a`, `platforms.a2a`, then the
legacy direct root `a2a` block. Ordinary keys are replaced by the higher
source; `extra` is deep-merged so unrelated lower-source keys survive. Within
one effective block, an explicit `extra.role`, `extra.port`, or
`extra.inbound_disabled` wins over the corresponding shorthand key.

## Outbound — call other agents

The agent gets five tools:

- `a2a_discover(url)` — what can this agent do?
- `a2a_call(agent, message, context_id?)` — send it a task, get the reply.
- `a2a_list()` — configured peers, saved conversations, metrics.
- `a2a_history(context_id)` — recall a saved A2A conversation.
- `a2a_orchestrate(capability, message, mode?)` — fan-out a task to every
  peer advertising a capability (`all` / `first` / `best`).

## Inbound — be callable

When the `a2a` platform is enabled in its default, `inbound`, or `local` mode,
Hermes serves a v1.0 Agent Card at
`http://<host>:<port>/.well-known/agent-card.json` (the legacy
`/.well-known/agent.json` path is also answered for pre-1.0 clients) and
accepts JSON-RPC
`message/send`, `message/stream` (SSE), `tasks/get|list|cancel|subscribe`,
and push notification configs (inline or via
`tasks/pushNotificationConfig/create`). Incoming tasks are injected into your
**live** agent session — the same agent that's talking to you, with full
memory — and the reply is returned over A2A. Completed tasks stay queryable
via `tasks/get`.

## Security

- **No token ⇒ localhost only.** The server binds `127.0.0.1` and refuses to
  widen unless you configure a token *and* set `A2A_HOST`.
- **Per-peer tokens**: `A2A_PEER_TOKENS="alice:tok1,bob:tok2"` gives each
  remote agent its own credential; that authenticated name (never anything
  in the request body) drives rate limiting, trust, and audit.
- Inbound text — including `/`-prefixed text — is run through
  prompt-injection filters and framed as untrusted peer input; remote peers
  cannot invoke operator slash commands.
- Outbound text is scrubbed of credential-shaped strings.
- Push callbacks are SSRF-guarded and HMAC-SHA256 signed (`X-A2A-Signature`).
- Every exchange is logged to `~/.hermes/a2a_audit.jsonl`.
- Conversations persist to `~/.hermes/a2a_conversations/` — they survive context
  compaction and restarts (`a2a_history` recalls them).

### Inbound tool posture

Inbound A2A turns are read-only by default. The closed default surface is
`read_file`, `search_files`, `skills_list`, `skill_view`, `web_search`,
`web_extract`, `kanban_show`, `kanban_list`, `a2a_history`, and `a2a_list`.
A missing mutation marker means read-only; malformed or conflicting markers
are rejected.

Mutable access requires every gate below:

1. The request metadata contains the JSON boolean
   `"hermes.ai/mutationAllowed": true`.
2. The caller used an authenticated peer credential and is globally trusted.
3. The served route lists that identity in `mutation_allowed_peers`.
4. The route declares non-empty `mutable_toolsets`.

`execute_code` and `delegate_task` are not admitted by A2A, even when a route
names their toolsets. They create nested execution authorities; enabling them
requires a future design that propagates and revalidates the signed posture at
every sandbox RPC and delegated-child boundary.

```yaml
gateway:
  platforms:
    a2a:
      enabled: true
      extra:
        agents:
          builder:
            profile: builder
            allowed_peers: [conductor]
            mutation_allowed_peers: [conductor]
            mutable_toolsets: [terminal]
            advertised_toolsets: [web]
            # Publicly advertise mutable capability only when intentional.
            advertise_mutable_capability: false
```

`allowed_peers` narrows all access to a named route. An omitted list retains
the global trusted-peer policy. Mutation posture is bound to the authenticated
peer, served route, context, and final tool-schema fingerprint; a mismatch or
missing/corrupt resume binding fails closed or is quarantined read-only.

## Env vars

| Var | Default | Meaning |
|---|---|---|
| `A2A_PEER_TOKENS` | _(unset)_ | Per-peer credentials `name:token,…` (preferred). |
| `A2A_BEARER_TOKEN` | _(unset)_ | Shared token; identity falls back to caller IP. |
| `A2A_HOST` | `127.0.0.1` | Bind host. Only widens with a token set. |
| `A2A_PORT` | `9900` | Inbound port. |
| `A2A_AGENT_NAME` | hostname-derived | Name on the Agent Card. |
| `A2A_PUBLIC_URL` | _(unset)_ | Routable URL advertised on the card (reverse proxies). |
| `A2A_TRUSTED_PEERS` | _(unset)_ | Allow-list of authenticated identities. |
| `A2A_ALLOW_ALL_USERS` | `false` | Allow any authed peer (dev only). |
| `A2A_RATE_LIMIT` | `60` | Requests/minute per identity. |
| `A2A_MAX_PINGPONG_TURNS` | `5` | Anti-loop turn cap per context (max 20). |
| `A2A_REPLY_TIMEOUT` | `300` | Seconds to wait for the agent's reply. |
| `A2A_PUSH_SECRET` | bearer token | HMAC secret for push signing. |
| `A2A_ADVERTISED_TOOLSETS` | all registered | Restrict skills on the Agent Card. |

See `DESIGN.md` for architecture and the requirement-tracing table.
