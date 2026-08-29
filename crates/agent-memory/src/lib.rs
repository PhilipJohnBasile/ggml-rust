#![forbid(unsafe_code)]

use std::{
    collections::BTreeMap,
    fmt,
    time::{SystemTime, UNIX_EPOCH},
};

use sha2::{Digest, Sha256};

/// A memory namespace with explicit ownership semantics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryScope {
    /// Organizational knowledge. Agents may read it, but only an orchestrator
    /// may update it.
    Shared,
    /// Working memory owned by one agent identifier.
    Agent(String),
}

/// The authority used for memory reads, writes, and dreaming commits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Principal {
    id: String,
    role: PrincipalRole,
}

/// The three deliberately small roles needed by the memory policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrincipalRole {
    Reader,
    Agent,
    Orchestrator,
}

impl Principal {
    /// Creates a read-only principal.
    ///
    /// # Errors
    ///
    /// Returns an error when the identifier is empty or contains a NUL byte.
    pub fn reader(id: impl Into<String>) -> Result<Self, MemoryError> {
        Self::new(id, PrincipalRole::Reader)
    }

    /// Creates an agent principal. It can write only its own agent scope.
    ///
    /// # Errors
    ///
    /// Returns an error when the identifier is empty or contains a NUL byte.
    pub fn agent(id: impl Into<String>) -> Result<Self, MemoryError> {
        Self::new(id, PrincipalRole::Agent)
    }

    /// Creates an orchestrator principal. It can coordinate shared memory and
    /// dreaming commits.
    ///
    /// # Errors
    ///
    /// Returns an error when the identifier is empty or contains a NUL byte.
    pub fn orchestrator(id: impl Into<String>) -> Result<Self, MemoryError> {
        Self::new(id, PrincipalRole::Orchestrator)
    }

    /// Returns the stable principal identifier.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns the authority role.
    #[must_use]
    pub const fn role(&self) -> PrincipalRole {
        self.role
    }

    fn new(id: impl Into<String>, role: PrincipalRole) -> Result<Self, MemoryError> {
        let id = id.into();
        if id.is_empty() || id.contains('\0') {
            return Err(MemoryError::InvalidPrincipal);
        }
        Ok(Self { id, role })
    }
}

/// One requested content-addressed memory write.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryWrite {
    path: String,
    scope: MemoryScope,
    content: String,
    session_id: String,
    expected_hash: Option<String>,
}

impl MemoryWrite {
    /// Creates a write request. A request without an expected hash may create
    /// a new path, but cannot overwrite an existing head.
    ///
    /// # Errors
    ///
    /// Returns an error when the path, scope, or session identifier is invalid.
    pub fn new(
        scope: MemoryScope,
        path: impl Into<String>,
        content: impl Into<String>,
        session_id: impl Into<String>,
    ) -> Result<Self, MemoryError> {
        let path = validate_path(path.into())?;
        let session_id = session_id.into();
        if session_id.is_empty() || session_id.contains('\0') {
            return Err(MemoryError::InvalidSession);
        }
        validate_scope(&scope)?;
        Ok(Self {
            path,
            scope,
            content: content.into(),
            session_id,
            expected_hash: None,
        })
    }

    /// Sets the content hash that must still be the current head.
    #[must_use]
    pub fn with_expected_hash(mut self, expected_hash: impl Into<String>) -> Self {
        self.expected_hash = Some(expected_hash.into());
        self
    }

    /// Returns the target path.
    #[must_use]
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Returns the target scope.
    #[must_use]
    pub const fn scope(&self) -> &MemoryScope {
        &self.scope
    }
}

/// An immutable version in a memory file's history.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryEntry {
    path: String,
    scope: MemoryScope,
    version: u64,
    content: String,
    content_hash: String,
    written_by: String,
    session_id: String,
    sequence: u64,
    written_at_unix_ms: u128,
}

impl MemoryEntry {
    /// Returns the memory path.
    #[must_use]
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Returns the memory scope.
    #[must_use]
    pub const fn scope(&self) -> &MemoryScope {
        &self.scope
    }

    /// Returns this path's monotonically increasing version.
    #[must_use]
    pub const fn version(&self) -> u64 {
        self.version
    }

    /// Returns the stored content.
    #[must_use]
    pub fn content(&self) -> &str {
        &self.content
    }

    /// Returns the SHA-256 content hash used for compare-and-swap writes.
    #[must_use]
    pub fn content_hash(&self) -> &str {
        &self.content_hash
    }

    /// Returns the principal that wrote this version.
    #[must_use]
    pub fn written_by(&self) -> &str {
        &self.written_by
    }

    /// Returns the originating session identifier.
    #[must_use]
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Returns the store-wide logical sequence number.
    #[must_use]
    pub const fn sequence(&self) -> u64 {
        self.sequence
    }

    /// Returns the wall-clock attribution time in Unix milliseconds.
    #[must_use]
    pub const fn written_at_unix_ms(&self) -> u128 {
        self.written_at_unix_ms
    }
}

/// Failures returned by memory authorization, validation, and CAS operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryError {
    InvalidPrincipal,
    InvalidSession,
    InvalidPath,
    InvalidScope,
    Missing {
        path: String,
    },
    PermissionDenied {
        path: String,
    },
    PreconditionRequired {
        path: String,
        actual_hash: String,
    },
    Conflict {
        path: String,
        expected_hash: String,
        actual_hash: String,
    },
    ScopeMismatch {
        path: String,
    },
    InvalidVersion {
        path: String,
        version: u64,
    },
}

impl fmt::Display for MemoryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPrincipal => formatter.write_str("principal identifier is invalid"),
            Self::InvalidSession => formatter.write_str("session identifier is invalid"),
            Self::InvalidPath => formatter.write_str("memory path is invalid"),
            Self::InvalidScope => formatter.write_str("memory scope is invalid"),
            Self::Missing { path } => write!(formatter, "memory path {path} does not exist"),
            Self::PermissionDenied { path } => {
                write!(
                    formatter,
                    "principal is not authorized for memory path {path}"
                )
            }
            Self::PreconditionRequired { path, actual_hash } => write!(
                formatter,
                "memory path {path} requires a compare-and-swap precondition; current hash is {actual_hash}"
            ),
            Self::Conflict {
                path,
                expected_hash,
                actual_hash,
            } => write!(
                formatter,
                "memory path {path} changed: expected {expected_hash}, found {actual_hash}"
            ),
            Self::ScopeMismatch { path } => {
                write!(formatter, "memory path {path} cannot change scope")
            }
            Self::InvalidVersion { path, version } => {
                write!(formatter, "memory path {path} has no version {version}")
            }
        }
    }
}

impl std::error::Error for MemoryError {}

/// A versioned in-memory store. It is intentionally file-shaped: callers can
/// persist each entry as a standalone record without changing the API.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MemoryStore {
    histories: BTreeMap<String, Vec<MemoryEntry>>,
    next_sequence: u64,
}

impl MemoryStore {
    /// Creates an empty store.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            histories: BTreeMap::new(),
            next_sequence: 0,
        }
    }

    /// Writes a new version after checking permissions and the expected head.
    ///
    /// # Errors
    ///
    /// Returns an error when authorization fails or the compare-and-swap
    /// precondition does not match the current head.
    pub fn write(
        &mut self,
        principal: &Principal,
        request: MemoryWrite,
    ) -> Result<MemoryEntry, MemoryError> {
        authorize_write(principal, &request.scope, &request.path)?;
        let history = self.histories.entry(request.path.clone()).or_default();
        let current = history.last();
        if current.is_some_and(|entry| entry.scope != request.scope) {
            return Err(MemoryError::ScopeMismatch { path: request.path });
        }
        match (current, request.expected_hash.as_deref()) {
            (Some(entry), None) => {
                return Err(MemoryError::PreconditionRequired {
                    path: request.path,
                    actual_hash: entry.content_hash.clone(),
                });
            }
            (Some(entry), Some(expected_hash)) if expected_hash != entry.content_hash => {
                return Err(MemoryError::Conflict {
                    path: request.path,
                    expected_hash: expected_hash.to_owned(),
                    actual_hash: entry.content_hash.clone(),
                });
            }
            (None, Some(expected_hash)) if !expected_hash.is_empty() => {
                return Err(MemoryError::Conflict {
                    path: request.path,
                    expected_hash: expected_hash.to_owned(),
                    actual_hash: String::new(),
                });
            }
            _ => {}
        }
        self.next_sequence = self.next_sequence.saturating_add(1);
        let entry = MemoryEntry {
            path: request.path.clone(),
            scope: request.scope,
            version: current.map_or(1, |entry| entry.version + 1),
            content_hash: content_hash(&request.content),
            content: request.content,
            written_by: principal.id.clone(),
            session_id: request.session_id,
            sequence: self.next_sequence,
            written_at_unix_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |duration| duration.as_millis()),
        };
        history.push(entry.clone());
        Ok(entry)
    }

    /// Reads the current head when the principal is authorized for its scope.
    ///
    /// # Errors
    ///
    /// Returns an error when the path is missing, invalid, or unauthorized.
    pub fn read(&self, principal: &Principal, path: &str) -> Result<MemoryEntry, MemoryError> {
        let entry = self.head(path)?;
        authorize_read(principal, &entry.scope, path)?;
        Ok(entry.clone())
    }

    /// Reads a specific historical version.
    ///
    /// # Errors
    ///
    /// Returns an error when the path or version is missing or unauthorized.
    pub fn read_version(
        &self,
        principal: &Principal,
        path: &str,
        version: u64,
    ) -> Result<MemoryEntry, MemoryError> {
        let history = self
            .histories
            .get(path)
            .ok_or_else(|| MemoryError::Missing {
                path: path.to_owned(),
            })?;
        let entry = history
            .iter()
            .find(|entry| entry.version == version)
            .ok_or_else(|| MemoryError::InvalidVersion {
                path: path.to_owned(),
                version,
            })?;
        authorize_read(principal, &entry.scope, path)?;
        Ok(entry.clone())
    }

    /// Returns the full immutable history for an authorized path.
    ///
    /// # Errors
    ///
    /// Returns an error when the path is missing or unauthorized.
    pub fn history(
        &self,
        principal: &Principal,
        path: &str,
    ) -> Result<Vec<MemoryEntry>, MemoryError> {
        let history = self
            .histories
            .get(path)
            .ok_or_else(|| MemoryError::Missing {
                path: path.to_owned(),
            })?;
        if let Some(entry) = history.last() {
            authorize_read(principal, &entry.scope, path)?;
        }
        Ok(history.clone())
    }

    /// Returns a deterministic hash of all current heads for CAS commits.
    #[must_use]
    pub fn state_hash(&self) -> String {
        let mut hasher = Sha256::new();
        for (path, history) in &self.histories {
            if let Some(entry) = history.last() {
                hasher.update((path.len() as u64).to_le_bytes());
                hasher.update(path.as_bytes());
                hasher.update(entry.content_hash.as_bytes());
                hasher.update(entry.version.to_le_bytes());
            }
        }
        format!("sha256:{:x}", hasher.finalize())
    }

    /// Starts an out-of-band dreaming pass from a stable clone of this store.
    ///
    /// # Errors
    ///
    /// Returns an error when the caller is not an orchestrator.
    pub fn start_dreaming(&self, principal: &Principal) -> Result<DreamingPass, MemoryError> {
        if principal.role != PrincipalRole::Orchestrator {
            return Err(MemoryError::PermissionDenied {
                path: "<dreaming>".to_owned(),
            });
        }
        Ok(DreamingPass {
            input: self.clone(),
            output: self.clone(),
            base_hash: self.state_hash(),
        })
    }

    fn head(&self, path: &str) -> Result<&MemoryEntry, MemoryError> {
        validate_path(path.to_owned())?;
        self.histories
            .get(path)
            .and_then(|history| history.last())
            .ok_or_else(|| MemoryError::Missing {
                path: path.to_owned(),
            })
    }
}

/// A stable input clone and mutable output clone for a batch memory pass.
#[derive(Debug, Clone)]
pub struct DreamingPass {
    input: MemoryStore,
    output: MemoryStore,
    base_hash: String,
}

impl DreamingPass {
    /// Returns the frozen input state seen by each curator.
    #[must_use]
    pub const fn input(&self) -> &MemoryStore {
        &self.input
    }

    /// Returns the evolving output state.
    #[must_use]
    pub const fn output(&self) -> &MemoryStore {
        &self.output
    }

    /// Runs one transcript curator and applies its proposed writes to output.
    /// The curator reads only the frozen input state, which keeps sessions
    /// independent while the orchestrator serializes their commits.
    ///
    /// # Errors
    ///
    /// Returns an error from the curator or one of its proposed writes.
    pub fn curate_session<F>(
        &mut self,
        principal: &Principal,
        transcript: &SessionTranscript,
        curator: F,
    ) -> Result<usize, MemoryError>
    where
        F: FnOnce(&MemoryStore, &SessionTranscript) -> Result<Vec<MemoryWrite>, MemoryError>,
    {
        let writes = curator(&self.input, transcript)?;
        let mut applied = 0;
        for write in writes {
            self.output.write(principal, write)?;
            applied += 1;
        }
        Ok(applied)
    }

    /// Atomically publishes the output clone if the live store is unchanged.
    ///
    /// # Errors
    ///
    /// Returns an error when the caller is not an orchestrator or the live
    /// store changed since this pass began.
    pub fn commit(
        self,
        principal: &Principal,
        target: &mut MemoryStore,
    ) -> Result<(), MemoryError> {
        if principal.role != PrincipalRole::Orchestrator {
            return Err(MemoryError::PermissionDenied {
                path: "<dreaming>".to_owned(),
            });
        }
        let actual_hash = target.state_hash();
        if actual_hash != self.base_hash {
            return Err(MemoryError::Conflict {
                path: "<store>".to_owned(),
                expected_hash: self.base_hash,
                actual_hash,
            });
        }
        *target = self.output;
        Ok(())
    }
}

/// One agent session transcript supplied to a dreaming curator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionTranscript {
    agent_id: String,
    session_id: String,
    content: String,
}

impl SessionTranscript {
    /// Creates a transcript with explicit agent and session attribution.
    ///
    /// # Errors
    ///
    /// Returns an error when either identifier is empty or contains a NUL byte.
    pub fn new(
        agent_id: impl Into<String>,
        session_id: impl Into<String>,
        content: impl Into<String>,
    ) -> Result<Self, MemoryError> {
        let agent_id = agent_id.into();
        let session_id = session_id.into();
        if agent_id.is_empty() || agent_id.contains('\0') {
            return Err(MemoryError::InvalidPrincipal);
        }
        if session_id.is_empty() || session_id.contains('\0') {
            return Err(MemoryError::InvalidSession);
        }
        Ok(Self {
            agent_id,
            session_id,
            content: content.into(),
        })
    }

    /// Returns the originating agent identifier.
    #[must_use]
    pub fn agent_id(&self) -> &str {
        &self.agent_id
    }

    /// Returns the originating session identifier.
    #[must_use]
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Returns the raw transcript content.
    #[must_use]
    pub fn content(&self) -> &str {
        &self.content
    }
}

fn validate_path(path: String) -> Result<String, MemoryError> {
    if path.is_empty()
        || path.contains('\0')
        || path.starts_with('/')
        || path
            .split('/')
            .any(|segment| segment.is_empty() || segment == "." || segment == "..")
    {
        return Err(MemoryError::InvalidPath);
    }
    Ok(path)
}

fn validate_scope(scope: &MemoryScope) -> Result<(), MemoryError> {
    if let MemoryScope::Agent(agent_id) = scope
        && (agent_id.is_empty() || agent_id.contains('\0'))
    {
        return Err(MemoryError::InvalidScope);
    }
    Ok(())
}

fn authorize_read(
    principal: &Principal,
    scope: &MemoryScope,
    path: &str,
) -> Result<(), MemoryError> {
    let allowed = match scope {
        MemoryScope::Shared => true,
        MemoryScope::Agent(owner) => {
            principal.role == PrincipalRole::Orchestrator
                || (principal.role == PrincipalRole::Agent && principal.id == *owner)
        }
    };
    if allowed {
        Ok(())
    } else {
        Err(MemoryError::PermissionDenied {
            path: path.to_owned(),
        })
    }
}

fn authorize_write(
    principal: &Principal,
    scope: &MemoryScope,
    path: &str,
) -> Result<(), MemoryError> {
    let allowed = match scope {
        MemoryScope::Shared => principal.role == PrincipalRole::Orchestrator,
        MemoryScope::Agent(owner) => {
            principal.role == PrincipalRole::Orchestrator
                || (principal.role == PrincipalRole::Agent && principal.id == *owner)
        }
    };
    if allowed {
        Ok(())
    } else {
        Err(MemoryError::PermissionDenied {
            path: path.to_owned(),
        })
    }
}

fn content_hash(content: &str) -> String {
    let digest = Sha256::digest(content.as_bytes());
    format!("sha256:{digest:x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn versions_are_attributed_and_cas_protected() {
        let orchestrator = Principal::orchestrator("orchestrator").unwrap();
        let agent = Principal::agent("agent-a").unwrap();
        let reader = Principal::reader("viewer").unwrap();
        let mut store = MemoryStore::new();
        let first = store
            .write(
                &orchestrator,
                MemoryWrite::new(
                    MemoryScope::Shared,
                    "team/conventions.md",
                    "use rustfmt",
                    "session-1",
                )
                .unwrap(),
            )
            .unwrap();
        assert_eq!(first.version(), 1);
        assert_eq!(first.written_by(), "orchestrator");
        assert!(first.written_at_unix_ms() > 0);
        assert_eq!(store.read(&reader, "team/conventions.md").unwrap(), first);

        let agent_memory = store
            .write(
                &agent,
                MemoryWrite::new(
                    MemoryScope::Agent("agent-a".to_owned()),
                    "notes/last-run.md",
                    "ship passed",
                    "session-2",
                )
                .unwrap(),
            )
            .unwrap();
        assert_eq!(agent_memory.version(), 1);
        assert!(matches!(
            store.read(&reader, "notes/last-run.md"),
            Err(MemoryError::PermissionDenied { .. })
        ));
        assert!(matches!(
            store.write(
                &agent,
                MemoryWrite::new(
                    MemoryScope::Shared,
                    "team/conventions.md",
                    "overwrite",
                    "session-3",
                )
                .unwrap(),
            ),
            Err(MemoryError::PermissionDenied { .. })
        ));
        assert!(matches!(
            store.write(
                &orchestrator,
                MemoryWrite::new(
                    MemoryScope::Agent("agent-a".to_owned()),
                    "team/conventions.md",
                    "wrong scope",
                    "session-3b",
                )
                .unwrap()
                .with_expected_hash(first.content_hash()),
            ),
            Err(MemoryError::ScopeMismatch { .. })
        ));

        let conflict = store.write(
            &orchestrator,
            MemoryWrite::new(
                MemoryScope::Shared,
                "team/conventions.md",
                "new",
                "session-4",
            )
            .unwrap()
            .with_expected_hash("sha256:stale"),
        );
        assert!(matches!(conflict, Err(MemoryError::Conflict { .. })));
        let second = store
            .write(
                &orchestrator,
                MemoryWrite::new(
                    MemoryScope::Shared,
                    "team/conventions.md",
                    "new",
                    "session-4",
                )
                .unwrap()
                .with_expected_hash(first.content_hash()),
            )
            .unwrap();
        assert_eq!(second.version(), 2);
        assert_eq!(
            store.history(&reader, "team/conventions.md").unwrap().len(),
            2
        );
    }

    #[test]
    fn dreaming_clones_then_atomically_commits_curated_sessions() {
        let orchestrator = Principal::orchestrator("orchestrator").unwrap();
        let curator = Principal::agent("agent-a").unwrap();
        let mut store = MemoryStore::new();
        store
            .write(
                &orchestrator,
                MemoryWrite::new(MemoryScope::Shared, "team/context.md", "initial", "seed")
                    .unwrap(),
            )
            .unwrap();
        let mut pass = store.start_dreaming(&orchestrator).unwrap();
        let transcript =
            SessionTranscript::new("agent-a", "session-10", "deploy succeeded").unwrap();
        let applied = pass
            .curate_session(&curator, &transcript, |input, transcript| {
                assert_eq!(
                    input
                        .read(&orchestrator, "team/context.md")
                        .unwrap()
                        .content(),
                    "initial"
                );
                Ok(vec![MemoryWrite::new(
                    MemoryScope::Agent(transcript.agent_id().to_owned()),
                    "notes/deploy.md",
                    transcript.content(),
                    transcript.session_id(),
                )?])
            })
            .unwrap();
        assert_eq!(applied, 1);
        assert!(matches!(
            store.read(&orchestrator, "notes/deploy.md"),
            Err(MemoryError::Missing { .. })
        ));
        pass.commit(&orchestrator, &mut store).unwrap();
        assert_eq!(
            store.read(&curator, "notes/deploy.md").unwrap().content(),
            "deploy succeeded"
        );

        let mut stale_pass = store.start_dreaming(&orchestrator).unwrap();
        let stale_transcript = SessionTranscript::new("agent-a", "session-11", "stale").unwrap();
        stale_pass
            .curate_session(&curator, &stale_transcript, |_, transcript| {
                Ok(vec![MemoryWrite::new(
                    MemoryScope::Agent(transcript.agent_id().to_owned()),
                    "notes/stale.md",
                    transcript.content(),
                    transcript.session_id(),
                )?])
            })
            .unwrap();
        store
            .write(
                &orchestrator,
                MemoryWrite::new(
                    MemoryScope::Shared,
                    "team/context.md",
                    "changed live",
                    "live-session",
                )
                .unwrap()
                .with_expected_hash(
                    store
                        .read(&orchestrator, "team/context.md")
                        .unwrap()
                        .content_hash(),
                ),
            )
            .unwrap();
        assert!(matches!(
            stale_pass.commit(&orchestrator, &mut store),
            Err(MemoryError::Conflict { path, .. }) if path == "<store>"
        ));
    }

    #[test]
    fn rejects_ambiguous_paths_and_invalid_principals() {
        assert!(matches!(
            Principal::agent(""),
            Err(MemoryError::InvalidPrincipal)
        ));
        assert!(matches!(
            MemoryWrite::new(MemoryScope::Shared, "../secret", "x", "s"),
            Err(MemoryError::InvalidPath)
        ));
        assert!(matches!(
            MemoryWrite::new(MemoryScope::Agent(String::new()), "x.md", "x", "s"),
            Err(MemoryError::InvalidScope)
        ));
    }
}
