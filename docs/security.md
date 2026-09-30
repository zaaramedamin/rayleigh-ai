# Security and Privacy

## Principles

1. **Local-first.** No personal document, note or query text is sent to a cloud API. Embeddings and generation run locally. A cloud option, if ever added, is opt-in and clearly flagged.
2. **No blanket filesystem access.** Only allow-listed folders are indexed. The app never silently expands that scope or reads outside it.
3. **Retrieved content is untrusted data.** Documents (and any future tool or web output) are never treated as instructions and cannot grant permissions or trigger actions.
4. **Least privilege.** Any future tool or integration gets the narrowest scope it needs. Sensitive or irreversible actions need explicit confirmation. Destructive actions are denied by default.
5. **Grounded answers.** Citations come from application-controlled metadata. The model never invents a source id or path.
6. **Safe file handling.** Uploaded filenames are metadata only, never paths. Stored files use hashed paths.
7. **No secrets in git.** `.env`, `data/`, `models/` and database files are gitignored from the start.

## Threats to keep in mind

- Path traversal through filenames.
- Prompt injection through document content.
- Accidental indexing of files outside the allow-list.
- Leaking personal data through logs. Logs must not contain document or query text.
