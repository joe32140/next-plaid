use std::path::PathBuf;

use anyhow::Result;

use crate::commands::search::{resolve_model, resolve_pool_factor};
use colgrep::{
    ensure_model, find_parent_index, index_exists, Config, IndexBuilder, IndexState,
    ParentIndexInfo,
};

pub struct InitOptions<'a> {
    pub cli_model: Option<&'a str>,
    pub no_pool: bool,
    pub pool_factor: Option<usize>,
    pub auto_confirm: bool,
    pub batch_size: Option<usize>,
    pub encode_batch_size: Option<usize>,
    pub index_chunk_size: Option<usize>,
    pub static_batch: bool,
}

fn resolve_index_runtime_overrides(
    config: &Config,
    cli_batch_size: Option<usize>,
) -> (Option<usize>, Option<usize>) {
    (
        config.configured_parallel_sessions(),
        cli_batch_size
            .map(|batch_size| batch_size.max(1))
            .or_else(|| config.configured_batch_size()),
    )
}

/// Whether a parent index holds at least one file under the subdirectory being indexed.
///
/// Reads the parent's state rather than its vectors, so this costs a JSON load and no model.
fn parent_covers_subdir(info: &ParentIndexInfo) -> bool {
    match IndexState::load(&info.index_dir) {
        Ok(state) => state
            .files
            .keys()
            .any(|file| file.starts_with(&info.relative_subdir)),
        // An unreadable parent state is not evidence of coverage: index `path` itself.
        Err(_) => false,
    }
}

pub fn cmd_init(path: &PathBuf, options: InitOptions<'_>) -> Result<()> {
    let path = std::fs::canonicalize(path)
        .map_err(|_| anyhow::anyhow!("Path does not exist: {}", path.display()))?;

    if !path.is_dir() {
        anyhow::bail!("Path is not a directory: {}", path.display());
    }

    let config = Config::load().unwrap_or_default();
    let model = resolve_model(&config, options.cli_model);
    let pool_factor = resolve_pool_factor(&config, options.pool_factor, options.no_pool);

    let quantized = !config.use_fp32();
    let (parallel_sessions, batch_size) =
        resolve_index_runtime_overrides(&config, options.batch_size);

    // Check if path is inside an already-indexed parent project, and reuse that index only
    // if it actually holds files under `path`. A directory the parent's walk skips — most
    // often one its .gitignore excludes — is a subdirectory of the parent without being
    // covered by it, and reusing the parent there would report success having indexed none
    // of the requested files.
    let parent_info = find_parent_index(&path, &model)?.filter(parent_covers_subdir);
    let effective_root = match &parent_info {
        Some(info) => info.project_path.clone(),
        None => path.clone(),
    };

    // Check if index already exists for the effective root
    let has_existing_index = index_exists(&effective_root, &model);

    // Ensure model is downloaded
    let model_path = ensure_model(Some(&model), has_existing_index)?;
    // The repo may ship only one precision; fall back rather than fail at load.
    let quantized = colgrep::resolve_quantized(&model_path, quantized);

    let mut builder = IndexBuilder::with_options(
        &effective_root,
        &model,
        &model_path,
        quantized,
        pool_factor,
        parallel_sessions,
        batch_size,
    )?;
    builder.set_auto_confirm(options.auto_confirm);
    builder.set_dynamic_batch(!options.static_batch);
    if let Some(encode_batch_size) = options.encode_batch_size {
        builder.set_encode_batch_size(encode_batch_size.max(1));
    }
    if let Some(index_chunk_size) = options.index_chunk_size {
        builder.set_index_chunk_size(index_chunk_size.max(1));
    }
    let stats = builder.index(None, false)?;

    let changes = stats.added + stats.changed + stats.deleted;
    if changes > 0 {
        if let Some(ref info) = parent_info {
            eprintln!(
                "Indexed {} (subdir: {}) (added: {}, changed: {}, deleted: {}, unchanged: {})",
                info.project_path.display(),
                info.relative_subdir.display(),
                stats.added,
                stats.changed,
                stats.deleted,
                stats.unchanged,
            );
        } else {
            eprintln!(
                "Indexed {} (added: {}, changed: {}, deleted: {}, unchanged: {})",
                effective_root.display(),
                stats.added,
                stats.changed,
                stats.deleted,
                stats.unchanged,
            );
        }
    } else {
        eprintln!(
            "Index is up to date for {} ({} files)",
            effective_root.display(),
            stats.unchanged
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_index_runtime_overrides_preserves_explicit_values() {
        let config = Config {
            parallel_sessions: Some(3),
            batch_size: Some(7),
            ..Default::default()
        };

        let (parallel_sessions, batch_size) = resolve_index_runtime_overrides(&config, Some(9));

        assert_eq!(parallel_sessions, Some(3));
        assert_eq!(batch_size, Some(9));
    }

    #[test]
    fn test_resolve_index_runtime_overrides_defers_auto_defaults() {
        let config = Config::default();

        let (parallel_sessions, batch_size) = resolve_index_runtime_overrides(&config, None);

        assert_eq!(parallel_sessions, None);
        assert_eq!(batch_size, None);
    }

    #[test]
    fn test_resolve_index_runtime_overrides_normalizes_values() {
        let config = Config {
            parallel_sessions: Some(0),
            batch_size: Some(0),
            ..Default::default()
        };

        let (parallel_sessions, batch_size) = resolve_index_runtime_overrides(&config, Some(0));

        assert_eq!(parallel_sessions, Some(1));
        assert_eq!(batch_size, Some(1));
    }

    fn parent_with_files(files: &[&str]) -> (tempfile::TempDir, ParentIndexInfo) {
        let dir = tempfile::tempdir().unwrap();
        let mut state = IndexState::default();
        for file in files {
            state.files.insert(
                PathBuf::from(file),
                colgrep::FileInfo {
                    content_hash: 0,
                    mtime: 0,
                    size: 0,
                },
            );
        }
        state.save(dir.path()).unwrap();
        let info = ParentIndexInfo {
            index_dir: dir.path().to_path_buf(),
            project_path: PathBuf::from("/project"),
            relative_subdir: PathBuf::from("corpus"),
        };
        (dir, info)
    }

    #[test]
    fn test_parent_covers_subdir_when_it_holds_files_there() {
        let (_dir, info) = parent_with_files(&["src/main.rs", "corpus/doc1.md"]);

        assert!(parent_covers_subdir(&info));
    }

    #[test]
    fn test_parent_does_not_cover_subdir_it_skipped() {
        // The parent walk skipped `corpus/` — e.g. .gitignore excludes it — so an index of
        // the parent holds none of the files `init corpus/` was asked to index.
        let (_dir, info) = parent_with_files(&["src/main.rs", "README.md"]);

        assert!(!parent_covers_subdir(&info));
    }

    #[test]
    fn test_parent_does_not_cover_subdir_when_state_is_unreadable() {
        let dir = tempfile::tempdir().unwrap();
        let info = ParentIndexInfo {
            index_dir: dir.path().join("missing"),
            project_path: PathBuf::from("/project"),
            relative_subdir: PathBuf::from("corpus"),
        };

        assert!(!parent_covers_subdir(&info));
    }
}
