use std::ffi::{c_char, c_void, CStr};
use std::path::PathBuf;

use heed::types::Bytes;
use heed::{Database, Env, EnvOpenOptions};

// C ABI structs — must exactly match torchrec tde::details::IOPullParameter / IOPushParameter

#[repr(C)]
pub struct IOPullParameter {
    pub table_name: *const c_char,
    pub num_cols: u32,
    pub num_global_ids: u32,
    pub col_ids: *const i64,
    pub global_ids: *const i64,
    pub num_optimizer_stats: u32,
    pub on_complete_context: *mut c_void,
    pub on_global_id_fetched: Option<
        unsafe extern "C" fn(
            ctx: *mut c_void,
            offset: u32,
            optimizer_state: u32,
            data: *mut c_void,
            data_len: u32,
        ),
    >,
    pub on_all_fetched: Option<unsafe extern "C" fn(ctx: *mut c_void)>,
}

#[repr(C)]
pub struct IOPushParameter {
    pub table_name: *const c_char,
    pub num_cols: u32,
    pub num_global_ids: u32,
    pub col_ids: *const i64,
    pub global_ids: *const i64,
    pub num_optimizer_stats: u32,
    pub optimizer_stats_ids: *const u32,
    pub num_offsets: u32,
    pub offsets: *const u64,
    pub data: *const c_void,
    pub on_complete_context: *mut c_void,
    pub on_push_complete: Option<unsafe extern "C" fn(ctx: *mut c_void)>,
}

struct LmdbBackend {
    env: Env,
    db: Database<Bytes, Bytes>,
}

const DEFAULT_MAP_SIZE: usize = 60 * 1024 * 1024 * 1024;

struct LmdbConfig {
    path: PathBuf,
    map_size: usize,
}

fn parse_size(raw: &str) -> usize {
    let s = raw.trim().to_uppercase();
    let (num_part, mult) = if let Some(rest) = s.strip_suffix("GB") {
        (rest.trim(), 1024usize.pow(3))
    } else if let Some(rest) = s.strip_suffix('G') {
        (rest.trim(), 1024usize.pow(3))
    } else if let Some(rest) = s.strip_suffix("MB") {
        (rest.trim(), 1024usize.pow(2))
    } else if let Some(rest) = s.strip_suffix('M') {
        (rest.trim(), 1024usize.pow(2))
    } else {
        (s.as_str(), 1)
    };
    let num: f64 = num_part.parse().unwrap_or(50.0);
    (num * mult as f64) as usize
}

fn parse_lmdb_config(cfg: &str) -> LmdbConfig {
    let mut path = PathBuf::from(".");
    let mut map_size = DEFAULT_MAP_SIZE;

    for part in cfg.split('&') {
        let part = part.trim();
        if let Some((key, val)) = part.split_once('=') {
            match key.trim() {
                "path" => path = PathBuf::from(val.trim()),
                "map_size" => map_size = parse_size(val.trim()),
                _ => {}
            }
        }
    }

    LmdbConfig { path, map_size }
}

fn make_key(table_name: &str, gid: i64, cid: i64, os_id: u32) -> Vec<u8> {
    let mut buf = Vec::with_capacity(64);
    use std::io::Write;
    let _ = write!(&mut buf, "{}_gid_{}_cid_{}_osid_{}", table_name, gid, cid, os_id);
    buf
}



/// `IO_type` must be a `*const u8` global (NOT a slice/array).
/// The TorchRec plugin loader does:
///   `dlsym(handle, "IO_type")` → address of this static
///   `*reinterpret_cast<const char**>(addr)` → derefs to "lmdb"

static IO_TYPE_BYTES: [u8; 5] = *b"lmdb\0";

#[no_mangle]
pub static mut IO_type: *const u8 = IO_TYPE_BYTES.as_ptr();

/// Returns null on failure.
#[no_mangle]
pub unsafe extern "C" fn IO_Initialize(cfg: *const c_char) -> *mut c_void {
    let cfg_str = match CStr::from_ptr(cfg).to_str() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[lmdb-ps] invalid UTF-8 config: {e}");
            return std::ptr::null_mut();
        }
    };

    let config = parse_lmdb_config(cfg_str);

    if let Err(e) = std::fs::create_dir_all(&config.path) {
        eprintln!("[lmdb-ps] failed to create directory {:?}: {e}", config.path);
        return std::ptr::null_mut();
    }

    let env = match unsafe {
        EnvOpenOptions::new()
            .map_size(config.map_size)
            .max_dbs(1)
            .open(&config.path)
    } {
        Ok(env) => env,
        Err(e) => {
            eprintln!("[lmdb-ps] failed to open LMDB at {:?}: {e}", config.path);
            return std::ptr::null_mut();
        }
    };

    let mut wtxn = match env.write_txn() {
        Ok(txn) => txn,
        Err(e) => {
            eprintln!("[lmdb-ps] failed to begin txn for db creation: {e}");
            return std::ptr::null_mut();
        }
    };
    let db: Database<Bytes, Bytes> = match env.create_database(&mut wtxn, Some("embeddings")) {
        Ok(db) => db,
        Err(e) => {
            eprintln!("[lmdb-ps] failed to create database: {e}");
            return std::ptr::null_mut();
        }
    };
    if let Err(e) = wtxn.commit() {
        eprintln!("[lmdb-ps] failed to commit db creation: {e}");
        return std::ptr::null_mut();
    }

    Box::into_raw(Box::new(LmdbBackend { env, db })) as *mut c_void
}

#[no_mangle]
pub unsafe extern "C" fn IO_Finalize(instance: *mut c_void) {
    if instance.is_null() {
        return;
    }
    let _ = Box::from_raw(instance as *mut LmdbBackend);
}

/// Pull semantics:
///   - on_global_id_fetched(ctx, offset, os_id, data_ptr, data_len) per key
///   - data_ptr == null, data_len == 0  →  miss (matches Redis nil)
///   - on_all_fetched(ctx) after all keys processed
///   - Always calls on_all_fetched even on partial errors.
#[no_mangle]
pub unsafe extern "C" fn IO_Pull(instance: *mut c_void, param: IOPullParameter) {
    let backend = &*(instance as *mut LmdbBackend);

    let table_name = match CStr::from_ptr(param.table_name).to_str() {
        Ok(s) => s,
        Err(_) => {
            if let Some(cb) = param.on_all_fetched {
                cb(param.on_complete_context);
            }
            return;
        }
    };

    let num_cols = if param.num_cols == 0 { 1 } else { param.num_cols };
    let num_global_ids = param.num_global_ids as usize;
    let num_os = param.num_optimizer_stats as usize;
    let ctx = param.on_complete_context;
    let on_fetched = param.on_global_id_fetched;

    let get_col_id = |j: u32| -> i64 {
        if param.num_cols == 0 {
            -1
        } else {
            *param.col_ids.add(j as usize)
        }
    };

    let rtxn = match backend.env.read_txn() {
        Ok(txn) => txn,
        Err(e) => {
            eprintln!("[lmdb-ps] Pull: read txn error: {e}");
            if let Some(cb) = param.on_all_fetched {
                cb(ctx);
            }
            return;
        }
    };

    for i in 0..num_global_ids {
        let gid = *param.global_ids.add(i);
        for j in 0..num_cols {
            let cid = get_col_id(j);
            for os_id in 0..num_os {
                let offset = (j + i as u32 * num_cols) as u32;
                let key = make_key(table_name, gid, cid, os_id as u32);

                match backend.db.get(&rtxn, &key) {
                    Ok(Some(value)) => {
                        if let Some(cb) = on_fetched {
                            cb(ctx, offset, os_id as u32, value.as_ptr() as *mut c_void, value.len() as u32);
                        }
                    }
                    Ok(None) => {
                        if let Some(cb) = on_fetched {
                            cb(ctx, offset, os_id as u32, std::ptr::null_mut(), 0);
                        }
                    }
                    Err(e) => {
                        eprintln!("[lmdb-ps] Pull: db.get error gid={gid}, os_id={os_id}: {e}");
                        if let Some(cb) = on_fetched {
                            cb(ctx, offset, os_id as u32, std::ptr::null_mut(), 0);
                        }
                    }
                }
            }
        }
    }

    drop(rtxn);
    if let Some(cb) = param.on_all_fetched {
        cb(ctx);
    }
}

/// Push semantics:
///   - cover-write (matching Redis HSET)
///   - single LMDB write transaction for the entire batch
///   - on first put error, the transaction is dropped (rollback) — no partial writes
///   - Always calls on_push_complete
#[no_mangle]
pub unsafe extern "C" fn IO_Push(instance: *mut c_void, param: IOPushParameter) {
    let backend = &*(instance as *const LmdbBackend);

    let table_name = match CStr::from_ptr(param.table_name).to_str() {
        Ok(s) => s,
        Err(_) => {
            if let Some(cb) = param.on_push_complete {
                cb(param.on_complete_context);
            }
            return;
        }
    };

    let num_cols = if param.num_cols == 0 { 1 } else { param.num_cols };
    let num_global_ids = param.num_global_ids as usize;
    let num_os = param.num_optimizer_stats as usize;
    let ctx = param.on_complete_context;

    let get_col_id = |j: u32| -> i64 {
        if param.num_cols == 0 {
            -1
        } else {
            *param.col_ids.add(j as usize)
        }
    };

    let mut wtxn = match backend.env.write_txn() {
        Ok(txn) => txn,
        Err(e) => {
            eprintln!("[lmdb-ps] Push: write txn error: {e}");
            if let Some(cb) = param.on_push_complete {
                cb(ctx);
            }
            return;
        }
    };

    let mut failed = false;
    'outer: for i in 0..num_global_ids {
        let gid = *param.global_ids.add(i);
        for j in 0..num_cols {
            let cid = get_col_id(j);
            for k in 0..num_os {
                let os_id = *param.optimizer_stats_ids.add(k);

                let offset_idx = k + j as usize * num_os + i * num_cols as usize * num_os;
                let beg = *param.offsets.add(offset_idx) as usize;
                let end = *param.offsets.add(offset_idx + 1) as usize;

                let data_ptr = (param.data as *const u8).add(beg);
                let value = std::slice::from_raw_parts(data_ptr, end - beg);
                let key = make_key(table_name, gid, cid, os_id);

                if let Err(e) = backend.db.put(&mut wtxn, &key, value) {
                    eprintln!("[lmdb-ps] Push: put error gid={gid}: {e}");
                    failed = true;
                    break 'outer;
                }
            }
        }
    }

    if !failed {
        if let Err(e) = wtxn.commit() {
            eprintln!("[lmdb-ps] Push: commit error: {e}");
        }
    }

    if let Some(cb) = param.on_push_complete {
        cb(ctx);
    }
}
