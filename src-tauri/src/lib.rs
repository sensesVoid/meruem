use tauri::Manager;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            // The Python FastAPI sidecar is spawned by the frontend's
            // lifecycle manager (see frontend/src/lib/sidecar.ts).
            let _ = app;
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![backend_health])
        .run(tauri::generate_context!())
        .expect("error while running Meruem");
}

#[tauri::command]
fn backend_health() -> String {
    "ping".into()
}