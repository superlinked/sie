use std::process::Command;

#[test]
fn help_does_not_print_secret_env_values() {
    let output = Command::new(env!("CARGO_BIN_EXE_sie-server-sidecar"))
        .arg("--help")
        .env("SIE_CONFIG_SERVICE_TOKEN", "config-token-in-env")
        .env("SIE_ADMIN_TOKEN", "admin-token-in-env")
        .env("SIE_GATEWAY_API_KEY", "gateway-key-in-env")
        .output()
        .expect("run sie-server-sidecar --help");
    assert!(output.status.success(), "{output:?}");
    let help = String::from_utf8(output.stdout).expect("utf-8 help");
    assert!(help.contains("SIE_CONFIG_SERVICE_TOKEN"));
    assert!(help.contains("SIE_GATEWAY_API_KEY"));
    assert!(!help.contains("config-token-in-env"));
    assert!(!help.contains("admin-token-in-env"));
    assert!(!help.contains("gateway-key-in-env"));
}
