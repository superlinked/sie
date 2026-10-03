//! Native Alibaba OSS payload writer.
//!
//! OSS is intentionally not routed through `object_store`: its current OSS
//! backend signs with Signature V1. This module always uses region-scoped OSS
//! Signature V4 and a deliberately restricted credential source (either
//! explicit environment credentials or ACK RRSA OIDC, never both). Node metadata, profiles,
//! credential files, URI providers, and generic AssumeRole are omitted.

use std::io;
use std::time::Duration;

use async_trait::async_trait;
use reqsign_aliyun_oss::{
    AssumeRoleWithOidcCredentialProvider, Credential, EnvCredentialProvider, RequestSigner,
    SigningVersion,
};
use reqsign_core::{Context, OsEnv, ProvideCredential, Signer};
use reqsign_file_read_tokio::TokioFileRead;
use reqsign_http_send_reqwest::ReqwestHttpSend;
use reqwest::{Client, Method, StatusCode};
use url::Url;

use super::payload_store::PayloadStore;

const MAX_ATTEMPTS: usize = 3;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Clone, Debug, PartialEq, Eq)]
struct OssLocation {
    bucket: String,
    prefix: String,
}

#[derive(Debug)]
struct OssCredentialProvider {
    source: OssCredentialSource,
}

impl OssCredentialProvider {
    fn new(source: OssCredentialSource) -> Self {
        Self { source }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OssCredentialSource {
    Environment,
    Oidc,
}

impl ProvideCredential for OssCredentialProvider {
    type Credential = Credential;

    async fn provide_credential(
        &self,
        context: &Context,
    ) -> reqsign_core::Result<Option<Self::Credential>> {
        match self.source {
            OssCredentialSource::Environment => {
                EnvCredentialProvider::new()
                    .provide_credential(context)
                    .await
            }
            OssCredentialSource::Oidc => {
                AssumeRoleWithOidcCredentialProvider::new()
                    .provide_credential(context)
                    .await
            }
        }
    }
}

impl OssLocation {
    fn parse(value: &str) -> io::Result<Self> {
        if value.contains('%') {
            return Err(invalid(
                "OSS payload-store URL must not contain percent-encoded components",
            ));
        }
        let rest = value
            .strip_prefix("oss://")
            .ok_or_else(|| invalid("OSS payload-store URL must use oss://"))?;
        let (raw_authority, raw_path) = rest.split_once('/').unwrap_or((rest, ""));
        if !valid_bucket(raw_authority) {
            return Err(invalid("OSS payload-store bucket is invalid or empty"));
        }
        let prefix = validate_raw_prefix(raw_path)?;
        let parsed = Url::parse(value).map_err(|_| invalid("invalid OSS payload-store URL"))?;
        if parsed.scheme() != "oss" {
            return Err(invalid("OSS payload-store URL must use oss://"));
        }
        if !parsed.username().is_empty() || parsed.password().is_some() {
            return Err(invalid(
                "OSS payload-store URL must not contain credentials",
            ));
        }
        if parsed.query().is_some() || parsed.fragment().is_some() || parsed.port().is_some() {
            return Err(invalid(
                "OSS payload-store URL must not contain a port, query, or fragment",
            ));
        }
        let bucket = parsed
            .host_str()
            .filter(|bucket| valid_bucket(bucket))
            .ok_or_else(|| invalid("OSS payload-store bucket is invalid or empty"))?
            .to_owned();
        Ok(Self { bucket, prefix })
    }

    fn object_key(&self, key: &str) -> String {
        if self.prefix.is_empty() {
            key.to_owned()
        } else {
            format!("{}/{key}", self.prefix)
        }
    }

    fn canonical_ref(&self, key: &str) -> String {
        if self.prefix.is_empty() {
            format!("oss://{}/{key}", self.bucket)
        } else {
            format!("oss://{}/{}/{key}", self.bucket, self.prefix)
        }
    }
}

/// Native OSS V4 PUT/DELETE implementation used by the gateway.
pub struct OssPayloadStore {
    client: Client,
    signer: Signer<Credential>,
    location: OssLocation,
    endpoint: Url,
}

impl OssPayloadStore {
    pub fn from_url(value: &str) -> io::Result<Self> {
        let location = OssLocation::parse(value)?;
        let region = required_region()?;
        let credential_source = credential_source_from_environment()?;
        let internal = internal_endpoint_enabled()?;
        let suffix = if internal { "-internal" } else { "" };
        let endpoint = Url::parse(&format!(
            "https://{}.oss-{}{}.aliyuncs.com",
            location.bucket, region, suffix
        ))
        .map_err(|_| invalid("failed to derive OSS endpoint"))?;
        let client = Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .build()
            .map_err(|_| invalid("failed to build OSS HTTP client"))?;
        let provider = OssCredentialProvider::new(credential_source);
        Ok(Self::new_with_provider(
            location, region, endpoint, client, provider,
        ))
    }

    fn new_with_provider(
        location: OssLocation,
        region: String,
        endpoint: Url,
        client: Client,
        provider: impl ProvideCredential<Credential = Credential>,
    ) -> Self {
        let context = Context::new()
            .with_file_read(TokioFileRead)
            .with_http_send(ReqwestHttpSend::new(client.clone()))
            .with_env(OsEnv);
        let request_signer = RequestSigner::new(&location.bucket)
            .with_region(region)
            .with_signing_version(SigningVersion::V4);
        Self {
            client,
            signer: Signer::new(context, provider, request_signer),
            location,
            endpoint,
        }
    }

    fn object_url(&self, key: &str) -> Result<Url, String> {
        let mut url = self.endpoint.clone();
        {
            let mut segments = url
                .path_segments_mut()
                .map_err(|_| "OSS request URL construction failed".to_owned())?;
            segments.clear();
            for segment in self.location.object_key(key).split('/') {
                segments.push(segment);
            }
        }
        Ok(url)
    }

    async fn execute(
        &self,
        operation: &'static str,
        method: Method,
        key: &str,
        data: &[u8],
    ) -> Result<(), String> {
        for attempt in 0..MAX_ATTEMPTS {
            let url = self.object_url(key)?;
            let request = http::Request::builder()
                .method(method.clone())
                .uri(url.as_str())
                .header(http::header::CONTENT_LENGTH, data.len())
                .body(reqwest::Body::from(data.to_vec()))
                .map_err(|_| format!("OSS {operation} request construction failed"))?;
            let (mut parts, body) = request.into_parts();
            self.signer
                .sign(&mut parts, None)
                .await
                .map_err(|_| format!("OSS {operation} credential or signing failed"))?;
            let request = http::Request::from_parts(parts, body)
                .try_into()
                .map_err(|_| format!("OSS {operation} request construction failed"))?;

            match self.client.execute(request).await {
                Ok(response) if response.status().is_success() => return Ok(()),
                Ok(response)
                    if method == Method::DELETE && response.status() == StatusCode::NOT_FOUND =>
                {
                    return Ok(())
                }
                Ok(response) => {
                    let status = response.status();
                    if retryable_status(status) && attempt + 1 < MAX_ATTEMPTS {
                        retry_delay(attempt).await;
                        continue;
                    }
                    let code = provider_error_code(&response);
                    return Err(sanitized_http_error(operation, status, code.as_deref()));
                }
                Err(_) if attempt + 1 < MAX_ATTEMPTS => retry_delay(attempt).await,
                Err(_) => return Err(format!("OSS {operation} transport failed after retry")),
            }
        }
        Err(format!("OSS {operation} failed after retry"))
    }
}

#[async_trait]
impl PayloadStore for OssPayloadStore {
    async fn put(&self, key: &str, data: &[u8]) -> Result<String, String> {
        validate_payload_key(key)?;
        self.execute("PUT", Method::PUT, key, data).await?;
        Ok(self.location.canonical_ref(key))
    }

    async fn delete(&self, key: &str) -> Result<(), String> {
        validate_payload_key(key)?;
        self.execute("DELETE", Method::DELETE, key, &[]).await
    }
}

fn required_region() -> io::Result<String> {
    let region = std::env::var("SIE_OSS_REGION")
        .map_err(|_| invalid("SIE_OSS_REGION is required for oss:// payload storage"))?;
    if !valid_region(&region) {
        return Err(invalid("SIE_OSS_REGION is invalid"));
    }
    Ok(region)
}

fn internal_endpoint_enabled() -> io::Result<bool> {
    match std::env::var("SIE_OSS_USE_INTERNAL_ENDPOINT") {
        Ok(value) if value.eq_ignore_ascii_case("true") || value == "1" => Ok(true),
        Ok(value) if value.eq_ignore_ascii_case("false") || value == "0" => Ok(false),
        Ok(_) => Err(invalid(
            "SIE_OSS_USE_INTERNAL_ENDPOINT must be true, false, 1, or 0",
        )),
        Err(std::env::VarError::NotPresent) => Ok(false),
        Err(_) => Err(invalid("SIE_OSS_USE_INTERNAL_ENDPOINT is invalid")),
    }
}

fn credential_source_from_environment() -> io::Result<OssCredentialSource> {
    if std::env::var_os("ALIBABA_CLOUD_STS_ENDPOINT").is_some() {
        return Err(invalid("custom Alibaba STS endpoints are not supported"));
    }
    let alibaba_static = validate_pair(
        "ALIBABA_CLOUD_ACCESS_KEY_ID",
        "ALIBABA_CLOUD_ACCESS_KEY_SECRET",
    )?;
    let oss_static = validate_pair("OSS_ACCESS_KEY_ID", "OSS_ACCESS_KEY_SECRET")?;
    if optional_token_is_invalid("ALIBABA_CLOUD_SECURITY_TOKEN", alibaba_static)
        || optional_token_is_invalid("OSS_SESSION_TOKEN", oss_static)
        || (alibaba_static && oss_static)
    {
        return Err(invalid(
            "Alibaba static credential configuration is ambiguous or incomplete",
        ));
    }
    let oidc_names = [
        "ALIBABA_CLOUD_ROLE_ARN",
        "ALIBABA_CLOUD_OIDC_PROVIDER_ARN",
        "ALIBABA_CLOUD_OIDC_TOKEN_FILE",
    ];
    let oidc_present = oidc_names.map(|name| std::env::var_os(name).is_some());
    let oidc = oidc_names.map(environment_value_is_nonempty);
    if oidc_present.iter().any(|set| *set) && !oidc.iter().all(|set| *set) {
        return Err(invalid("Alibaba RRSA configuration is incomplete"));
    }
    select_credential_source(alibaba_static || oss_static, oidc).map_err(invalid)
}

fn optional_token_is_invalid(name: &str, base_credentials_configured: bool) -> bool {
    !optional_token_configuration_is_valid(
        base_credentials_configured,
        std::env::var_os(name).is_some(),
        environment_value_is_nonempty(name),
    )
}

fn optional_token_configuration_is_valid(
    base_configured: bool,
    token_present: bool,
    token_nonempty: bool,
) -> bool {
    !token_present || (base_configured && token_nonempty)
}

fn select_credential_source(
    static_configured: bool,
    oidc: [bool; 3],
) -> Result<OssCredentialSource, &'static str> {
    if oidc.iter().any(|set| *set) && !oidc.iter().all(|set| *set) {
        return Err("Alibaba RRSA configuration is incomplete");
    }
    if oidc.iter().all(|set| *set) {
        if static_configured {
            return Err("Alibaba static credentials and RRSA must not be configured together");
        }
        return Ok(OssCredentialSource::Oidc);
    }
    Ok(OssCredentialSource::Environment)
}

fn validate_pair(first: &str, second: &str) -> io::Result<bool> {
    let first_present = std::env::var_os(first).is_some();
    let second_present = std::env::var_os(second).is_some();
    let first_set = environment_value_is_nonempty(first);
    let second_set = environment_value_is_nonempty(second);
    if first_present != second_present || first_present != first_set || second_present != second_set
    {
        return Err(invalid(
            "Alibaba static credential configuration is incomplete",
        ));
    }
    Ok(first_set)
}

fn environment_value_is_nonempty(name: &str) -> bool {
    std::env::var(name).is_ok_and(|value| !value.trim().is_empty())
}

fn validate_payload_key(key: &str) -> Result<(), String> {
    if key.is_empty()
        || key.contains("..")
        || !key
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
    {
        return Err("invalid OSS payload key".to_owned());
    }
    Ok(())
}

fn validate_raw_prefix(raw_path: &str) -> io::Result<String> {
    if raw_path.is_empty() {
        return Ok(String::new());
    }
    if raw_path.starts_with('/')
        || raw_path.contains("//")
        || raw_path.contains('\\')
        || raw_path.split('/').any(|segment| {
            segment == "."
                || segment == ".."
                || !segment
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
        })
    {
        return Err(invalid("OSS payload-store prefix is malformed"));
    }
    Ok(raw_path.strip_suffix('/').unwrap_or(raw_path).to_owned())
}

fn valid_bucket(bucket: &str) -> bool {
    (3..=63).contains(&bucket.len())
        && bucket
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
        && bucket
            .as_bytes()
            .first()
            .is_some_and(u8::is_ascii_alphanumeric)
        && bucket
            .as_bytes()
            .last()
            .is_some_and(u8::is_ascii_alphanumeric)
}

fn valid_region(region: &str) -> bool {
    !region.is_empty()
        && region
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
        && region
            .as_bytes()
            .first()
            .is_some_and(u8::is_ascii_alphanumeric)
        && region
            .as_bytes()
            .last()
            .is_some_and(u8::is_ascii_alphanumeric)
}

fn retryable_status(status: StatusCode) -> bool {
    status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error()
}

async fn retry_delay(attempt: usize) {
    tokio::time::sleep(Duration::from_millis(50 * (attempt as u64 + 1))).await;
}

fn provider_error_code(response: &reqwest::Response) -> Option<String> {
    response
        .headers()
        .get("x-oss-error-code")
        .and_then(|value| value.to_str().ok())
        .and_then(clean_error_code)
}

fn clean_error_code(value: &str) -> Option<String> {
    if value.is_empty()
        || value.len() > 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
    {
        return None;
    }
    Some(value.to_owned())
}

fn sanitized_http_error(operation: &str, status: StatusCode, code: Option<&str>) -> String {
    match code {
        Some(code) => format!(
            "OSS {operation} failed: status={} code={code}",
            status.as_u16()
        ),
        None => format!("OSS {operation} failed: status={}", status.as_u16()),
    }
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqsign_aliyun_oss::StaticCredentialProvider;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use wiremock::matchers::{header_regex, method, path, query_param};
    use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

    fn loopback_store(server: &MockServer) -> OssPayloadStore {
        let client = Client::builder().timeout(REQUEST_TIMEOUT).build().unwrap();
        OssPayloadStore::new_with_provider(
            OssLocation::parse("oss://sie-test-bucket/payloads").unwrap(),
            "eu-central-1".to_owned(),
            Url::parse(&server.uri()).unwrap(),
            client,
            StaticCredentialProvider::new("test-access-key", "test-secret-key"),
        )
    }

    #[derive(Clone)]
    struct RetryThenOk(Arc<AtomicUsize>);

    impl Respond for RetryThenOk {
        fn respond(&self, _request: &Request) -> ResponseTemplate {
            if self.0.fetch_add(1, Ordering::SeqCst) == 0 {
                ResponseTemplate::new(503)
            } else {
                ResponseTemplate::new(200)
            }
        }
    }

    #[test]
    fn strict_location_contract() {
        let location = OssLocation::parse("oss://sie-test-bucket/payloads").unwrap();
        assert_eq!(location.bucket, "sie-test-bucket");
        assert_eq!(location.prefix, "payloads");
        assert_eq!(
            location.canonical_ref("request_0.bin"),
            "oss://sie-test-bucket/payloads/request_0.bin"
        );
        for invalid_value in [
            "oss:///payloads",
            "oss://user@sie-test-bucket/payloads",
            "oss://sie-test-bucket/payloads?x=1",
            "oss://sie-test-bucket/payloads#fragment",
            "oss://sie-test-bucket/a//b",
            "oss://sie-test-bucket/a/%2e%2e/b",
            "oss://sie-test-bucket/a/../b",
            "oss://sie-test-bucket/a/./b",
            "oss://sie-test-bucket//payloads",
            "oss://sie-test-bucket/payloads//",
            "oss://SIE-test-bucket/payloads",
        ] {
            assert!(
                OssLocation::parse(invalid_value).is_err(),
                "{invalid_value}"
            );
        }
    }

    #[test]
    fn rrsa_is_exclusive_and_cannot_be_masked_by_static_credentials() {
        assert_eq!(
            select_credential_source(false, [true, true, true]),
            Ok(OssCredentialSource::Oidc)
        );
        assert!(select_credential_source(true, [true, true, true]).is_err());
        assert!(select_credential_source(false, [true, false, true]).is_err());
        assert_eq!(
            select_credential_source(true, [false, false, false]),
            Ok(OssCredentialSource::Environment)
        );
        assert!(!optional_token_configuration_is_valid(true, true, false));
        assert!(!optional_token_configuration_is_valid(false, true, true));
        assert!(optional_token_configuration_is_valid(true, true, true));
    }

    #[test]
    fn payload_keys_are_plain_and_safe() {
        assert!(validate_payload_key("request_0.bin").is_ok());
        for key in ["", "../x", "a/b", "a\\b", "a?query", "a%2fb"] {
            assert!(validate_payload_key(key).is_err(), "{key}");
        }
    }

    #[test]
    fn errors_expose_only_closed_fields() {
        let error = sanitized_http_error(
            "PUT",
            StatusCode::FORBIDDEN,
            clean_error_code("AccessDenied").as_deref(),
        );
        assert_eq!(error, "OSS PUT failed: status=403 code=AccessDenied");
        for secret in ["Authorization", "SecurityToken", "AccessKey", "RequestId"] {
            assert!(!error.contains(secret));
        }
    }

    #[test]
    fn loads_shared_wire_fixture() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../wire-fixtures/oss_payload_store.json"
        ))
        .unwrap();
        let configured = format!(
            "oss://{}/{}",
            fixture["bucket"].as_str().unwrap(),
            fixture["prefix"].as_str().unwrap()
        );
        let location = OssLocation::parse(&configured).unwrap();
        let key = fixture["plain_key"].as_str().unwrap();
        assert_eq!(location.object_key(key), fixture["object_key"]);
        assert_eq!(location.canonical_ref(key), fixture["full_reference"]);
    }

    #[tokio::test]
    async fn loopback_put_delete_use_v4_and_retry() {
        let server = MockServer::start().await;
        let attempts = Arc::new(AtomicUsize::new(0));
        Mock::given(method("PUT"))
            .and(path("/payloads/request_0.bin"))
            .and(header_regex("authorization", "^OSS4-HMAC-SHA256 "))
            .respond_with(RetryThenOk(attempts.clone()))
            .expect(2)
            .mount(&server)
            .await;
        Mock::given(method("DELETE"))
            .and(path("/payloads/request_0.bin"))
            .and(header_regex("authorization", "^OSS4-HMAC-SHA256 "))
            .respond_with(ResponseTemplate::new(404))
            .expect(1)
            .mount(&server)
            .await;

        let store = loopback_store(&server);
        assert_eq!(
            store.put("request_0.bin", b"payload").await.unwrap(),
            "oss://sie-test-bucket/payloads/request_0.bin"
        );
        store.delete("request_0.bin").await.unwrap();
        assert_eq!(attempts.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn oidc_refresh_rereads_projected_token() {
        let server = MockServer::start().await;
        let directory = tempfile::TempDir::new().unwrap();
        let token_path = directory.path().join("rrsa-token");
        std::fs::write(&token_path, "first-token").unwrap();
        let expiration =
            (reqsign_core::time::Timestamp::now() + Duration::from_secs(60)).format_rfc3339_zulu();
        let response = serde_json::json!({
            "Credentials": {
                "SecurityToken": "temporary-security-token",
                "Expiration": expiration,
                "AccessKeySecret": "temporary-secret",
                "AccessKeyId": "temporary-access-key"
            }
        });
        for token in ["first-token", "second-token"] {
            Mock::given(method("GET"))
                .and(query_param("Action", "AssumeRoleWithOIDC"))
                .and(query_param("OIDCToken", token))
                .respond_with(ResponseTemplate::new(200).set_body_json(&response))
                .expect(1)
                .mount(&server)
                .await;
        }
        Mock::given(method("PUT"))
            .and(path("/payloads/request_0.bin"))
            .and(header_regex("authorization", "^OSS4-HMAC-SHA256 "))
            .respond_with(ResponseTemplate::new(200))
            .expect(2)
            .mount(&server)
            .await;

        let client = Client::builder().timeout(REQUEST_TIMEOUT).build().unwrap();
        let provider = AssumeRoleWithOidcCredentialProvider::new()
            .with_role_arn("acs:ram::000000000000:role/test")
            .with_oidc_provider_arn("acs:ram::000000000000:oidc-provider/test")
            .with_oidc_token_file(token_path.to_string_lossy())
            .with_sts_endpoint(server.uri());
        let store = OssPayloadStore::new_with_provider(
            OssLocation::parse("oss://sie-test-bucket/payloads").unwrap(),
            "eu-central-1".to_owned(),
            Url::parse(&server.uri()).unwrap(),
            client,
            provider,
        );
        store.put("request_0.bin", b"one").await.unwrap();
        std::fs::write(&token_path, "second-token").unwrap();
        store.put("request_0.bin", b"two").await.unwrap();
    }
}
