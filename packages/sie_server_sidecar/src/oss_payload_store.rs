//! Native Alibaba OSS payload reader.
//!
//! This mirrors the gateway's bounded native OSS client but exposes only GET.
//! All requests use Signature V4 and an exclusive environment-or-ACK-RRSA
//! credential source that deliberately excludes ECS metadata and file/profile fallbacks.

use std::time::Duration;

use async_trait::async_trait;
use futures_util::StreamExt;
use reqsign_aliyun_oss::{
    AssumeRoleWithOidcCredentialProvider, Credential, EnvCredentialProvider, RequestSigner,
    SigningVersion,
};
use reqsign_core::{Context, OsEnv, ProvideCredential, Signer};
use reqsign_file_read_tokio::TokioFileRead;
use reqsign_http_send_reqwest::ReqwestHttpSend;
use reqwest::{Client, Method, StatusCode};
use url::Url;

use crate::payload_store::{PayloadError, PayloadStore};

const MAX_ATTEMPTS: usize = 3;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_PAYLOAD_BYTES: u64 = crate::prep::media::MAX_OFFLOADED_PAYLOAD_BYTES as u64;

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
    fn parse(value: &str) -> Result<Self, PayloadError> {
        if value.contains('%') {
            return Err(invalid_ref(
                "OSS payload-store URL must not contain percent-encoded components",
            ));
        }
        let rest = value
            .strip_prefix("oss://")
            .ok_or_else(|| invalid_ref("OSS payload-store URL must use oss://"))?;
        let (raw_authority, raw_path) = rest.split_once('/').unwrap_or((rest, ""));
        if !valid_bucket(raw_authority) {
            return Err(invalid_ref("OSS payload-store bucket is invalid or empty"));
        }
        let prefix = validate_raw_prefix(raw_path)?;
        let parsed = Url::parse(value).map_err(|_| invalid_ref("invalid OSS payload-store URL"))?;
        if parsed.scheme() != "oss" {
            return Err(invalid_ref("OSS payload-store URL must use oss://"));
        }
        if !parsed.username().is_empty() || parsed.password().is_some() {
            return Err(invalid_ref(
                "OSS payload-store URL must not contain credentials",
            ));
        }
        if parsed.query().is_some() || parsed.fragment().is_some() || parsed.port().is_some() {
            return Err(invalid_ref(
                "OSS payload-store URL must not contain a port, query, or fragment",
            ));
        }
        let bucket = parsed
            .host_str()
            .filter(|bucket| valid_bucket(bucket))
            .ok_or_else(|| invalid_ref("OSS payload-store bucket is invalid or empty"))?
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

    fn canonical_prefix(&self) -> String {
        if self.prefix.is_empty() {
            format!("oss://{}", self.bucket)
        } else {
            format!("oss://{}/{}", self.bucket, self.prefix)
        }
    }

    fn relative_key<'a>(&self, payload_ref: &'a str) -> Result<&'a str, PayloadError> {
        let key = if payload_ref.starts_with("oss://") {
            let prefix = self.canonical_prefix();
            let rest = payload_ref
                .strip_prefix(&prefix)
                .and_then(|rest| rest.strip_prefix('/'))
                .ok_or_else(|| {
                    invalid_ref("OSS payload reference is outside the configured store")
                })?;
            if rest.starts_with('/') {
                return Err(invalid_ref("OSS payload reference is malformed"));
            }
            rest
        } else if super::payload_store::is_known_object_store_ref(payload_ref) {
            return Err(invalid_ref(
                "payload reference uses a different object-store scheme",
            ));
        } else {
            payload_ref
        };
        validate_payload_key(key)?;
        Ok(key)
    }
}

/// Native OSS V4 GET implementation used by the worker sidecar.
pub struct OssPayloadStore {
    client: Client,
    signer: Signer<Credential>,
    location: OssLocation,
    endpoint: Url,
}

impl OssPayloadStore {
    pub fn from_url(value: &str) -> Result<Self, PayloadError> {
        let location = OssLocation::parse(value)?;
        let region = required_region()?;
        let credential_source = credential_source_from_environment()?;
        let internal = internal_endpoint_enabled()?;
        let suffix = if internal { "-internal" } else { "" };
        let endpoint = Url::parse(&format!(
            "https://{}.oss-{}{}.aliyuncs.com",
            location.bucket, region, suffix
        ))
        .map_err(|_| unsupported("failed to derive OSS endpoint"))?;
        let client = Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .build()
            .map_err(|_| unsupported("failed to build OSS HTTP client"))?;
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

    fn object_url(&self, key: &str) -> Result<Url, PayloadError> {
        let mut url = self.endpoint.clone();
        {
            let mut segments = url
                .path_segments_mut()
                .map_err(|_| object_error("GET", None, None))?;
            segments.clear();
            for segment in self.location.object_key(key).split('/') {
                segments.push(segment);
            }
        }
        Ok(url)
    }

    async fn get_object(&self, key: &str) -> Result<Vec<u8>, PayloadError> {
        'attempts: for attempt in 0..MAX_ATTEMPTS {
            let url = self.object_url(key)?;
            let request = http::Request::builder()
                .method(Method::GET)
                .uri(url.as_str())
                .header(http::header::CONTENT_LENGTH, 0)
                .body(reqwest::Body::from(Vec::new()))
                .map_err(|_| object_error("GET", None, None))?;
            let (mut parts, body) = request.into_parts();
            self.signer
                .sign(&mut parts, None)
                .await
                .map_err(|_| unsupported("OSS GET credential or signing failed"))?;
            let request = http::Request::from_parts(parts, body)
                .try_into()
                .map_err(|_| object_error("GET", None, None))?;

            match self.client.execute(request).await {
                Ok(response) if response.status().is_success() => {
                    if let Some(size) = response.content_length() {
                        if size > MAX_PAYLOAD_BYTES {
                            return Err(PayloadError::TooLarge {
                                actual: size,
                                max: MAX_PAYLOAD_BYTES,
                            });
                        }
                    }
                    let mut stream = response.bytes_stream();
                    let mut bytes = Vec::new();
                    while let Some(chunk) = stream.next().await {
                        let chunk = match chunk {
                            Ok(chunk) => chunk,
                            Err(_) if attempt + 1 < MAX_ATTEMPTS => {
                                retry_delay(attempt).await;
                                continue 'attempts;
                            }
                            Err(_) => {
                                return Err(PayloadError::ObjectStore(
                                    "OSS GET transport failed after retry".to_owned(),
                                ));
                            }
                        };
                        let actual = bytes.len() as u64 + chunk.len() as u64;
                        if actual > MAX_PAYLOAD_BYTES {
                            return Err(PayloadError::TooLarge {
                                actual,
                                max: MAX_PAYLOAD_BYTES,
                            });
                        }
                        bytes.extend_from_slice(&chunk);
                    }
                    return Ok(bytes);
                }
                Ok(response) if response.status() == StatusCode::NOT_FOUND => {
                    return Err(invalid_ref("OSS payload reference was not found"));
                }
                Ok(response) => {
                    let status = response.status();
                    if retryable_status(status) && attempt + 1 < MAX_ATTEMPTS {
                        retry_delay(attempt).await;
                        continue;
                    }
                    let code = provider_error_code(&response);
                    return Err(object_error("GET", Some(status), code.as_deref()));
                }
                Err(_) if attempt + 1 < MAX_ATTEMPTS => retry_delay(attempt).await,
                Err(_) => {
                    return Err(PayloadError::ObjectStore(
                        "OSS GET transport failed after retry".to_owned(),
                    ));
                }
            }
        }
        Err(PayloadError::ObjectStore(
            "OSS GET failed after retry".to_owned(),
        ))
    }
}

#[async_trait]
impl PayloadStore for OssPayloadStore {
    async fn get(&self, payload_ref: &str) -> Result<Vec<u8>, PayloadError> {
        let key = self.location.relative_key(payload_ref)?;
        self.get_object(key).await
    }
}

fn required_region() -> Result<String, PayloadError> {
    let region = std::env::var("SIE_OSS_REGION")
        .map_err(|_| unsupported("SIE_OSS_REGION is required for oss:// payload storage"))?;
    if !valid_region(&region) {
        return Err(unsupported("SIE_OSS_REGION is invalid"));
    }
    Ok(region)
}

fn internal_endpoint_enabled() -> Result<bool, PayloadError> {
    match std::env::var("SIE_OSS_USE_INTERNAL_ENDPOINT") {
        Ok(value) if value.eq_ignore_ascii_case("true") || value == "1" => Ok(true),
        Ok(value) if value.eq_ignore_ascii_case("false") || value == "0" => Ok(false),
        Ok(_) => Err(unsupported(
            "SIE_OSS_USE_INTERNAL_ENDPOINT must be true, false, 1, or 0",
        )),
        Err(std::env::VarError::NotPresent) => Ok(false),
        Err(_) => Err(unsupported("SIE_OSS_USE_INTERNAL_ENDPOINT is invalid")),
    }
}

fn credential_source_from_environment() -> Result<OssCredentialSource, PayloadError> {
    if std::env::var_os("ALIBABA_CLOUD_STS_ENDPOINT").is_some() {
        return Err(unsupported(
            "custom Alibaba STS endpoints are not supported",
        ));
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
        return Err(unsupported(
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
        return Err(unsupported("Alibaba RRSA configuration is incomplete"));
    }
    select_credential_source(alibaba_static || oss_static, oidc).map_err(unsupported)
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

fn validate_pair(first: &str, second: &str) -> Result<bool, PayloadError> {
    let first_present = std::env::var_os(first).is_some();
    let second_present = std::env::var_os(second).is_some();
    let first_set = environment_value_is_nonempty(first);
    let second_set = environment_value_is_nonempty(second);
    if first_present != second_present || first_present != first_set || second_present != second_set
    {
        return Err(unsupported(
            "Alibaba static credential configuration is incomplete",
        ));
    }
    Ok(first_set)
}

fn environment_value_is_nonempty(name: &str) -> bool {
    std::env::var(name).is_ok_and(|value| !value.trim().is_empty())
}

fn validate_payload_key(key: &str) -> Result<(), PayloadError> {
    if key.is_empty()
        || key.contains("..")
        || !key
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
    {
        return Err(invalid_ref("OSS payload key is invalid"));
    }
    Ok(())
}

fn validate_raw_prefix(raw_path: &str) -> Result<String, PayloadError> {
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
        return Err(invalid_ref("OSS payload-store prefix is malformed"));
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

fn object_error(operation: &str, status: Option<StatusCode>, code: Option<&str>) -> PayloadError {
    let mut message = format!("OSS {operation} failed");
    if let Some(status) = status {
        message.push_str(&format!(": status={}", status.as_u16()));
    }
    if let Some(code) = code {
        message.push_str(&format!(" code={code}"));
    }
    PayloadError::ObjectStore(message)
}

fn invalid_ref(message: &str) -> PayloadError {
    PayloadError::InvalidRef(message.to_owned())
}

fn unsupported(message: &str) -> PayloadError {
    PayloadError::Unsupported(message.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqsign_aliyun_oss::StaticCredentialProvider;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use wiremock::matchers::{header_regex, method, path};
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
                ResponseTemplate::new(429)
            } else {
                ResponseTemplate::new(200).set_body_bytes(b"payload")
            }
        }
    }

    #[test]
    fn exact_full_ref_and_plain_key_contract() {
        let location = OssLocation::parse("oss://sie-test-bucket/payloads").unwrap();
        assert_eq!(
            location.relative_key("request_0.bin").unwrap(),
            "request_0.bin"
        );
        assert_eq!(
            location
                .relative_key("oss://sie-test-bucket/payloads/request_0.bin")
                .unwrap(),
            "request_0.bin"
        );
        for payload_ref in [
            "oss://other-bucket/payloads/request_0.bin",
            "oss://sie-test-bucket/other/request_0.bin",
            "s3://sie-test-bucket/payloads/request_0.bin",
            "../request_0.bin",
            "nested/request_0.bin",
        ] {
            assert!(location.relative_key(payload_ref).is_err(), "{payload_ref}");
        }
    }

    #[test]
    fn strict_location_contract() {
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
    fn errors_expose_only_closed_fields() {
        let error = object_error(
            "GET",
            Some(StatusCode::FORBIDDEN),
            clean_error_code("AccessDenied").as_deref(),
        )
        .to_string();
        assert_eq!(
            error,
            "object store: OSS GET failed: status=403 code=AccessDenied"
        );
        for secret in ["Authorization", "SecurityToken", "AccessKey", "RequestId"] {
            assert!(!error.contains(secret));
        }
    }

    #[test]
    fn loads_shared_wire_fixture() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../../wire-fixtures/oss_payload_store.json"))
                .unwrap();
        let configured = format!(
            "oss://{}/{}",
            fixture["bucket"].as_str().unwrap(),
            fixture["prefix"].as_str().unwrap()
        );
        let location = OssLocation::parse(&configured).unwrap();
        let key = fixture["plain_key"].as_str().unwrap();
        assert_eq!(location.object_key(key), fixture["object_key"]);
        assert_eq!(
            location
                .relative_key(fixture["full_reference"].as_str().unwrap())
                .unwrap(),
            key
        );
    }

    #[tokio::test]
    async fn loopback_get_uses_v4_and_retries() {
        let server = MockServer::start().await;
        let attempts = Arc::new(AtomicUsize::new(0));
        Mock::given(method("GET"))
            .and(path("/payloads/request_0.bin"))
            .and(header_regex("authorization", "^OSS4-HMAC-SHA256 "))
            .respond_with(RetryThenOk(attempts.clone()))
            .expect(2)
            .mount(&server)
            .await;

        let store = loopback_store(&server);
        assert_eq!(store.get("request_0.bin").await.unwrap(), b"payload");
        assert_eq!(attempts.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn mid_body_transport_failure_rebuilds_and_resigns() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            for attempt in 0..2 {
                let (mut stream, _) = listener.accept().await.unwrap();
                let mut request = Vec::new();
                loop {
                    let mut chunk = [0; 1024];
                    let read = stream.read(&mut chunk).await.unwrap();
                    if read == 0 {
                        break;
                    }
                    request.extend_from_slice(&chunk[..read]);
                    if request.windows(4).any(|window| window == b"\r\n\r\n") {
                        break;
                    }
                }
                let request = String::from_utf8(request).unwrap().to_ascii_lowercase();
                assert!(request
                    .lines()
                    .any(|line| line.starts_with("authorization: oss4-hmac-sha256 ")));
                if attempt == 0 {
                    stream
                        .write_all(
                            b"HTTP/1.1 200 OK\r\nContent-Length: 7\r\nConnection: close\r\n\r\npay",
                        )
                        .await
                        .unwrap();
                } else {
                    stream
                        .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 7\r\nConnection: close\r\n\r\npayload")
                        .await
                        .unwrap();
                }
                stream.shutdown().await.unwrap();
            }
        });

        let client = Client::builder().timeout(REQUEST_TIMEOUT).build().unwrap();
        let store = OssPayloadStore::new_with_provider(
            OssLocation::parse("oss://sie-test-bucket/payloads").unwrap(),
            "eu-central-1".to_owned(),
            Url::parse(&format!("http://{address}")).unwrap(),
            client,
            StaticCredentialProvider::new("test-access-key", "test-secret-key"),
        );
        assert_eq!(store.get("request_0.bin").await.unwrap(), b"payload");
        server.await.unwrap();
    }

    #[tokio::test]
    async fn loopback_get_enforces_metadata_size_and_404() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/payloads/large.bin"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(vec![
                0;
                MAX_PAYLOAD_BYTES as usize
                    + 1
            ]))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/payloads/missing.bin"))
            .respond_with(ResponseTemplate::new(404))
            .mount(&server)
            .await;

        let store = loopback_store(&server);
        assert!(matches!(
            store.get("large.bin").await,
            Err(PayloadError::TooLarge { .. })
        ));
        assert!(matches!(
            store.get("missing.bin").await,
            Err(PayloadError::InvalidRef(_))
        ));
    }

    #[tokio::test]
    async fn access_denied_error_is_sanitized() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/payloads/denied.bin"))
            .respond_with(
                ResponseTemplate::new(403)
                    .set_body_string("AccessDenied raw-id must never be surfaced"),
            )
            .mount(&server)
            .await;
        let error = loopback_store(&server)
            .get("denied.bin")
            .await
            .unwrap_err()
            .to_string();
        assert_eq!(error, "object store: OSS GET failed: status=403");
        assert!(!error.contains("raw-id"));
    }
}
