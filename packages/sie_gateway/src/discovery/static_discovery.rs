pub struct StaticDiscovery {
    urls: Vec<String>,
}

impl StaticDiscovery {
    pub fn new(urls: Vec<String>) -> Self {
        Self { urls }
    }

    pub fn get_worker_urls(&self) -> &[String] {
        &self.urls
    }
}
