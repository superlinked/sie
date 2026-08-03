# n8n pre-loaded with the three DUSK gate webhooks (decision/report/alert),
# active from container start -- no manual UI import step, no external
# service in the workflow itself (each webhook just responds immediately).
FROM n8nio/n8n:2.29.10

COPY dusk-webhooks.json /dusk-webhooks.json
COPY docker-entrypoint.sh /docker-entrypoint-dusk.sh

USER root
RUN chmod +x /docker-entrypoint-dusk.sh && chown node:node /dusk-webhooks.json
USER node

HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=5 \
    CMD ["wget", "-qO-", "http://localhost:5678/healthz"]

ENTRYPOINT ["tini", "--", "/docker-entrypoint-dusk.sh"]
